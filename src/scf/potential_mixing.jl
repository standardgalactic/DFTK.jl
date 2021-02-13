# TODO Refactor the code to be more in line with SCF
using Statistics

function RejectStepGuaranteed()
    previously_rejected = true
    function callback(info, αopt, α)
        do_reject = (previously_rejected = !previously_rejected)
        do_reject, αopt
    end
end

function RejectStepEnergyHeuristics(;max_reject=1, reject_tol=1e-8, n_previous=2)
    n_reject = 0
    ΔE_previous = Float64[]
    α_factor = 1.0

    function callback(info, αopt, α)
        relative_energy_change = info.ΔE / abs(info.energies.total)
        avg_ΔE = mean(ΔE_previous[max(begin, end - n_previous + 1):end])
        αopt_in_trusted_region = isnothing(αopt) ? true : 5e-2 < αopt < 3.0
        log_error_predicted = isnothing(info.ΔE_pred) ? -Inf : log10(abs(info.ΔEerror / info.ΔE_pred))

        do_reject = false
        if info.ΔE < reject_tol
            # Never reject if change is small or negative
            do_reject = false
        elseif info.n_iter ≤ 1 && info.ΔE < 1
            # For the first step bigger differences are ok
            do_reject = false
        elseif relative_energy_change > 5e-2
            # This is more for the initial SCF steps
            # (e.g. where Anderson goes in a very stupid direction)
            do_reject = true
            mpi_master() && println("      --> Reject: Relative E change")
        elseif info.ΔE > avg_ΔE
            # This is more for the later SCF steps (e.g. where Kerker gets stuck)
            mpi_master() && println("      --> Reject: Increase beyond average")
            do_reject = true
        end

        if do_reject
            n_reject += 1
            if log_error_predicted > 0 || !αopt_in_trusted_region
                mpi_master() && println("      --> α not trusted: $log_error_predicted  $αopt")
                # The αopt is not trustworthy
                # => Take smaller and smaller fractions of base α
                n_reject > 1 && (α_factor /= 2)
                αopt = α_factor * α
            end
            if n_reject > max_reject
                # Beyond maximal number of rejects (to avoid infinite reject loops)
                # => just ignore the reject
                do_reject = false
            end
        end
        if !do_reject
            info.ΔE < 0 && push!(ΔE_previous, abs(info.ΔE))
            n_reject = 0
        end

        do_reject, αopt
        # Old version
        # if info.ΔE < reject_tol
        #     push!(ΔE_previous, abs(info.ΔE))
        #     n_reject = 0
        #     α_factor = 1
        #     return false, αopt  # Never reject if change is small or negative
        # end

        # if info.n_iter ≤ 1 && info.ΔE < 1
        #     n_reject = 0
        #     α_factor = 1
        #     return false, αopt  # For the first step bigger differences are ok
        # end

        # relative_energy_change = info.ΔE / abs(info.energies.total)
        # if relative_energy_change > 5e-2 && n_reject < max_reject
        #     n_reject += 1
        #     return true, αopt
        # end

        # avg_ΔE = mean(ΔE_previous[max(begin, end - n_previous + 1):end])
        # if info.ΔE > avg_ΔE && n_reject < max_reject
        #     n_reject += 1
        #     return true, αopt
        # else
        #     n_reject = 0
        #     α_factor = 1
        #     return false, αopt
        # end
    end
end


function estimate_optimal_step_size(basis, δF, δV, ρout, ρ_spin_out, ρnext, ρ_spin_next)
    # δF = F(V_out) - F(V_in)
    # δV = V_next - V_in
    # δρ = ρ(V_next) - ρ(V_in)
    dVol = basis.model.unit_cell_volume / prod(basis.fft_size)
    n_spin = basis.model.n_spin_components

    δρ = (ρnext - ρout).real
    if !isnothing(ρ_spin_out)
        δρspin = (ρ_spin_next - ρ_spin_out).real
        δρ_RFA     = from_real(basis, δρ)
        δρspin_RFA = from_real(basis, δρspin)

        δρα = (δρ + δρspin) / 2
        δρβ = (δρ - δρspin) / 2
        δρ = cat(δρα, δρβ, dims=4)
    else
        δρ_RFA = from_real(basis, δρ)
        δρspin_RFA = nothing
        δρ = reshape(δρ, basis.fft_size..., 1)
    end

    slope = dVol * dot(δF, δρ)
    Kδρ = apply_kernel(basis, δρ_RFA, δρspin_RFA; ρ=ρout, ρspin=ρ_spin_out)
    if n_spin == 1
        Kδρ = reshape(Kδρ[1].real, basis.fft_size..., 1)
    else
        Kδρ = cat(Kδρ[1].real, Kδρ[2].real, dims=4)
    end

    curv = dVol*(-dot(δV, δρ) + dot(δρ, Kδρ))
    # curv = abs(curv)  # Not sure we should explicitly do this

    # E = slope * t + 1/2 curv * t^2
    αopt = -slope/curv

    αopt, slope, curv
end

function show_statistics(A, dVdV)
    λ, X = eigen(A)
    λdV, XdV = eigen(A, dVdV)
    idx = sortperm(λ, by=x -> abs(real(x)))[1:min(20, end)]
    println()
    @show λ[idx]
    @show λdV
    # display(eigvecs(A)[:, idx])
    println()
end


function anderson(;m=Inf, mode=:diis)
    # Fixed-point map  f(V)  = δF(V) = step(V) - V, where step(V) = (Vext + Vhxc(ρ(V)))
    # SCF update       Pf(V) = α P⁻¹ f(V)
    # SCF map          g(V)  = V + Pf(V)
    #
    # Finds the linear combination Vₙ₊₁ = g(Vₙ) + ∑ᵢ βᵢ (g(Vᵢ) - g(Vₙ))
    # such that |Pf(Vₙ) + ∑ᵢ βᵢ (Pf(Vᵢ) - Pf(Vₙ))|² is minimal
    #
    Vs   = []  # The V     for each iteration
    PfVs = []  # The Pf(V) for each iteration

    function get_next(basis, Vₙ, PfVₙ)
        n_spin = basis.model.n_spin_components

        Vₙopt = copy(vec(Vₙ))
        PfVₙopt = copy(vec(PfVₙ))
        # Vₙ₊₁ = Vₙ + PfVₙ
        A = nothing
        if !isempty(Vs)
            M = hcat(PfVs...) .- vec(PfVₙ)  # Mᵢⱼ = (PfVⱼ)ᵢ - (PfVₙ)ᵢ
            # We need to solve 0 = M' PfVₙ + M'M βs <=> βs = - (M'M)⁻¹ M' PfVₙ
            βs = -M \ vec(PfVₙ)
            dV = hcat(Vs...) .- vec(Vₙ)
            # show_statistics(M'M, dV'dV)
            for (iβ, β) in enumerate(βs)
                Vₙopt += β * (Vs[iβ] - vec(Vₙ))
                PfVₙopt += β * (PfVs[iβ] - vec(PfVₙ))
                # Vₙ₊₁ += reshape(β * (Vs[iβ] + PfVs[iβ] - vec(Vₙ) - vec(PfVₙ)),
                #                 basis.fft_size..., n_spin)
            end
        end
        if mode == :crop
            push!(Vs, vec(Vₙopt))
            push!(PfVs, vec(PfVₙopt))
        else
            push!(Vs, vec(Vₙ))
            push!(PfVs, vec(PfVₙ))
        end
        if length(Vs) > m
            Vs = Vs[2:end]
            PfVs = PfVs[2:end]
        end
        @assert length(Vs) <= m

        # Vₙ₊₁
        reshape(Vₙopt + PfVₙopt, basis.fft_size..., n_spin)
    end
end


using Plots
function plot_along_line(EVρ, E_prev, V_prev, δV, αopt, slope, curv)
    println("        -> Running plot")
    αs = append!([αopt], 0.1:0.2:1.5)
    Es = [EVρ(V_prev + α * δV)[1].total for α in αs] .- E_prev
    p = scatter(αs[2:end], Es[2:end], color=1, m=:x, ms=4, label="computed", legend=:topleft)
    p = scatter!(p, [αs[1]], [Es[1]], color=2, m=:+, ms=4, label="optimal")

    model(α) = slope * α + curv * α^2 / 2
    rel_error = log10(abs((Es[1] - model(αs[1])) / Es[1]))
    αfine = min(αopt, 0.0):0.05:max(αopt, 1.5)
    plot!(p, αfine, model.(αfine), color=1, label="model ($(@sprintf "%5.3f" rel_error)")
end


@timing function potential_mixing(basis::PlaneWaveBasis;
                                  n_bands=default_n_bands(basis.model),
                                  ρ=guess_density(basis),
                                  V=nothing,
                                  ρspin=guess_spin_density(basis),
                                  ψ=nothing,
                                  tol=1e-6,
                                  maxiter=100,
                                  solver=scf_nlsolve_solver(),
                                  eigensolver=lobpcg_hyper,
                                  n_ep_extra=3,
                                  determine_diagtol=ScfDiagtol(),
                                  mixing=SimpleMixing(),
                                  is_converged=ScfConvergenceEnergy(tol),
                                  callback=ScfDefaultCallback(),
                                  compute_consistent_energies=true,
                                  m=Inf,
                                  reject_step=RejectStepEnergyHeuristics(),
                                  use_guaranteed=false,
                                  # if use_guaranteed:
                                  plotprefix=nothing,
                                  α_trial=1.0,
                                  α_min=0.05,
                                  α_max=1.5,
                                  )
    T = eltype(basis)
    model = basis.model

    # All these variables will get updated by fixpoint_map
    if ψ !== nothing
        @assert length(ψ) == length(basis.kpoints)
        for ik in 1:length(basis.kpoints)
            @assert size(ψ[ik], 2) == n_bands + n_ep_extra
        end
    end
    occupation = nothing
    eigenvalues = nothing
    εF = nothing
    n_iter = 0
    energies = nothing
    ham = nothing
    n_spin = basis.model.n_spin_components
    ρout = ρ
    ρ_spin_out = ρspin

    energies, ham = energy_hamiltonian(ρ.basis, nothing, nothing; ρ=ρ, ρspin=ρspin)
    if isnothing(V)
        V = cat(total_local_potential(ham)..., dims=4)
    end
    dVol = model.unit_cell_volume / prod(basis.fft_size)

    function EVρ(V; diagtol=tol / 10)
        Vunpack = [@view V[:, :, :, σ] for σ in 1:n_spin]
        ham_V = hamiltonian_with_total_potential(ham, Vunpack)
        res_V = next_density(ham_V; n_bands=n_bands,
                             ψ=ψ, n_ep_extra=3, miniter=1, tol=diagtol)
        new_E, new_ham = energy_hamiltonian(basis, res_V.ψ, res_V.occupation;
                                            ρ=res_V.ρout, ρspin=res_V.ρ_spin_out,
                                            eigenvalues=res_V.eigenvalues, εF=res_V.εF)
        (energies=new_E, Vout=total_local_potential(new_ham), res_V...)
    end

    α = mixing.α
    δF = nothing
    V_prev = V
    ρ_prev = ρ
    ρ_spin_prev = ρspin
    info = (ρin=ρ_prev, ρnext=ρ, n_iter=1)
    diagtol = determine_diagtol(info)
    converged = false
    ΔE_pred = nothing
    ΔEerror = Inf
    αopt = nothing
    slope = nothing
    curv = nothing

    # TODO In a first phase (i.e. if we are so far outside the linear regime that
    #      the estimate_optimal_step_size is just not working, we need to use an
    #      even simpler approach (i.e. use the α suggested by the user and half it
    #      in case of a reject.

    get_next = anderson(m=m)
    Eprev = energies.total
    for i = 1:maxiter
        nextstate = EVρ(V; diagtol=diagtol)
        energies, Vout, ψout, eigenvalues, occupation, εF, ρout, ρ_spin_out = nextstate
        E = energies.total
        Vout = cat(Vout..., dims=4)

        ΔE = E - Eprev
        if abs(ΔE) < tol && i > 1
            converged = true
            break
        end

        # Determine optimal damping for the step just taken along with the estimates
        # for the slope and curvature along the search direction just explored
        if !isnothing(δF) && !use_guaranteed
            δV_prev = V - V_prev
            αopt, slope, curv = estimate_optimal_step_size(basis, δF, δV_prev,
                                                           ρ_prev, ρ_spin_prev,
                                                           ρout, ρ_spin_out)
            ΔE_pred = slope + curv * α^2 / 2
        end
        if !isnothing(ΔE_pred)
            ΔEerror = abs(ΔE - ΔE_pred)
            if mpi_master()
                println("      αopt         = ", αopt)
                println("      ΔE           = ", ΔE)
                println("      predicted ΔE = ", ΔE_pred)
                println("      ΔE abs. err. = ", ΔEerror)
                println("      ΔE rel. err. = ", log10(abs(ΔEerror / ΔE)))
            end
        end

        info = (basis=basis, ham=nothing, n_iter=i, energies=energies,
                ψ=ψ, eigenvalues=eigenvalues, occupation=occupation, εF=εF,
                ρout=ρout, ρ_spin_out=ρ_spin_out, ρin=ρ_prev, stage=:iterate,
                diagonalization=nextstate.diagonalization, converged=converged,
                αopt=αopt, ΔEerror=ΔEerror, ΔE=ΔE, ΔE_pred=ΔE_pred, V=V, V_prev=V_prev)
        callback(info)

        if use_guaranteed
            do_reject = false
            if ΔE > 1e-6 && !isnothing(αopt)
                do_reject = true
                αopt /= 2
            end
        else
            do_reject, αopt = reject_step(info, αopt, mixing.α)
        end
        if do_reject && !isnothing(αopt)
            ΔE_pred = slope * αopt + curv * αopt^2 / 2
            if mpi_master()
                println("      --> reject step <--")
                println("      αopt (adj)   = ", αopt)
                println("      pred αopt ΔE = ", ΔE_pred)
                println()
            end
            V = V_prev + αopt * (V - V_prev)
            continue  # Do not commit the new state
        end
        mpi_master() && println()

        # Horrible mapping to the density-based SCF to use this function
        diagtol = determine_diagtol((ρin=ρ_prev, ρnext=ρout, n_iter=i + 1))

        # Update state
        Eprev = E
        ψ = ψout
        δF = (Vout - V)
        ρ_prev = ρout
        ρ_spin_prev = ρ_spin_out

        # TODO A bit hackish for now ...
        #      ... the (α / mixing.α) is to get rid of the implicit α of the mixing
        info = (ψ=ψ, eigenvalues=eigenvalues, occupation=occupation, εF=εF,
                ρout=ρout, ρ_spin_out=ρ_spin_out, n_iter=i)
        if use_guaranteed
            prefac = 1 / mixing.α  # To get rid of the implicit α of the mixing
        else
            prefac = (α / mixing.α)
        end
        Pinv_δF = prefac * mix(mixing, basis, δF; info...)

        # Update V
        V_prev = V
        if use_guaranteed
            # Get the next step by running Anderson
            δV = get_next(basis, V_prev, Pinv_δF) - V_prev

            # How far along the search direction defined by δV do we want to go
            nextstate = EVρ(V_prev + α_trial * δV; diagtol=diagtol)
            ρnext, ρ_spin_next = nextstate.ρout, nextstate.ρ_spin_out
            αopt, slope, curv = estimate_optimal_step_size(basis, δF, α_trial * δV,
                                                           ρout, ρ_spin_out,
                                                           ρnext, ρ_spin_next)

            αopt = max(α_min, αopt)  # Empirical constants
            αopt = min(αopt, α_max)

            ΔE_pred = slope * αopt + curv * αopt^2 / 2
            if !isnothing(plotprefix) && (i < 11)
                p = plot_along_line(EVρ, Eprev, V_prev, δV, αopt, slope, curv)
                savefig(p, plotprefix * ".quadratic.$i.pdf")
            end
            V = V_prev + αopt * δV
        else
            # Just use Anderson
            V = get_next(basis, V_prev, Pinv_δF)
        end
    end

    Vunpack = [@view V[:, :, :, σ] for σ in 1:n_spin]
    ham = hamiltonian_with_total_potential(ham, Vunpack)
    info = (ham=ham, basis=basis, energies=energies, converged=converged,
            ρ=ρout, ρspin=ρ_spin_out, eigenvalues=eigenvalues, occupation=occupation, εF=εF,
            n_iter=n_iter, n_ep_extra=n_ep_extra, ψ=ψ, stage=:finalize)
    callback(info)
    info
end


function potmix_quadratic_model(basis, α0, Vin, Vout, Vnext, ρin, ρ_spin_in, ρnext, ρ_spin_next)
    # Vout    = step(V), where step(V) = (Vext + Vhxc(ρ(V)))
    # Vnext   = Vin + α0 * (Anderson(Vin, P⁻¹( Vout - Vin )) - Vin)
    # ρin     = ρ(Vin)
    # ρnext   = ρ(Vnext)
    # α0 * δV = Vnext - Vin = α0 * (Anderson(Vin, P⁻¹( Vout - Vin )) - Vin)
    # δρ      = ρnext - ρin
    #
    # We build a quadratic model for
    #   ϕ(α) = E(Vin  + α δV)  at α = 0
    #        = E(Vin) + α ∇E|_(V=Vin) ⋅ δV + ½ α^2 <δV | ∇²E|_(V=Vin) | δV>
    #        = E(Vin) + (α/α0) ∇E|_(V=Vin) ⋅ (α0 * δV) + ½ (α/α0)² <α0 * δV | ∇²E|_(V=Vin) | α0 * δV>
    #
    # Now
    #      ∇E|_(V=Vin)  = - χ₀(Vout - Vin)
    #      ∇²E|_(V=Vin) ≃ - χ₀ (1 - K χ₀)        (only true if Vin is an SCF minimum, K taken at Vin)
    # and therefore using the self-adjointness of χ₀
    #      ∇E|_(V=Vin) ⋅ δV         = -(Vout - Vin) ⋅ χ₀(δV) = - (Vout - Vin) ⋅ δρ
    #      <δV | ∇²E|_(V=Vin) | δV> = - δV ⋅ δρ + δρ ⋅ K(δρ)
    #

    dVol = basis.model.unit_cell_volume / prod(basis.fft_size)
    n_spin = basis.model.n_spin_components

    δV = Vnext - Vin
    δF = Vout  - Vin

    δρ = (ρnext - ρin).real
    if !isnothing(ρ_spin_in)
        δρspin = (ρ_spin_next - ρ_spin_in).real
        δρ_RFA     = from_real(basis, δρ)
        δρspin_RFA = from_real(basis, δρspin)

        δρα = (δρ + δρspin) / 2
        δρβ = (δρ - δρspin) / 2
        δρ = cat(δρα, δρβ, dims=4)
    else
        δρ_RFA = from_real(basis, δρ)
        δρspin_RFA = nothing
        δρ = reshape(δρ, basis.fft_size..., 1)
    end

    slope = dVol * dot(δF, δρ) / α0
    Kδρ = apply_kernel(basis, δρ_RFA, δρspin_RFA; ρ=ρin, ρspin=ρ_spin_in)
    if n_spin == 1
        Kδρ = reshape(Kδρ[1].real, basis.fft_size..., 1)
    else
        Kδρ = cat(Kδρ[1].real, Kδρ[2].real, dims=4)
    end
    curv = dVol * (-dot(δV, δρ) + dot(δρ, Kδρ)) / α0^2

    slope, curv
end


@timing function potential_mixing_guaranteed(basis::PlaneWaveBasis;
                                             n_bands=default_n_bands(basis.model),
                                             ρ=guess_density(basis), V=nothing,
                                             ρspin=guess_spin_density(basis),
                                             ψ=nothing, tol=1e-6, maxiter=100,
                                             eigensolver=lobpcg_hyper,
                                             n_ep_extra=3,
                                             determine_diagtol=ScfDiagtol(),
                                             mixing=SimpleMixing(α=1.0),
                                             is_converged=ScfConvergenceEnergy(tol),
                                             callback=ScfDefaultCallback(),
                                             m=10, α_trial=mixing.α, α_min=1 / 32,
                                             # For debugging and to get "standard" algo
                                             always_accept=false
                                            )
    @assert mixing.α == 1.0  # Otherwise weird issues ...

    model  = basis.model
    n_spin = basis.model.n_spin_components
    dVol   = model.unit_cell_volume / prod(basis.fft_size)

    # Initial guess for V and ψ (if none given)
    if ψ !== nothing
        @assert length(ψ) == length(basis.kpoints)
        for ik in 1:length(basis.kpoints)
            @assert size(ψ[ik], 2) == n_bands + n_ep_extra
        end
    end
    energies, ham = energy_hamiltonian(basis, nothing, nothing; ρ=ρ, ρspin=ρspin)
    isnothing(V) && (V = cat(total_local_potential(ham)..., dims=4))


    function EVρ(V; diagtol=tol / 10, ψ=nothing)
        Vunpack = [@view V[:, :, :, σ] for σ in 1:n_spin]
        ham_V = hamiltonian_with_total_potential(ham, Vunpack)
        res_V = next_density(ham_V; n_bands=n_bands,
                             ψ=ψ, n_ep_extra=n_ep_extra, miniter=1, tol=diagtol)
        new_E, new_ham = energy_hamiltonian(basis, res_V.ψ, res_V.occupation;
                                            ρ=res_V.ρout, ρspin=res_V.ρ_spin_out,
                                            eigenvalues=res_V.eigenvalues, εF=res_V.εF)
        (energies=new_E, Vout=total_local_potential(new_ham), res_V...)
    end

    # Initialise iteration state
    # All quantitities without any _out or _next specifier are derived from V == Vin
    ρ_prev    = ρ
    n_iter    = 0
    converged = false
    diagtol   = determine_diagtol((ρin=ρ_prev, n_iter=n_iter))
    state     = EVρ(V; diagtol=diagtol, ψ=ψ)

    get_next = anderson(m=m)
    for i = 1:maxiter
        n_iter = i
        energies, Vout, ψ, eigenvalues, occupation, εF, ρ, ρ_spin = state
        Etotal = energies.total
        Vout   = cat(Vout..., dims=4)
        δF     = Vout - V

        # TODO A bit hackish for now with the ρout = ρ, ρin=ρ_prev stuff ...
        info = (basis=basis, ham=nothing, n_iter=i, energies=energies,
                ψ=ψ, eigenvalues=eigenvalues, occupation=occupation, εF=εF,
                ρout=ρ, ρ_spin_out=ρ_spin, ρin=ρ_prev, ρnext=ρ, stage=:iterate,
                diagonalization=state.diagonalization, converged=converged)
        callback(info)
        if is_converged(info)
            converged = true
            break
        end

        # XXX mixing contains an implicit α at the moment
        Pinv_δF = mix(mixing, basis, δF; info...) / mixing.α
        δV      = get_next(basis, V, Pinv_δF) - V

        # Determine stepsize and take next step
        α = α_trial
        guess = ψ
        diagtol = determine_diagtol(merge(info, (n_iter=i + 1, )))
        while true
            Vnext = V + α * δV
            state_next  = EVρ(Vnext; diagtol=diagtol, ψ=guess)
            Etotal_next = state_next.energies.total
            ρnext       = state_next.ρout
            ρ_spin_next = state_next.ρ_spin_out

            diagiter = mpi_mean(mean(state_next.diagonalization.iterations), basis.comm_kpts)
            if mpi_master()
                println("    α = $α")
                println("        ΔE        = $(Etotal_next - Etotal)   diag = $diagiter")
            end

            if Etotal_next - Etotal < 50tol || α ≤ α_min || always_accept
                # Accept any energy-decreasing step (or if α is already too small)
                state  = state_next
                ρ_prev = ρ
                V = Vnext
                mpi_master() && println()
                break
            end

            slope, curv = potmix_quadratic_model(basis, α, V, Vout, Vnext, ρ, ρ_spin, ρnext, ρ_spin_next)

            Emodel(α) = Etotal + slope * α + curv * α^2 / 2
            model_relerror = abs(Etotal_next - Emodel(α)) / abs(Etotal_next - Etotal)

            if mpi_master()
                println("        ΔE_pred   = $(Emodel(α) - Etotal)")
                println("        relerror  = $model_relerror")
                println("        slope     = $slope    (<0)")
                println("        curv      = $curv     (>0)")
                println("        αopt      = $(-slope / curv)")
                println("        ΔE_next   = $(Emodel(-slope / curv) - Etotal)")
            end

            modeltol = 0.1  # Relative error in the model, which is acceptable
            reject_model = (
                   curv < 0  # Otherwise stationary point is a maximum
                || model_relerror > modeltol  # Model not trustworthy
                || (slope > 0 && model_relerror > 0.1modeltol)  # Uphill slope not trusted
            )
            if reject_model
                mpi_master() && println("        ---> Rejecting model")
                α_next = α / 2
            elseif slope > 0
                mpi_master() && println("        ---> Uphill slope")
                α_next = α_min
            else
                # Use model to get optimal damping
                α_next = -slope / curv
                α_next = min(0.95α, -slope / curv)  # to avoid getting stuck
            end
            α_next = max(α_next, α_min)  # Don't undershoot

            # Adjust guess: Use whatever state is closest
            guess = α_next > α / 2 ? state_next.ψ : ψ
            α = α_next
        end
    end

    Vunpack = [@view V[:, :, :, σ] for σ in 1:n_spin]
    ham  = hamiltonian_with_total_potential(ham, Vunpack)
    info = (ham=ham, basis=basis, energies=energies, converged=converged,
            ρ=ρ, ρspin=ρspin, eigenvalues=state.eigenvalues, occupation=state.occupation,
            εF=state.εF, n_iter=n_iter, n_ep_extra=n_ep_extra, ψ=ψ, stage=:finalize)
    callback(info)
    info
end
