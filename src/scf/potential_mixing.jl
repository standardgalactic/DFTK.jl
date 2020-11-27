# TODO Refactor the code to be more in line with SCF



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

function show_statistics(A)
    λ, X = eigen(A)
    idx = sortperm(λ, by=x -> abs(real(x)))[1:min(20, end)]
    println()
    @show λ[idx]
    display(eigvecs(A)[:, idx])
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

    function get_next(basis, ρₙ, ρspinₙ, Vₙ, PfVₙ)
        n_spin = basis.model.n_spin_components

        Vₙopt = copy(vec(Vₙ))
        PfVₙopt = copy(vec(PfVₙ))
        # Vₙ₊₁ = Vₙ + PfVₙ
        A = nothing
        if !isempty(Vs)
            M = hcat(PfVs...) .- vec(PfVₙ)  # Mᵢⱼ = (PfVⱼ)ᵢ - (PfVₙ)ᵢ
            # We need to solve 0 = M' PfVₙ + M'M βs <=> βs = - (M'M)⁻¹ M' PfVₙ
            βs = -M \ vec(PfVₙ)
            show_statistics(M'M)
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

function anderson_rho(;m=Inf, mode=:diis)
    # TODO Huge code duplication with anderson()
    #
    # Fixed-point map  f(V)  = δF(V) = step(V) - V, where step(V) = (Vext + Vhxc(ρ(V)))
    # SCF update       Pf(V) = α P⁻¹ f(V)
    # SCF map          g(V)  = V + Pf(V)
    #
    # Finds the linear combination Vₙ₊₁ = g(Vₙ) + ∑ᵢ βᵢ (g(Vᵢ) - g(Vₙ))
    # such that (ρₙ - ρᵢ)^† [Pf(Vₙ) + ∑ᵢ βᵢ (Pf(Vᵢ) - Pf(Vₙ))] = 0
    #
    Vs   = []  # The V     for each iteration
    ρs   = []  # The ρ used to build the V
    PfVs = []  # The Pf(V) for each iteration

    function get_next(basis, ρₙ, ρspinₙ, Vₙ, PfVₙ)
        n_spin = basis.model.n_spin_components
        @assert n_spin == 1
        @assert ρspinₙ === nothing

        ρₙ = reshape(ρₙ.real, basis.fft_size..., n_spin)
        ρₙopt = copy(vec(ρₙ))
        Vₙopt = copy(vec(Vₙ))
        PfVₙopt = copy(vec(PfVₙ))
        # Vₙ₊₁ = Vₙ + PfVₙ
        if !isempty(Vs)
            M = hcat(PfVs...) .- vec(PfVₙ)  # Mᵢⱼ = (PfVⱼ)ᵢ - (PfVₙ)ᵢ
            P = hcat(ρs...) .- vec(ρₙ)      # Pᵢⱼ = (ρⱼ)ᵢ   - (ρₙ)ᵢ
            # We need to solve P' PfVₙ + P'M βs = 0
            rhs = - P'vec(PfVₙ)
            A = P'M
            show_statistics(A)
            βs = real.(A \ rhs)
            for (iβ, β) in enumerate(βs)
                Vₙopt   += β * (Vs[iβ] - vec(Vₙ))
                PfVₙopt += β * (PfVs[iβ] - vec(PfVₙ))
                ρₙopt   += β * (ρs[iβ] - vec(ρₙ))
                #Vₙ₊₁ += reshape(β * (Vs[iβ] + PfVs[iβ] - vec(Vₙ) - vec(PfVₙ)),
                #                basis.fft_size..., n_spin)
            end
        end
        if mode == :crop
            error("Untested")
            push!(ρs, vec(ρₙopt))  # TODO or ρₙ or sth else?
            push!(Vs, vec(Vₙopt))
            push!(PfVs, vec(PfVₙopt))
        else
            push!(ρs, vec(ρₙ))
            push!(Vs, vec(Vₙ))
            push!(PfVs, vec(PfVₙ))
        end
        if length(Vs) > m
            Vs = Vs[2:end]
            ρs = ρs[2:end]
            PfVs = PfVs[2:end]
        end
        @assert length(Vs) <= m

        # Vₙ₊₁
        reshape(Vₙopt + PfVₙopt, basis.fft_size..., n_spin)
    end
end


@timing function potential_mixing(basis::PlaneWaveBasis;
                                  n_bands=default_n_bands(basis.model),
                                  ρ=guess_density(basis),
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

    _, ham = energy_hamiltonian(ρ.basis, nothing, nothing; ρ=ρ, ρspin=ρspin)
    V = cat(total_local_potential(ham)..., dims=4)

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
    ΔE_prev_down = [zero(T)]
    info = (ρin=ρ_prev, ρnext=ρ, n_iter=1)
    diagtol = determine_diagtol(info)
    converged = false

    # get_next = anderson(m=10, mode=:crop)
    get_next = anderson_rho(m=Inf, mode=:crop)
    Eprev = Inf
    for i = 1:maxiter
        nextstate = EVρ(V; diagtol=diagtol)
        energies, Vout, ψout, eigenvalues, occupation, εF, ρout, ρ_spin_out = nextstate
        E = energies.total
        Vout = cat(Vout..., dims=4)

        ΔE = E - Eprev
        if abs(ΔE) < tol
            converged = true
            break
        end

        info = (basis=basis, ham=nothing, n_iter=i, energies=energies,
                ψ=ψ, eigenvalues=eigenvalues, occupation=occupation, εF=εF,
                ρout=ρout, ρ_spin_out=ρ_spin_out, ρin=ρ_prev, stage=:iterate,
                diagonalization=nextstate.diagonalization, converged=converged)
        callback(info)

        reject_step = ΔE > 5mean(abs.(ΔE_prev_down[max(begin, end-1):end]))
        if reject_step && i > 1
            # Determine optimal damping
            δV_prev = V - V_prev
            αstep, slope, curv = estimate_optimal_step_size(basis, δF, δV_prev,
                                                            ρ_prev, ρ_spin_prev,
                                                            ρout, ρ_spin_out)

            # E(α) = slope * α + ½ curv * α²
            # println("    rel curv     = ", curv / (dVol*dot(δV_prev, δV_prev)))
            println("      ΔE           = ", ΔE)
            println("      predicted ΔE = ", slope + curv/2)
            println("      αopt         = ", αstep)
            println("      pred αopt ΔE = ", slope * αstep + curv * αstep^2 / 2)
            println()

            V = V_prev + αstep * δV_prev
            continue
        else
            αstep = α
        end

        # Horrible mapping to the density-based SCF to use this function
        diagtol = determine_diagtol((ρin=ρ_prev, ρnext=ρout, n_iter=i + 1))

        # Update state
        ΔE < 0 && i > 1 && (push!(ΔE_prev_down, ΔE))
        Eprev = E
        ψ = ψout
        ρ_prev = ρout
        ρ_spin_prev = ρ_spin_out
        δF = (Vout - V)


        # TODO A bit hackish for now ...
        #      ... the (αstep / mixing.α) is to get rid of the implicit α of the mixing
        info = (ψ=ψ, eigenvalues=eigenvalues, occupation=occupation, εF=εF,
                ρout=ρout, ρ_spin_out=ρ_spin_out, n_iter=i)
        Pinv_δF = (αstep / mixing.α) * mix(mixing, basis, δF; info...)

        # Update V
        V_prev = V
        V = get_next(basis, ρout, ρ_spin_out, V, Pinv_δF)
    end

    Vunpack = [@view V[:, :, :, σ] for σ in 1:n_spin]
    ham = hamiltonian_with_total_potential(ham, Vunpack)
    info = (ham=ham, basis=basis, energies=energies, converged=converged,
            ρ=ρout, ρspin=ρ_spin_out, eigenvalues=eigenvalues, occupation=occupation, εF=εF,
            n_iter=n_iter, n_ep_extra=n_ep_extra, ψ=ψ, stage=:finalize)
    callback(info)
    info
end
