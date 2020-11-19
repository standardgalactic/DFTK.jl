# Change by implementing a heuristics that does the extra EρV only when needed.
# Test it a bit
# Refactor the code to be more in line with SCF

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
    curv = abs(curv)  # Not sure we should explicitly do this

    # E = slope * t + 1/2 curv * t^2
    αopt = -slope/curv

    αopt, slope, curv
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
    V0 = cat(total_local_potential(ham)..., dims=4)

    V = V0
    Vprev = V
    α = 1.0

    dVol = model.unit_cell_volume / prod(basis.fft_size)

    function EρV(V)
        Vunpack = [@view V[:, :, :, σ] for σ in 1:n_spin]
        ham_V = hamiltonian_with_total_potential(ham, Vunpack)
        res_V = next_density(ham_V; n_bands=n_bands,
                             ψ=ψ, n_ep_extra=3, miniter=1, tol=tol / 10)
        new_E, new_ham = energy_hamiltonian(basis, res_V.ψ, res_V.occupation;
                                            ρ=res_V.ρout, ρspin=res_V.ρ_spin_out,
                                            eigenvalues=res_V.eigenvalues, εF=res_V.εF)
        ψ = res_V.ψ
        # println(res_V.eigenvalues[1][5] - res_V.eigenvalues[1][4])
        new_E.total, res_V.ρout, res_V.ρ_spin_out, total_local_potential(new_ham)
    end

    Vs = []
    δFs = []
    Eprev = Inf
    for i = 1:maxiter
        E, ρout, ρ_spin_out, GV = EρV(V)
        GV = cat(GV..., dims=4)
        println("ΔE this step:         = ", E - Eprev)
        if !isnothing(ρ_spin_out)
            println("Magnet                  ", sum(ρ_spin_out.real) * dVol)
        end
        Eprev = E
        δF = GV - V

        # generate new direction δV from history
        function weight(dV)  # Precondition with Kerker
            dVr = copy(reshape(dV, basis.fft_size..., n_spin))
            Gsq = [sum(abs2, model.recip_lattice * G) for G in G_vectors(basis)]
            w = (Gsq .+ 1) ./ (Gsq)
            w[1] = 1
            # for σ in 1:n_spin
            #     dVr[:, :, :, σ] = from_fourier(basis, w .* from_real(basis, dVr[:, :, :, σ]).fourier).real
            # end
            dV
        end
        δV = δF
        if !isempty(Vs)
            mat = hcat(δFs...) .- vec(δF)
            mat = mapslices(weight, mat; dims=[1])
            alphas = -mat \ weight(vec(δF))
            # alphas = -(mat'mat) * mat' * vec(δF)
            for iα = 1:length(Vs)
                δV += reshape(alphas[iα] * (Vs[iα] + δFs[iα] - vec(V) - vec(δF)), basis.fft_size..., n_spin)
            end
        end
        push!(Vs, vec(V))
        push!(δFs, vec(δF))

        # The SCF step
        new_V = V + δV

        # Optimal step size
        new_E, ρnext, ρ_spin_next, _ = EρV(new_V)

        ΔE = new_E - E
        abs(ΔE) < tol && break

        αopt, slope, curv = estimate_optimal_step_size(basis, δF, δV, ρout, ρ_spin_out, ρnext, ρ_spin_next)

        println("Step $i")
        println("rel curv: ", curv / (dVol*dot(δV, δV)))

        # E = slope * t + 1/2 curv * t^2
        # αopt = -slope/curv
        ΔEopt = -1/2*slope^2 / curv

        println("SimpleSCF actual   ΔE = ", ΔE)
        println("SimpleSCF pred     ΔE = ", slope + curv/2)
        # println("Opt       actual      = ", EρV(V + αopt*(new_V - V))[1] - E)
        println("Opt       pred     ΔE = ", ΔEopt)
        println("αopt                  = ", αopt)
        println()

        V = V + αopt*(new_V - V)
        # V = V + (new_V - V)
    end

    Vunpack = [@view V[:, :, :, σ] for σ in 1:n_spin]
    ham = hamiltonian_with_total_potential(ham, Vunpack)
    info = (ham=ham, basis=basis, energies=energies, converged=converged,
            ρ=ρout, ρspin=ρ_spin_out, eigenvalues=eigenvalues, occupation=occupation, εF=εF,
            n_iter=n_iter, n_ep_extra=n_ep_extra, ψ=ψ)
    info
end
