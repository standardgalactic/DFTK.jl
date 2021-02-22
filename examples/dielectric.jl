# Compute a few eigenvalues of the dielectric matrix (q=0,ω=0) iteratively

using DFTK
using Plots
using KrylovKit
using Printf

# Calculation parameters
kgrid = [1, 1, 1]
Ecut = 5

# Silicon lattice
a = 10.26
lattice = a / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))
atoms = [Si => [ones(3)/8, -ones(3)/8]]

# change the symmetry to compute the dielectric operator with and without symmetries
model = model_LDA(lattice, atoms, symmetries=false)
basis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)
scfres = self_consistent_field(basis, tol=1e-14)

# Apply ε† = 1 - χ0 (vc + fxc)
function eps_fun(dρ)
    dρ = reshape(dρ, size(scfres.ρ))
    dv = apply_kernel(basis, dρ; ρ=scfres.ρ)
    χdv = apply_χ0(scfres.ham, scfres.ψ, scfres.εF, scfres.eigenvalues, dv)
    vec(dρ - χdv)
end

# eager diagonalizes the subspace matrix at each iteration
eigsolve(eps_fun, vec(randn(size(scfres.ρ))), 5, :LM; eager=true, verbosity=3)
