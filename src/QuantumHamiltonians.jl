module QuantumHamiltonians

import FFTW
import OrdinaryDiffEq as ODE
import OrdinaryDiffEqLinear as ODE_LIN
import OrdinaryDiffEqExponentialRK as ODE_EXP
import SciMLOperators
import KrylovKit
using ArnoldiMethod: partialschur, partialeigen
using LinearAlgebra: Hermitian, Diagonal, diag, diagind, diagview, factorize, eigen, dot, mul!, normalize!, copy_adjoint!
import LinearAlgebra: ldiv! # overloaded in XSpaceHamiltonians for preconditioning
import LinearAlgebra as LA # mainly for the identity operator LA.I
import LinearMaps as LM
import LinearSolve as LS
import NonlinearSolve as NLS
using SparseArrays
using FLoops: @floop
using LoopVectorization: @turbo

export PSpaceHamiltonian, diagonalize!, make_eigenfunction, make_eigenfunctions, matrix_density, make_wavefunction,
       compute_wanniers!, make_wannierfunctions, make_wanniers_real, compute_tunneling, compute_tb_hamiltonian,
       propagate, get_EμN, bdg_spectrum, find_stationary
export StateVector, XSpaceHamiltonian

include("PSpaceHamiltonian.jl")
include("Wanniers.jl")
include("FourierTransformerP.jl")
include("DenseHamiltonian.jl")
include("SparseHamiltonian.jl")
include("momentum.jl")

include("StateVector.jl")
include("FourierTransformerX.jl")
include("XSpaceHamiltonian.jl")
include("XSpacePreconditioners.jl")

include("gpe_stationary.jl")
include("gpe_dynamics_pspace.jl")
include("gpe_dynamics_xspace.jl")
include("BdGMaps.jl")

"A linear map holding a `LinearSolve` object, used for applying the inverse map."
struct LinSolveLinMap{T,L} <: LM.LinearMap{T}
    linsolve::L
    size::Dims{2}
end

Base.size(lm::LinSolveLinMap) = lm.size

function LM._unsafe_mul!(y, lm::LinSolveLinMap, x::AbstractVector)
    copy!(lm.linsolve.b, x)
    sol = LS.solve!(lm.linsolve) # `solve!` allocates up to 50 KiB :(
    copy!(y, sol.u)
    # println(sol.stats) # TODO add verbose option
end

end
