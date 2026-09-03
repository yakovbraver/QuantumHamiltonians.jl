module QuantumHamiltonians

import FFTW
import OrdinaryDiffEq as ODE
import OrdinaryDiffEqLinear as ODE_LIN
import OrdinaryDiffEqExponentialRK as ODE_EXP
import SciMLOperators
import KrylovKit
using ArnoldiMethod: partialschur, partialeigen
using LinearAlgebra: Hermitian, Diagonal, diag, diagind, diagview, factorize, eigen, ldiv!, dot, mul!, normalize!, copy_adjoint!
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

include("gpe_stationary.jl")
include("gpe_dynamics_pspace.jl")
include("gpe_dynamics_xspace.jl")
include("BdGMaps.jl")

end
