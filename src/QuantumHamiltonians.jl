module QuantumHamiltonians

import FFTW
import OrdinaryDiffEq as ODE
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
       propagate, get_Eμη, bdg_spectrum, find_stationary
export StateVector, XSpaceHamiltonian

include("PSpaceHamiltonian.jl")
include("Wanniers.jl")
include("DenseHamiltonian.jl")
include("SparseHamiltonian.jl")
include("FourierTransformerP.jl")
include("momentum.jl")

include("pspace_gpe_dynamics.jl")
include("pspace_gpe_stationary.jl")
include("BdGMaps.jl")

include("StateVector.jl")
include("FourierTransformerX.jl")
include("XSpaceHamiltonian.jl")
include("XSpaceGPEDynamics.jl")

end