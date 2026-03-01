module XSpaceHamiltonians

import FFTW
import OrdinaryDiffEq as ODE
import SciMLOperators
using ArnoldiMethod: partialschur, partialeigen
using LinearAlgebra: Hermitian, Diagonal, diag, diagind, diagview, factorize, eigen, ldiv!, dot, mul!, normalize!, copy_adjoint!
import LinearAlgebra as LA # mainly for the identity operator LA.I
import LinearMaps as LM
import LinearSolve as LS
import NonlinearSolve as NLS
using SparseArrays
using FLoops: @floop
using LoopVectorization: @turbo

export XSpaceHamiltonian, diagonalize!, make_eigenfunction, make_eigenfunctions, matrix_density, make_wavefunction,
       compute_wanniers!, make_wannierfunctions, make_wanniers_real, compute_tunneling, compute_tb_hamiltonian,
       propagate, get_E_μ, bdg_spectrum, find_stationary

include("XSpaceHamiltonian.jl")
include("Wanniers.jl")
include("DenseHamiltonian.jl")
include("SparseHamiltonian.jl")
include("FourierTransformer.jl")
include("momentum.jl")
include("BdGMaps.jl")

end