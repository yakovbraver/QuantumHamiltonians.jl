module XSpaceHamiltonians

import FFTW
using ArnoldiMethod: partialschur, partialeigen
using LinearAlgebra: Hermitian, Diagonal, diag, diagind, diagview, ldiv!, factorize, eigen, dot
import LinearAlgebra as LA # mainly for the identity operator LA.I
using LinearMaps
import LinearSolve as LS
using SparseArrays
using LDLFactorizations
using FLoops: @floop

export XSpaceHamiltonian, DenseHamiltonian1D, diagonalize!, make_eigenfunction, make_eigenfunctions,
       compute_wanniers!, make_wannierfunctions, make_wanniers_real, compute_tunneling, compute_tb_hamiltonian

include("XSpaceHamiltonian.jl")
include("Wanniers.jl")
include("DenseHamiltonian1D.jl")
include("DenseHamiltonian2D.jl")
include("SparseHamiltonian2D.jl")
include("FourierTransformer.jl")
include("momentum.jl")

end