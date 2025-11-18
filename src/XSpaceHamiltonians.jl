module XSpaceHamiltonians

import FFTW
using ArnoldiMethod: partialschur
using LinearAlgebra: Symmetric, Hermitian, Diagonal, diag, diagind, diagview, ldiv!, factorize, eigen, schur, dot
import LinearAlgebra as LA # mainly for the identity operator LA.I
using LinearMaps: LinearMap
using SparseArrays
using LDLFactorizations
using FLoops: @floop

export DenseHamiltonian, DenseHamiltonian1D, SparseHamiltonian, diagonalize!, make_eigenfunction, make_eigenfunctions, compute_wanniers!, make_wannierfunctions, make_wanniers_real, compute_tunneling

include("DenseHamiltonian.jl")
include("DenseHamiltonian1D.jl")
include("SparseHamiltonian.jl")

end