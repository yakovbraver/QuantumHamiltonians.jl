module XSpaceHamiltonians

import FFTW
using ArnoldiMethod: partialschur
using LinearAlgebra: Hermitian, diagind, Diagonal, ldiv!, factorize, eigen
using LinearMaps: LinearMap
using SparseArrays
using LDLFactorizations
using FLoops: @floop

export DenseHamiltonian, SparseHamiltonian, diagonalize!, make_wavefunction

include("DenseHamiltonian.jl")
include("SparseHamiltonian.jl")

end