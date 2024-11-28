module XSpaceHamiltonians

import FFTW
using ArnoldiMethod: partialschur
using LinearAlgebra: Hermitian, diagind, Diagonal, ldiv!, factorize, eigen
using LinearMaps: LinearMap
using SparseArrays
using LDLFactorizations
using FLoops: @floop

export DenseHamiltonian, diagonalize!, make_wavefunction

include("DenseHamiltonian.jl")

end