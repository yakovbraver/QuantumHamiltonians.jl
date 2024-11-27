module XSpaceHamiltonians

import FFTW
using ArnoldiMethod: partialschur
using LinearAlgebra: Hermitian, diagind, Diagonal, ldiv!, factorize, eigen
using LinearMaps: LinearMap
using SparseArrays
using LDLFactorizations

export DirichletHamiltonian, diagonalize!, make_wavefunction, PeriodicHamiltonian

include("DirichletHamiltonian.jl")
include("PeriodicHamiltonian.jl")

end