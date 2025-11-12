module XSpaceHamiltonians

import FFTW
using ArnoldiMethod: partialschur
using LinearAlgebra: Symmetric, Hermitian, diagind, diagview, Diagonal, ldiv!, factorize, eigen
import LinearAlgebra as LA # mainly for the identity operator LA.I
using LinearMaps: LinearMap
using SparseArrays
using LDLFactorizations
using FLoops: @floop

export DenseHamiltonian, DenseHamiltonian1D, SparseHamiltonian, diagonalize!, make_wavefunction

include("DenseHamiltonian.jl")
include("DenseHamiltonian1D.jl")
include("SparseHamiltonian.jl")

end