module XSpaceHamiltonians

import FFTW
using ArnoldiMethod: partialschur
using LinearAlgebra: Hermitian, diagind, diagview, Diagonal, ldiv!, factorize, eigen
import LinearAlgebra as LA # mainly for the identity operator LA.I
using LinearMaps: LinearMap
using SparseArrays
using LDLFactorizations
using FLoops: @floop

export DenseHamiltonian, SparseHamiltonian, diagonalize!, make_wavefunction

include("DenseHamiltonian.jl")
include("SparseHamiltonian.jl")

end