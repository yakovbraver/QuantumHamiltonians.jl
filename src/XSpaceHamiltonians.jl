module XSpaceHamiltonians

import FFTW
using ArnoldiMethod: partialschur, partialeigen
using LinearAlgebra: Hermitian, Diagonal, diag, diagind, diagview, factorize, eigen, ldiv!, dot, mul!
import LinearAlgebra as LA # mainly for the identity operator LA.I
using LinearMaps
import LinearSolve as LS
using SparseArrays
using FLoops: @floop

export XSpaceHamiltonian, diagonalize!, make_eigenfunction, make_eigenfunctions, matrix_density,
       compute_wanniers!, make_wannierfunctions, make_wanniers_real, compute_tunneling, compute_tb_hamiltonian

include("XSpaceHamiltonian.jl")
include("Wanniers.jl")
include("DenseHamiltonian.jl")
include("SparseHamiltonian2D.jl")
include("FourierTransformer.jl")
include("momentum.jl")

end