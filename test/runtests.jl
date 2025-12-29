using XSpaceHamiltonians
using Test
using LinearAlgebra: Symmetric

@testset "Basic tests" begin

    @testset "DenseHamiltonian tests" begin
        include("DenseHamiltonian_tests.jl")
    end
end