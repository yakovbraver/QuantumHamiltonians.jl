using XSpaceHamiltonians
using Test

@testset "Basic tests" begin

    @testset "DenseHamiltonian tests" begin
        include("DenseHamiltonian_tests.jl")
    end
end