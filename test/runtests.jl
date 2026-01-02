using XSpaceHamiltonians
using Test
using LinearAlgebra: Symmetric

@testset "Basic tests" begin

    @testset "2D tests" begin
        include("2D.jl")
    end
end