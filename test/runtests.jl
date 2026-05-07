using QuantumHamiltonians
using Test
using LinearAlgebra: Symmetric

@testset "1D tests" begin
    include("1D.jl")
end

@testset "2D tests" begin
    include("2D.jl")
end