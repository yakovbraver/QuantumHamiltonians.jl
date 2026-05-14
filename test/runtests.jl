using QuantumHamiltonians
using Test

# ~ 2 min
@testset "1D tests" begin
    include("1D.jl")
end

# ~ 1 min
@testset "2D tests" begin
    include("2D.jl")
end