using QuantumHamiltonians
using Test

# ~ 1 min
@testset "1D tests" begin
    include("1D.jl")
end

# ~ 35 s
@testset "2D tests" begin
    include("2D.jl")
end

# ~ 3 sec
@testset "x-space preconditioner tests" begin
    include("preconditioners.jl")
end
