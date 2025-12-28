@testset "Test `dft_to_matrix`" begin
    u = [collect(0:7)'; collect(10:17)'; collect(20:27)']
    
    H = XSpaceHamiltonians.dft_to_matrix(u, make_real=true)
    H_true =
    [  0   1   2   3   4  10  11  12  13  14  20  21  22  23  24
       1   0   1   2   3  17  10  11  12  13  27  20  21  22  23
       2   1   0   1   2  16  17  10  11  12  26  27  20  21  22
       3   2   1   0   1  15  16  17  10  11  25  26  27  20  21
       4   3   2   1   0  14  15  16  17  10  24  25  26  27  20
      10  17  16  15  14   0   1   2   3   4  10  11  12  13  14
      11  10  17  16  15   1   0   1   2   3  17  10  11  12  13
      12  11  10  17  16   2   1   0   1   2  16  17  10  11  12
      13  12  11  10  17   3   2   1   0   1  15  16  17  10  11
      14  13  12  11  10   4   3   2   1   0  14  15  16  17  10
      20  27  26  25  24  10  17  16  15  14   0   1   2   3   4
      21  20  27  26  25  11  10  17  16  15   1   0   1   2   3
      22  21  20  27  26  12  11  10  17  16   2   1   0   1   2
      23  22  21  20  27  13  12  11  10  17   3   2   1   0   1
      24  23  22  21  20  14  13  12  11  10   4   3   2   1   0]
    @test H == H_true
end

# @testset "Test that FFT of `𝑈` is real and even" begin
#     ϵ = 0.1 # testing for Float64
#     ϵc = 1
#     χ = 0
#     gf = GaugeField(ϵ, ϵc, χ; n_harmonics=10, fft_threshold=0.05)

#     L = π # periodicity of the potential
#     M = 20
#     dx = L / 2M
#     x = range(0, L-dx, 2M)
#     U = 𝑈(x, x; ϵ, ϵc, χ) .* (dx/L)^2
#     u = rfft(U)
#     @test sum(abs.(imag.(u))) < 1e-10 # test that imaginary part is zero

#     @views s = u[1:M, 1:M]
#     @test sum(abs.(s - transpose(s))) < 1e-10 # test that matrix is symmetric
# end

# TODO add additional type checks
# Analysis of https://doi.org/10.1103/PhysRevA.107.033328 (https://arxiv.org/abs/2304.00302)
@testset "Test 2D 1-component diagonalisation" begin
    function 𝑈(x::Real, y::Real)
        (sin(x+y)^2 + (ϵc*sin(x-y))^2) / 𝛼(x, y)^2 * 2ϵ^2 * (1+ϵc^2)
    end

    function 𝐴_x(x::Real, y::Real)
        sin(2y) .* ϵc .* sin(χ) ./ 𝛼(x, y)
    end

    function 𝐴_y(x::Real, y::Real)
        sin(2x) .* ϵc .* sin(χ) ./ 𝛼(x, y)
    end

    function 𝛼(x::Real, y::Real)
        η₋ = cos(x-y); η₊ = cos(x+y)
        return ϵ^2 * (1 + ϵc^2) + η₊^2 + (ϵc*η₋)^2 - 2ϵc*η₊*η₋*cos(χ)
    end

    ϵ::Float32 = 0.1
    ϵc::Float32 = 1

    ########## χ = 0

    χ::Float32 = 0

    xlimits = (0, π) .|> Float32
    ylimits = (0, π) .|> Float32

    M = 5

    ### Nonperiodic
    dh = DenseHamiltonian(xlimits, ylimits; isperiodic=false, M, 𝐻=[𝑈;;])
    @test dh.H isa Matrix{Float32}
    @test dh.V isa Matrix{Float32}

    # exact diagonalisation
    diagonalize!(dh, nev=0)
    @test dh.ε[1] ≈ 2.064 atol=1e-3

    # approximate diagonalisation
    diagonalize!(dh, nev=1)
    @test dh.ε[1] ≈ 2.064 atol=1e-3

    ### Periodic
    dh = DenseHamiltonian(xlimits, ylimits; isperiodic=true, M, 𝐻=[𝑈;;], 𝐻_iseven = [true;;])
    @test dh.H isa Matrix{Float32}

    # exact diagonalisation
    diagonalize!(dh, nev=0)
    @test dh.ε[1] ≈ 2.022 atol=1e-3

    # approximate diagonalisation
    diagonalize!(dh, nev=1)
    @test dh.ε[1] ≈ 2.022 atol=1e-3

    ########## χ = π/2, x from -π/2 to π/2

    χ = π/2

    xlimits = (-π/2, π/2) .|> Float32
    ylimits = (0, π) .|> Float32

    ### Nonperiodic
    dh = DenseHamiltonian(xlimits, ylimits; isperiodic=false, M, 𝐻=[𝑈;;], 𝐴_x, 𝐴_y)
    
    # exact diagonalisation
    diagonalize!(dh, nev=0)
    @test dh.ε[1] ≈ 3.179 atol=1e-3

    # approximate diagonalisation
    diagonalize!(dh, nev=1)
    @test dh.ε[1] ≈ 3.179 atol=1e-3

    ########## χ = π/2, full period

    χ = π/2

    xlimits = (0, 2π) .|> Float32
    ylimits = (0, 2π) .|> Float32

    dh = DenseHamiltonian(xlimits, ylimits; isperiodic=true, M, 𝐻=[𝑈;;], 𝐻_iseven = [true;;], 𝐴_x, 𝐴_y);

    # exact diagonalisation
    diagonalize!(dh, nev=0)
    @test dh.ε[1] ≈ 0.591 atol=1e-3

    # approximate diagonalisation
    diagonalize!(dh, nev=1)
    @test dh.ε[1] ≈ 0.591 atol=1e-3

end

@testset "Test 2D 3-component diagonalisation" begin
    function 𝛺₁(x::Real, y::Real)
        Ω₁₀
    end

    function 𝛺₂(x::Real, y::Real)
        - Ω₋ * cos(x-y) + Ω₊ * cos(x+y) 
        # - Ω₋ * cis(χ/2) * cos(x-y) + Ω₊ * cis(-χ/2) * cos(x+y) 
    end

    ϵ::Float32 = 0.1
    ϵc::Float32 = 1
    Ω₁₀::Float32 = 2000
    Ω₊ = Ω₁₀ / (2ϵ*√(1+ϵc^2)) # include division by 2 here
    Ω₋ = Ω₊ * ϵc
    # χ::Float32 = 0
    Γ₃::Float32 = 1e3

    R::Float32 = π
    xlimits = (0, R) .|> Float32
    ylimits = (0, R) .|> Float32

    M = 5
    𝐻 = [nothing nothing 𝛺₁      
         nothing nothing 𝛺₂
         nothing nothing nothing] # only lower triangle is needed
    𝐻_iseven = BitArray([0 0 1; 0 0 1; 0 0 0])

    ### Hermitian nonperiodic diagonalisation
    dh = DenseHamiltonian(xlimits, ylimits; isperiodic=false, M, 𝐻)
    @test dh.H isa Matrix{Float32}
        
    # exact diagonalisation
    diagonalize!(dh, nev=0)
    @test dh.ε[1] ≈ -13542 atol=10

    # approximate diagonalisation
    diagonalize!(dh, nev=1)
    @test dh.ε[1] ≈ 2.086 atol=1e-3 # approximate diagonalisation finds lowest-magnitude eigenvalue

    ### Hermitian periodic diagonalisation
    dh = DenseHamiltonian(xlimits, ylimits; isperiodic=true, M, 𝐻, 𝐻_iseven)
    @test dh.H isa Matrix{Complex{Float32}} # complex because the Fourier images of 𝛺 might be complex
        
    # exact diagonalisation
    diagonalize!(dh, nev=0)
    @test dh.ε[1] ≈ -14000 atol=10

    # approximate diagonalisation
    diagonalize!(dh, nev=1)
    @test dh.ε[1] ≈ 1.530 atol=1e-3
    
    ### non-Hermitian nonperiodic diagonalisation
    dh = DenseHamiltonian(xlimits, ylimits; isperiodic=false, M, 𝐻, Γ=[0, 0, Γ₃]);
    @test dh.H isa Matrix{Complex{Float32}}
    
    # exact diagonalisation
    diagonalize!(dh, nev=0)
    @test dh.ε[1] ≈ -13540 - 250im atol=10
    l = findfirst(x -> real(x) > 0, dh.ε)
    @test l == 26
    @test dh.ε[l] ≈ 2.088 atol=1e-3

    # approximate diagonalisation
    diagonalize!(dh, nev=1)
    @test dh.ε[1] ≈ 2.086 atol=1e-3

    ### non-Hermitian periodic diagonalisation
    dh = DenseHamiltonian(xlimits, ylimits; isperiodic=true, M, 𝐻, 𝐻_iseven, Γ=[0, 0, Γ₃]);
    @test dh.H isa Matrix{Complex{Float32}}
    
    # exact diagonalisation
    diagonalize!(dh, nev=0)
    @test dh.ε[1] ≈ -14000 - 250im atol=10
    l = findfirst(x -> real(x) > 0, dh.ε)
    @test l == 122
    @test dh.ε[l] ≈ 1.531 - 0.002im atol=1e-3

    # approximate diagonalisation
    diagonalize!(dh, nev=1)
    @test dh.ε[1] ≈ 1.531 - 0.002im atol=1e-3
end