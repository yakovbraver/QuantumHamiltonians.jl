@testset "Test various `fft_to_matrix`" begin
    H_true =
    [11  15  14  51  55  54  41  45  44
     12  11  15  52  51  55  42  41  45
     13  12  11  53  52  51  43  42  41
     21  25  24  11  15  14  51  55  54
     22  21  25  12  11  15  52  51  55
     23  22  21  13  12  11  53  52  51
     31  35  34  21  25  24  11  15  14
     32  31  35  22  21  25  12  11  15
     33  32  31  23  22  21  13  12  11]

    u = [10i+j for i = 1:5, j=1:5]
    H = XSpaceHamiltonians.fft_to_matrix(u)
    @test H == H_true
    
    H = XSpaceHamiltonians.fft_to_matrix_naive!(u)
    @test H == H_true
    
    u = [10i+j for i = 1:5, j=1:5]
    n_elem = XSpaceHamiltonians.filter_count_fft!(u)
    @test n_elem == 81
    H = XSpaceHamiltonians.fft_to_matrix_sparse!(u)
    @test H == H_true

    u = [(10i+j)*iseven(i+j) for i = 1:5, j=1:5]
    n_elem = XSpaceHamiltonians.filter_count_fft!(u)
    @test n_elem == 45

    u = [10i+j for i = 1:3, j=1:5]
    H = XSpaceHamiltonians.rfft_to_matrix!(u)
    @test H == Symmetric(H_true, :L)

    # H_sparse = XSpaceHamiltonians.dft_to_matrix_sparse!(u, make_real=true)
    # @test H_sparse == H_true
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
    @test dh.ε[1] ≈ 2.018 atol=1e-3

    # approximate diagonalisation
    diagonalize!(dh, nev=1)
    @test dh.ε[1] ≈ 2.018 atol=1e-3

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
    @test dh.ε[1] ≈ 0.571 atol=1e-3

    # approximate diagonalisation
    diagonalize!(dh, nev=1)
    @test dh.ε[1] ≈ 0.571 atol=1e-3

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
    @test dh.H isa Matrix{Float32}
        
    # exact diagonalisation
    diagonalize!(dh, nev=0)
    @test dh.ε[1] ≈ -14000 atol=10

    # approximate diagonalisation
    diagonalize!(dh, nev=1)
    @test dh.ε[1] ≈ 1.515 atol=1e-3
    
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
    @test dh.ε[l] ≈ 1.515 - 0.002im atol=1e-3

    # approximate diagonalisation
    diagonalize!(dh, nev=1)
    @test dh.ε[1] ≈ 1.515 - 0.002im atol=1e-3
end