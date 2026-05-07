@testset "Test various `fft_to_operator`" begin
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
    
    # we need a `FourierTransformerP` object to test `fft_to_operator_2D!`
    u = [10j+i for i = 1:5, j = 1:5]
    M = 1
    ft = QuantumHamiltonians.FourierTransformerP([(0.0, 1.0), (0.0, 1.0)], M; basis=:cis)
    ft.buff .= u # set as if `u` was the result of an actual fft
    H = similar(H_true)
    QuantumHamiltonians.fft_to_operator_2D!(H, ft)
    @test H == H_true
    
    ### legacy functions 

    # u = [10i+j for i = 1:5, j=1:5]
    # H = QuantumHamiltonians._fft_to_operator(u)
    # @test H == H_true

    # u = [10i+j for i = 1:5, j=1:5]
    # n_elem = QuantumHamiltonians.filter_count_fft!(u)
    # @test n_elem == 81
    # H = QuantumHamiltonians.fft_to_operator_sparse!(u)
    # @test H == H_true

    # u = [(10i+j)*iseven(i+j) for i = 1:5, j=1:5]
    # n_elem = QuantumHamiltonians.filter_count_fft!(u)
    # @test n_elem == 45

    # u = [10i+j for i = 1:3, j=1:5]
    # H = QuantumHamiltonians._rfft_to_operator!(u)
    # @test H == Symmetric(H_true, :L)
end

@testset "1-component diagonalisation" begin
    # Assymetric quantum harmonic oscillator: 𝐻 = -Δ/2 + 𝑥²/2 + 3𝑦²/2

    ω₁ = 1; ω₂ = 3;
    𝑈(x, y) = ω₁^2 * x^2 / 2 + ω₂^2 * y^2 / 2
    xlimits = (-5, 5) .|> Float64
    stateno = 2 # will test 2nd eigenstate (=4th excited state) 
    𝜓₁₀(x, y) = (ω₁/π)^(1/4) / √2 * exp(-ω₁*x^2/2) * 2x*√ω₁ * (ω₂/π)^(1/4) * exp(-ω₂*y^2/2)

    for basis in (:cis, :sin, :cos), kind in (:dense, :sparse, :xspace)
        M = basis == :sin ? 31 : 32

        if kind == :xspace
            qh = XSpaceHamiltonian([xlimits, xlimits], 𝑈; basis, M, δ=√0.5) # `√` because `δ` is the coefficient of ∂ₓ, not Δ
            vals, vecs, info = diagonalize(qh; nev=5)
            xs = qh.ft.xs
            ψ = vecs[stateno][1]
            ε = vals[1:4]
        else
            (kind == :sparse && basis != :cis) && continue # sparse is only implemented for cis
            kwargs = kind == :sparse ? (;fft_threshold=1e-2) : (;) # pass `fft_threshold` for sparse; otherwise pass an empty named tuple
            qh = PSpaceHamiltonian{kind}([xlimits, xlimits], 𝑈; basis, 𝑈_iseven=true, M, δ=√0.5, kwargs...)
            # test correctness of kind parameters
            @test qh isa PSpaceHamiltonian{kind, Float64, Float64, Float64, 3, 4}
            diagonalize!(qh, nev=4)
            xs, vec = make_eigenfunction(qh, stateno)
            ψ = vec[1]
            ε = qh.ε
        end
        
        @test ε ≈ 2:5 rtol=1e-5 # exact spectrum is εˣʸ = (𝑛ˣ + 1/2) + 3(𝑛ʸ + 1/2); lowest energies are 2, 3, 4, 5, 5, 6

        ψ_true = 𝜓₁₀.(xs[:, 1], xs[:, 2]')
        @test all(@. abs(ψ) - abs(ψ_true) < 1e-2) # test abs because a sign difference is possible
    end
end

# TODO add additional type checks
@testset "Test dense and sparse 2D 1-component diagonalisation" begin
    # Tests based on the system in https://doi.org/10.1103/PhysRevA.107.033328 (https://arxiv.org/abs/2304.00302)

    function 𝑈(x::Real, y::Real)
        (sin(x+y)^2 + (ϵc*sin(x-y))^2) / 𝛼(x, y)^2 * 2ϵ^2 * (1+ϵc^2)
    end

    function 𝐴ˣ(x::Real, y::Real)
        sin(2y) .* ϵc .* sin(χ) ./ 𝛼(x, y)
    end

    function 𝐴ʸ(x::Real, y::Real)
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

    ### sin basis
    qh = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:sin, M)
    @test qh.H isa Matrix{Float32}
    @test qh.V isa Matrix{Float32}

    # exact diagonalisation
    diagonalize!(qh, nev=0)
    @test qh.ε[1] ≈ 2.064 rtol=1e-3

    # approximate diagonalisation
    diagonalize!(qh, nev=1)
    @test qh.ε[1] ≈ 2.064 rtol=1e-3

    ### cis basis
    qh = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=true)
    @test qh.H isa Matrix{Float32}

    # exact diagonalisation
    diagonalize!(qh, nev=0)
    @test qh.ε[1] ≈ 2.018 rtol=1e-3

    # approximate diagonalisation
    diagonalize!(qh, nev=1)
    @test qh.ε[1] ≈ 2.018 rtol=1e-3

    ########## χ = π/2, x from -π/2 to π/2

    χ = π/2

    xlimits = (-π/2, π/2) .|> Float32
    ylimits = (0, π) .|> Float32

    ### sin basis
    qh = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:sin, M)
    
    # exact diagonalisation
    diagonalize!(qh, nev=0)
    @test qh.ε[1] ≈ 3.179 rtol=1e-3

    # approximate diagonalisation
    diagonalize!(qh, nev=1)
    @test qh.ε[1] ≈ 3.179 rtol=1e-3

    ########## χ = π/2, full period

    χ = π/2

    xlimits = (0, 2π) .|> Float32
    ylimits = (0, 2π) .|> Float32

    qh = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M, 𝑈_iseven=true);

    # exact diagonalisation
    diagonalize!(qh, nev=0)
    @test qh.ε[1] ≈ 0.571 atol=1e-3

    # approximate diagonalisation
    diagonalize!(qh, nev=1)
    @test qh.ε[1] ≈ 0.571 atol=1e-3

    qh = PSpaceHamiltonian{:sparse}([Float64.(xlimits), Float64.(ylimits)], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M, 𝑈_iseven=true) # cast to Float64 manually to suppress info message
    diagonalize!(qh, nev=1)
    @test qh.ε[1] ≈ 0.571 atol=1e-3
end

@testset "Test dense 2D 3-component diagonalisation" begin
    # Tests based on the system in https://doi.org/10.1103/PhysRevA.107.033328 (https://arxiv.org/abs/2304.00302)

    function 𝛺₁(x::Real, y::Real)
        Ω₁₀ / 2
    end

    function 𝛺₂(x::Real, y::Real)
        ( -Ω₋ * cos(x-y) + Ω₊ * cos(x+y) ) / 2
        # - Ω₋ * cis(χ/2) * cos(x-y) + Ω₊ * cis(-χ/2) * cos(x+y) 
    end

    ϵ::Float32 = 0.1
    ϵc::Float32 = 1
    Ω₁₀::Float32 = 2000
    Ω₊ = Ω₁₀ / (ϵ*√(1+ϵc^2))
    Ω₋ = Ω₊ * ϵc
    # χ::Float32 = 0
    Γ₃::Float32 = 1e3

    xlimits = (-π, π) .|> Float32
    ylimits = (-π, π) .|> Float32

    M = 1 # one harmonic is enough to capture the transform in the cis basis case :)
    𝑈 = [nothing nothing 𝛺₁      
         nothing nothing 𝛺₂
         nothing nothing nothing] # only upper triangle is needed
    𝑈_iseven=trues(3, 3)
    
    ### Hermitian cis basis diagonalisation
    qh = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven)
    @test qh.H isa Matrix{Float32}
    @test qh.H[1, 19] ≈ Ω₁₀ / 2
    @test qh.H[10, 23] ≈ Ω₊ / 4
    @test qh.H[11, 22] ≈ -Ω₊ / 4
        
    # exact diagonalisation
    diagonalize!(qh, nev=0)
    @test qh.ε[1] ≈ -7140 rtol=1e-3

    # approximate diagonalisation
    diagonalize!(qh, nev=1)
    @test qh.ε[1] ≈ 0.039 atol=1e-3

    ### Hermitian sin basis diagonalisation
    qh = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:sin, M)
    @test qh.H isa Matrix{Float32}
        
    # exact diagonalisation
    diagonalize!(qh, nev=0)
    @test qh.ε[1] ≈ -1000 rtol=1e-3

    # approximate diagonalisation
    diagonalize!(qh, nev=1)
    @test qh.ε[1] ≈ 0.5 rtol=1e-3
    
    ### non-Hermitian cis basis diagonalisation
    qh = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven, Γ=[0, 0, Γ₃]);
    @test qh.H isa Matrix{Complex{Float32}}
    
    # exact diagonalisation
    diagonalize!(qh, nev=0)
    @test qh.ε[1] ≈ -7136 - 250im rtol=1e-3
    l = findfirst(x -> real(x) > 0, qh.ε)
    @test l == 10
    @test qh.ε[l] ≈ 0.039 - 0.0002im atol=1e-3

    # approximate diagonalisation
    diagonalize!(qh, nev=1)
    @test qh.ε[1] ≈ 0.039 atol=1e-3

    ### non-Hermitian sin basis diagonalisation
    qh = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:sin, M, Γ=[0, 0, Γ₃]);
    @test qh.H isa Matrix{Complex{Float32}}
    
    # exact diagonalisation
    diagonalize!(qh, nev=0)
    @test qh.ε[1] ≈ -968 - 250im rtol=1e-3

    # approximate diagonalisation
    diagonalize!(qh, nev=1)
    @test qh.ε[1] ≈ 0.5 rtol=1e-3
end

@testset "Test sparse 2D 3-component diagonalisation" begin
    # Tests based on the system in https://doi.org/10.1103/PhysRevA.107.033328 (https://arxiv.org/abs/2304.00302)

    function 𝛺₁(x::Real, y::Real)
        Ω₁₀ / 2
    end

    function 𝛺₂(x::Real, y::Real)
        ( -Ω₋ * cos(x-y) + Ω₊ * cos(x+y) ) / 2
    end

    ϵ::Float64 = 0.1
    ϵc::Float64 = 1
    Ω₁₀::Float64 = 2000
    Ω₊ = Ω₁₀ / (ϵ*√(1+ϵc^2))
    Ω₋ = Ω₊ * ϵc
    Γ₃::Float64 = 1e3

    xlimits = (-π, π) .|> Float64
    ylimits = (-π, π) .|> Float64

    M = 1 # one harmonic is enough to capture the transform in the cis case :)
    𝑈 = [nothing nothing 𝛺₁      
         nothing nothing 𝛺₂
         nothing nothing nothing] # only upper triangle is needed
    𝑈_iseven = trues(3, 3)

    ### Hermitian cis basis diagonalisation
    qh = PSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven)
    @test qh.H[1, 19] ≈ Ω₁₀ / 2
    @test qh.H[10, 23] ≈ Ω₊ / 4
    @test qh.H[11, 22] ≈ -Ω₊ / 4
        
    diagonalize!(qh, nev=1)
    @test qh.ε[1] ≈ 0.039 atol=1e-3

    ### Non-Hermitian cis basis diagonalisation
    qh = PSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven, Γ=[0, 0, Γ₃])
    @test qh.H[1, 19] ≈ Ω₁₀ / 2
    @test qh.H[10, 23] ≈ Ω₊ / 4
    @test qh.H[11, 22] ≈ -Ω₊ / 4
        
    diagonalize!(qh, nev=1)
    @test qh.ε[1] ≈ 0.039 atol=1e-3
end