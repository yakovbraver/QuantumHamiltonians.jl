@testset "Various `fft_to_operator`" begin
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

@testset "1-component diagonalisation for a scalar potential (all bases, all Hamiltonian kinds)." begin
    # Assymetric quantum harmonic oscillator: 𝐻 = -Δ/2 + 𝑥²/2 + 3𝑦²/2

    ω₁ = 1; ω₂ = 3;
    𝑈(x, y) = ω₁^2 * x^2 / 2 + ω₂^2 * y^2 / 2
    xlimits = (-5, 5) .|> Float64
    stateno = 2 # will test 2nd eigenstate (=4th excited state) 
    𝜓₁₀(x, y) = (ω₁/π)^(1/4) / √2 * exp(-ω₁*x^2/2) * 2x*√ω₁ * (ω₂/π)^(1/4) * exp(-ω₂*y^2/2)

    for basis in (:cis, :sin, :cos), kind in (:dense, :sparse, :xspace, :xspace_statevector)
        M = basis == :sin ? 31 : 32

        if kind == :xspace
            qh = XSpaceHamiltonian([xlimits, xlimits], 𝑈; basis, M, δ=√0.5) # `√` because `δ` is the coefficient of ∂ₓ, not Δ
            diagonalize!(qh; nev=4)
            xs, ys, ψ = make_eigenfunction(qh, stateno)
            ε = qh.ε
        elseif kind == :xspace_statevector
            qh = XSpaceHamiltonian([xlimits, xlimits], 𝑈; basis, M, δ=√0.5)
            vals, vecs, info = QuantumHamiltonians.diagonalize_via_statevector(qh; nev=5)
            xs = qh.ft.xs[:, 1]
            ys = qh.ft.xs[:, 2]
            ψ = vecs[stateno]
            ε = vals[1:4]
        else
            (kind == :sparse && basis != :cis) && continue # sparse is only implemented for cis
            kwargs = kind == :sparse ? (;fft_threshold=1e-2) : (;) # pass `fft_threshold` for sparse; otherwise pass an empty named tuple
            qh = PSpaceHamiltonian{kind}([xlimits, xlimits], 𝑈; basis, 𝑈_iseven=true, M, δ=√0.5, kwargs...)
            # test correctness of kind parameters
            @test qh isa PSpaceHamiltonian{kind, Float64, Float64, Float64, 3, 4}
            diagonalize!(qh, nev=4)
            xs, ys, ψ = make_eigenfunction(qh, stateno)
            ε = qh.ε
        end
        
        @test ε ≈ 2:5 rtol=1e-5 # exact spectrum is εˣʸ = (𝑛ˣ + 1/2) + 3(𝑛ʸ + 1/2); lowest energies are 2, 3, 4, 5, 5, 6

        ψ_true = 𝜓₁₀.(xs, ys')
        @test all(@. abs(ψ[1]) - abs(ψ_true) < 1e-2) # test abs because a sign difference is possible
    end
end

# TODO add additional type checks
@testset "1-component diagonalisation for a vector potential (all bases, all Hamiltonian kinds, preconditioning)" begin
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

    ########## χ = 0 (no 𝐴)

    χ::Float32 = 0

    xlimits = (0, π) .|> Float32
    ylimits = (0, π) .|> Float32

    M = 16

    ### sin basis
    qh = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:sin, M)
    @test qh.H isa Matrix{Float32}
    @test qh.V isa Matrix{Float32}

    # exact diagonalisation (nev=0) and approximate (nev=1)
    for nev in (0, 1)
        diagonalize!(qh; nev)
        @test qh.ε[1] ≈ 2.117 rtol=1e-3
    end
    
    ### cis basis
    qh = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=true)
    @test qh.H isa Matrix{Float32}

    # exact diagonalisation (nev=0) and approximate (nev=1)
    for nev in (0, 1)
        diagonalize!(qh; nev)
        @test qh.ε[1] ≈ 1.960 rtol=1e-3
    end
    
    # XSpaceHamiltonian. It's faster, so we can test with Float64 and larger `M` (but seems less accurate than p-space for small M?)
    for symmetrize_pA in (true, false) # there is no 𝐴 so `symmetrize_pA` should have no effect and yield identical results
        qh = XSpaceHamiltonian([Float64.(xlimits), Float64.(ylimits)], 𝑈; basis=:cis, M=32, symmetrize_pA)
        diagonalize!(qh, nev=1)
        @test qh.ε[1] ≈ 1.9599702145 rtol=1e-10
    end

    ########## χ = π/2, x from -π/2 to π/2

    M = 5
    χ = π/2

    xlimits = (-π/2, π/2) .|> Float32
    ylimits = (0, π) .|> Float32

    ### sin basis
    qh = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:sin, M)
    
    # exact diagonalisation (nev=0) and approximate (nev=1)
    for nev in (0, 1)
        diagonalize!(qh; nev)
        @test qh.ε[1] ≈ 3.179 rtol=1e-3
    end

    ########## χ = π/2, full period

    χ = π/2

    xlimits = (0, 2π) .|> Float32
    ylimits = (0, 2π) .|> Float32

    qh = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M, 𝑈_iseven=true);

    # exact diagonalisation (nev=0) and approximate (nev=1)
    for nev in (0, 1)
        diagonalize!(qh; nev)
        @test qh.ε[1] ≈ 0.571 atol=1e-3
    end

    qh = PSpaceHamiltonian{:sparse}([Float64.(xlimits), Float64.(ylimits)], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M, 𝑈_iseven=true) # cast to Float64 manually to suppress info message
    diagonalize!(qh, nev=1)
    @test qh.ε[1] ≈ 0.571 atol=1e-3

    ### XSpaceHamiltonian. It's faster, so we can test with Float64 and larger `M`
    M = 32

    # two random vectors for checking Hermiticity as (⟨𝑣₁, 𝐻𝑣₂⟩ - ⟨𝐻𝑣₁, 𝑣₂⟩)
    Random.seed!(47)
    v1 = rand(ComplexF64, (2M)^2) |> normalize # don't bother with actual x-space normalisation
    v2 = rand(ComplexF64, (2M)^2) |> normalize

    for symmetrize_pA in (true, false)
        ε_true = symmetrize_pA ? 0.6523291893 : 0.6524389189 # the result depends on `symmetrize_pA`; the `false` case should be considered inaccurate
        qh = XSpaceHamiltonian([Float64.(xlimits), Float64.(ylimits)], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M, symmetrize_pA)

        # test hermiticity
        defect = dot(v1, qh*v2) - dot(qh*v1, v2) # if `symmetrize_pA`, the nonhermiticity defect should be on the order of machine precision
        @test abs(defect) < (symmetrize_pA ? 1e-13 : 0.3) 

        # test diagonalisation
        for kwargs in ((;),
                       (;invert=true, ls_kwargs=(;atol=1e-12, rtol=1e-12)),
                       (;invert=true, ls_kwargs=(;atol=1e-12, rtol=1e-12), ls_prec=:jacobi),
                       (;invert=true, ls_kwargs=(;atol=1e-12, rtol=1e-12), ls_prec=:jacobi, ls_prec_side=:right))
            diagonalize!(qh; nev=1, kwargs...)
            @test qh.ε[1] ≈ ε_true atol=1e-10
        end
        
        # in the symmetrised case, also check diagonalisation with CG
        if symmetrize_pA
            diagonalize!(qh; nev=1, invert=true, ls_kwargs=(;atol=1e-12, rtol=1e-12), ls_prec=:jacobi, ls_solver=QuantumHamiltonians.LS.KrylovJL_CG)
            @test qh.ε[1] ≈ ε_true atol=1e-10
        end
    end
end

@testset "3-component diagonalisation (dense, cis/sin)" begin
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

@testset "3-component diagonalisation (sparse)" begin
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

    for γ in (0.0, Γ₃)
        qh = PSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven, Γ=[0, 0, γ])
        @test qh.H[1, 19] ≈ Ω₁₀ / 2
        @test qh.H[10, 23] ≈ Ω₊ / 4
        @test qh.H[11, 22] ≈ -Ω₊ / 4
        
        diagonalize!(qh, nev=1)
        @test qh.ε[1] ≈ 0.039 atol=1e-3
    end
end

@testset "1-component GPE real stationary state via Newton-Raphson (all bases, all Hamiltonian kinds)" begin
    # Ground state in a parabolic potential

    𝑈(x::Real, y::Real) = x^2 + y^2

    R = 5.0
    xlimits = (-R, R)

    # Newton-Raphson under fixed total number of particles
    natoms = 1.0
    𝜓₀(x, y) = exp(-x^2 - y^2) # initial guess
    g = 100.0
    μ₀ = 5.0

    # imaginary time parameters
    T_max = 1.0
    dt = 1e-3
    LawsonEuler = QuantumHamiltonians.ODE_EXP.LawsonEuler(;krylov=true, m=5)

    for basis in (:cis, :sin, :cos), kind in (:dense, :sparse, :xspace)
        M = basis == :cis ? 16 :
            basis == :sin ? 31 : 32

        if kind == :xspace
            qh = XSpaceHamiltonian([xlimits, xlimits], 𝑈; basis, M)
        else
            (kind == :sparse && basis != :cis) && continue # sparse is only implemented for cis
            kwargs = kind == :sparse ? (;fft_threshold=1e-3) : (;) # pass `fft_threshold` for sparse; otherwise pass an empty named tuple
            qh = PSpaceHamiltonian{kind}([xlimits, xlimits], 𝑈; basis, 𝑈_iseven=true, M, kwargs...)
        end
        
        ### stationary state using Newton-Raphson

        xs, ys, ψ_nr, μ_nr = find_stationary(qh, [𝜓₀], [g;;], μ₀, natoms; searchreal=true)
        if kind != :xspace
            E, μ, N = get_EμN(qh, ψ_nr, [g;;]; state_is_pspace=false)
        else
            E, μ, N = get_EμN(qh, ψ_nr, [g;;])
        end

        @test E ≈ 5.79206 atol=1e-5
        @test μ[1] ≈ 8.28601 atol=1e-5
        @test μ[1] ≈ μ_nr[1] atol=1e-12 # check for consistency with `get_EμN`
        @test N[1] ≈ natoms  atol=1e-12 # check for consistency with `get_EμN`

        if kind == :xspace
            xs, ys, ψ = make_wavefunction(qh, ψ_nr)
        else
            xs, ys, ψ = QuantumHamiltonians.make_wavefunction_xspace(qh, ψ_nr)
        end

        @test ψ[1][end÷2, end÷2] ≈ 0.28 atol=0.01 # checking wf value at the centre

        ### stationary state using imaginary time

        if kind == :xspace || basis == :cis # x-space works well for all bases, but p-space only converges reliably for cis
            solvers = kind == :xspace ? (LawsonEuler, :lie1, :_lie1) : (LawsonEuler,)
            for solver in solvers
                sol = propagate(qh, 𝜓₀, g; T_max, dt, itime=true, solver)
                E, μ, N = get_EμN(qh, sol.u[end], [g;;])

                @test E ≈ 5.79206 atol=0.02
                @test μ[1] ≈ 8.28601 atol=0.02
                @test N[1] ≈ natoms  atol=1e-12 # check for consistency with `get_EμN`

                xs, ys, ψ = make_wavefunction(qh, sol.u[end])
                @test ψ[1][end÷2, end÷2] ≈ 0.28 atol=0.01 # checking wf value at the centre
            end
        end
    end
end

@testset "1-component GPE dynamics (x-space, cis basis, w/ and w/o vector potential, split-step and exponential)" begin
    """
    For each state, detect the vortex and claculates its (r, ϕ) coordinates. `ψs` is a vector of flattened state vectors at each time point.
    `x_bound` is the maximum x of the windows in which to search for the vortex. The window is [-x_bound, x_bound] in both x and y.
    """
    function get_r_ϕ(xh, ψs; x_bound, pad=0)
        rs = Vector{Float64}(undef, length(ψs))
        ϕs = Vector{Float64}(undef, length(ψs))
        previous_wrapped_ϕ = 0.0
        for it in eachindex(ψs)
            xs, ys, ψ = make_wavefunction(xh, ψs[it]; pad)
            xwindow = findall(-x_bound .<= xs .<= x_bound)
            ywindow = findall(-x_bound .<= ys .<= x_bound)
            density = abs2.(ψ[1][xwindow, ywindow])
            _, ix_min = findmin(density)
            x_vortex = xs[xwindow[ix_min[1]]]
            y_vortex = ys[ywindow[ix_min[2]]]
            rs[it] = hypot(x_vortex, y_vortex)
            wrapped_ϕ = atan(y_vortex, x_vortex)
            if it == firstindex(ψs)
                ϕs[it] = wrapped_ϕ
            else
                Δϕ = wrapped_ϕ - previous_wrapped_ϕ
                Δϕ -= 2π * round(Δϕ / (2π))
                ϕs[it] = ϕs[it - 1] + Δϕ
            end
            previous_wrapped_ϕ = wrapped_ϕ
        end
        return rs, ϕs
    end

    function 𝑈(x::Real, y::Real)
        ω^2/2 * (x^2 + y^2)
    end

    R::Float64 = 8.5
    xlimits = (-R, R) 
    ω::Float64 = 0.5 # trap frequency, ω⟂ in equations
    g::Float64 = 500

    𝐴ˣ(x, y) = -B*y/2 / √2
    𝐴ʸ(x, y) = B*x/2 / √2
    𝛬ˣ(x, y) = -B*y*x/2 / √2
    𝛬ʸ(x, y) = B*x*y/2 / √2

    B::Float64 = 0.1

    ℓ = +1
    natoms = 1.0

    M = 32

    T_max = 50 |> Float64
    dt = 1e-2 |> Float64
    nsaves = 50

    for A_iszero in (true, false)
        𝐴 = A_iszero ? [nothing, nothing] : [𝐴ˣ, 𝐴ʸ]
        xh = XSpaceHamiltonian([xlimits, xlimits], 𝑈, 𝐴; basis=:cis, M, δ=1/√2)

        # Thomas-Fermi trial wf for ground state search
        μ_TF = ω * √(g/π)
        ψTF = reshape((μ_TF .- real.(xh.U[1])) ./ g, (2M, 2M))
        clamp!(ψTF, 0, Inf)
        @. ψTF = √ψTF

        xs, ys, u, μ_gnd = find_stationary(xh, [ψTF], [g;;], μ_TF, natoms; reltol=1e-5)
        xs, ys, ψ_gnd = make_wavefunction(xh, u)

        ξ = 1/√(2μ_gnd[1]) # healing length

        # create an off-centre vortex by multiplying the ground state by tanh(𝑟/𝜉√2)cis(𝜙)
        xᶜ, yᶜ = 2.4, 0
        ψ_ofc = map(CartesianIndices(ψ_gnd[1])) do I
            x, y = xs[I[1]], ys[I[2]]
            ψ_gnd[1][I] * tanh(hypot(x-xᶜ, y-yᶜ)/(ξ√2)) * cis(ℓ*atan(y-yᶜ, x-xᶜ))
        end 

        for solver in (:strang2, :yoshida4, QuantumHamiltonians.ODE_EXP.ETDRK4(;krylov=true, m=5))
            if A_iszero
                E_drift = solver == :strang2 ? 3e-7 :
                          solver == :yoshida4 ? 6e-9 : 5e-7
                ω_fit_true = 0.10105
                r_end_true = 2.38381326640467
                ϕ_end_true = 5.01652021288370
                rϕ_atol = 1e-14
            else
                solver == :yoshida4 && continue # Yoshida is not yet supported for nonzero 𝐀
                E_drift = solver == :strang2 ? 1e-4 : 4e-7
                ω_fit_true = 0.0515
                r_end_true = 2.40
                ϕ_end_true = 2.53
                rϕ_atol = 1e-2
            end

            sol = propagate(xh, ψ_ofc, g, [𝛬ˣ, 𝛬ʸ]; T_max, dt, nsaves, solver) # can always pass gauge phases -- will be ignored if not needed

            E₀ = get_EμN(xh, sol.u[1], [g;;])[1]
            ΔE = map(sol.u) do u
                get_EμN(xh, u, [g;;])[1] - E₀ |> abs
            end
            @test maximum(ΔE) / E₀ < E_drift

            rs, ϕs = get_r_ϕ(xh, sol.u; x_bound=1.5xᶜ, pad=4)
            @test rs[end] ≈ r_end_true atol=rϕ_atol
            @test ϕs[end] ≈ ϕ_end_true atol=rϕ_atol
    
            # do a linear fit for `ϕs`
            K = [sol.t ones(length(sol.t))] \ ϕs
            @test K[1] ≈ ω_fit_true atol=1e-4
        end
    end
end
