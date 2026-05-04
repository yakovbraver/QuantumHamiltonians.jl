@testset "1-component diagonalisation" begin
    # Quantum harmonic oscillator: 𝐻 = -Δ/2 + 𝑥²/2 

    𝑈(x) = x^2 / 2
    xlimits = (-10, 10) .|> Float64
    𝜓₄(x) = 1/√(2^4*factorial(4)√π) * exp(-x^2/2) * (16x^4 - 48x^2 + 12) # analytical state with n = 4

    for basis in (:cis, :sin, :cos), type in (:dense, :sparse, :xspace)
        M = basis == :sin ? 63 : 64
        if type == :xspace
            qh = XSpaceHamiltonian([xlimits], 𝑈; basis, M, δ=√0.5) # `√` because `δ` is the coefficient of ∂ₓ, not Δ
        else
            (type == :sparse && basis != :cis) && continue # sparse is only implemented for cis
            kwargs = type == :sparse ? (;fft_threshold=1e-2) : (;)
            qh = PSpaceHamiltonian{type}([xlimits], 𝑈; basis, 𝑈_iseven=true, M, δ=√0.5, kwargs...) # `√` because `δ` is the coefficient of ∂ₓ, not Δ
            # test correctness of type parameters
            @test qh isa PSpaceHamiltonian{type, Float64, Float64, Float64, 2, 3}
        end

        # will test 5th eigenstate (=4th excited state)
        stateno = 5

        if type == :xspace
            vals, vecs, info = diagonalize(qh; nev=5); # for large M you may need to tweak solver parameters. E.g. for M = 256, to converge to the default Float64 tolerance of 1e-12, set `krylovdim=35` (default is 30)
            xs = qh.ft.xs
            ψ = vecs[stateno][1]
            ε = vals[1:5]
        else
            diagonalize!(qh, nev=5)
            xs, ψ = make_eigenfunctions(qh; statenos=[stateno], nx=50)
            ε = qh.ε
        end
        
        @test ε ≈ 0.5:1:4.5 rtol=1e-5

        if basis == :cis
            @test isapprox.(imag.(ψ), 0, atol=1e-8) |> all # 5th eigenstate is even, hence ψ should be purely real
        end
        
        ψ_true = 𝜓₄.(xs)
        @test all(@. abs(ψ) - abs(ψ_true) < 1e-4) # test abs because a sign difference is possible
    end
end

@testset "1-component nonlinear stationary state and BdG" begin
    # Bright soliton in a "parabolic + bump at the centre" potential -- "Soliton on a hill" from https://doi.org/10.1093/oso/9780192843234.001.0001, Section 22.5

    𝑈(x::Real) = (Ω*x)^2 / 2 + B*sech(β*x)^2

    Ω = 0.075 |> Float64
    B = 0.3 |> Float64
    β = 0.5 |> Float64
    δ = √0.5 |> Float64

    for basis in (:cis, :sin, :cos)
        p = 8
        M = basis == :cis ? (p == 7 ? 62 : p == 8 ? 122 : 247) :
            basis == :sin ? 2^p-1 : 2^p
        R = 15
        xlimits = (-R, R) .|> Float64
        qh = PSpaceHamiltonian{:dense}([xlimits], 𝑈; basis, 𝑈_iseven=true, M, δ)

        # Calculate stationary state and test energy
        𝜓₀(x) = sech(x)
        g = -1 |> Float64
        μ₀ = -1 |> Float64
        if basis == :sin
            _, sol = find_stationary(qh, [𝜓₀], [g;;], μ₀; abstol=5e-13)
        else
            _, sol = find_stationary(qh, [𝜓₀], [g;;], μ₀; abstol=5e-13)
        end
        @test Int(sol.retcode) == 1 # test for success
        E = get_Eμη(qh, sol.u, [g;;], v_is_pspace=false)[1]
        @test E ≈ -0.1581185113871 atol=1e-12 # default NonlinearSolve tolerance for Float64 is ≈ 3e-13

        # Calculate BdG and test relevant eigenvalues (calculating all eigenvalues here)
        vals, vecs = bdg_spectrum(qh, sol.u, g, μ₀)
        @test maximum(imag, vals) ≈ 0.3306185 atol=1e-7 # default ArnoldiMethod tolerance for Float64 is √eps ≈ 1.5e-8

        # also test BdG in p-space. Skip sin case because that requires 2M+1 harmonics for dimensions to match, but the result is then inaccurate
        if basis != :sin
            xh_half = PSpaceHamiltonian{:dense}([xlimits], 𝑈; basis, 𝑈_iseven=true, M=M÷2, δ)
            vals, vecs = XSpaceHamiltonians.bdg_spectrum_pspace(xh_half, sol.u, g, μ₀; ψ_iseven=true)
            @test maximum(imag, vals) ≈ 0.3306185 atol=1e-5 # using larger atol because with twice less harmonics this is less accurate
        end
    end
end

@testset "2-component nonlinear stationary state, BdG, and time evolution" begin
    # Stationary Manakov dark-bright soliton, see e.g. https://doi.org/10.1093/oso/9780192843234.001.0001, Section 27.1

    Float = Float64 # operating type

    basis = :cos
    M = 256

    R = 10 |> Float
    xlimits = (-R, R)

    qh = PSpaceHamiltonian{:dense}([xlimits], fill(nothing, 2, 2); basis, 𝑈_iseven=trues(2, 2), M, δ=√0.5)

    μ₀ = 4 |> Float
    η = 1 |> Float
    D = √(μ₀ - η^2)
    g = ones(Float, 2, 2)
    μs = [μ₀, (μ₀+η^2)/2] # actual chemical potentials of the two components

    ### Testing for fixed 𝜇's

    # Get stationary state starting from a basic tanh-sech trial
    xs, sol = find_stationary(qh, [tanh, sech], g, μs; abstol=1e-9)
    @test Int(sol.retcode) == 1 # test for success
    ψ_db = sol.u

    # Test against analytical solution (testing against abs because might converge to a different sign)
    𝛹 = [x -> abs( √μ₀ * tanh(D*x) ), x -> abs( η * sech(D*x) )]
    Ψ_exact = [𝛹[1].(xs); 𝛹[2].(xs)] |> vec
    @test sum(abs, Ψ_exact - abs.(ψ_db)) / length(ψ_db) < 1e-8

    ### Testing for fixed 𝑁ᵢ's

    natoms = get_Eμη(qh, ψ_db, [g;;], v_is_pspace=false)[3] # get numbers of atoms from the above state
    # Get stationary state starting from a basic tanh-sech trial, this time for fixed number of atoms and using a guess for 𝜇 given by [2, 1]
    xs, sol = find_stationary(qh, [tanh, sech], [g;;], [2.0, 1.0], natoms; abstol=1e-9)
    @test Int(sol.retcode) == 1 # test for success
    # check chemical potentials contained in the last two elements of sol.u
    @test sol.u[end-1] ≈ μs[1] atol=1e-9
    @test sol.u[end] ≈ μs[2] atol=1e-9
    ψ_db = sol.u[1:end-2]

    # Again test against analytical solution (testing against abs because might converge to a different sign)
    @test sum(abs, Ψ_exact - abs.(ψ_db)) / length(ψ_db) < 1e-8

    # Calculate BdG spectrum in x-space
    vals, vecs = bdg_spectrum(qh, ψ_db, g, μs)
    smallindx = findall(x -> abs(x) < 1e-2, vals) # find indices of very small values
    @test length(smallindx) == 6 # there should be exactly 6 values "close to zero"
    @test all(x -> abs(x) < 5e-5, vals[smallindx]) # those values should be ≲ 5e-5

    # Calculate BdG spectrum in p-space
    xh_half = PSpaceHamiltonian{:dense}([xlimits], fill(nothing, 2, 2); basis, 𝑈_iseven=trues(2, 2), M=M÷2, δ=√0.5);
    vals, vecs = XSpaceHamiltonians.bdg_spectrum_pspace(xh_half, [ψ_db[1:end÷2] ψ_db[end÷2+1:end]], g, μs)
    smallindx = findall(x -> abs(x) < 1e-2, vals) # find indices of very small values
    @test length(smallindx) == 6 # there should be exactly 6 values "close to zero"
    @test all(x -> abs(x) < 5e-5, vals[smallindx]) # those values should be ≲ 5e-5

    # time evolution
    nsaves = 2
    T_max = 50 |> Float
    dt = 1e-3 |> Float
    sol = propagate(qh, [ψ_db[1:end÷2], ψ_db[end÷2+1:end]], g; T_max, dt, itime=false, nsaves, solver=XSpaceHamiltonians.ODE.ETDRK2())
    @test Int(sol.retcode) == 1 # test for success

    # test energy conservation
    E1 = get_Eμη(qh, sol.u[1], g)[1]
    E2 = get_Eμη(qh, sol.u[end], g)[1]
    @test abs(E2 - E1) < 1e-5

    # test that density remains the same
    xs, Ψ1 = make_wavefunction(qh, sol.u[1])
    xs, Ψ2 = make_wavefunction(qh, sol.u[end])
    @test sum(abs2, abs2.(Ψ1) - abs2.(Ψ2)) < 5e-8
end

@testset "2-component nonlinear Floquet-BdG" begin
    # DB lattices from http://dx.doi.org/10.1103/PhysRevA.91.023619 (https://arxiv.org/abs/1402.1895)

    Float = Float64 # operating type

    basis = :cis
    M = 62

    # trial
    nT = 1 # number of periods
    Ψ₀ = [x -> sin(2π/(2R/nT)*x), x -> cos(2π/(2R/nT)*x)]

    R = 20*nT |> Float
    xlimits = (-R, R)

    qh = PSpaceHamiltonian{:dense}([xlimits], fill(nothing, 2, 2); basis, 𝑈_iseven=trues(2, 2), M, δ=√0.5);
    xh_half = PSpaceHamiltonian{:dense}([xlimits], fill(nothing, 2, 2); basis, 𝑈_iseven=trues(2, 2), M=M÷2, δ=√0.5);

    μs = [1.5, 1.23]
    g = [1   0.8
         0.8 0.95] .|> Float

    xs, sol = find_stationary(qh, Ψ₀, g, μs, abstol=1e-11) 
    ψ_lattice = sol.u

    qs = [π/4R] # quasimomentum chosen for the test
    vals_true = [0.009, 0.016, 0.031, 0.029].*im # true values of the imaginary parts of the unstable modes
    
    # BdG in x-space
    vals, _ = bdg_spectrum(qh, ψ_lattice, g, μs, [qs], nev=100)
    indx = findall(x -> imag(x) > 0.005, vals)
    @test length(indx) == 4 # there should be 4 unstable eigenvalues
    @test isapprox.(sort(vals[indx]; by=imag), vals_true; atol=1e-2) |> all

    # BdG in p-space
    vals, _ = XSpaceHamiltonians.bdg_spectrum_pspace(xh_half, [ψ_lattice[1:end÷2] ψ_lattice[end÷2+1:end]], g, μs, qs; nev=100)
    indx = findall(x -> imag(x) > 0.005, vals)
    @test length(indx) == 4 # there should be 4 unstable eigenvalues
    @test isapprox.(sort(vals[indx]; by=imag), vals_true; atol=1e-2) |> all
end