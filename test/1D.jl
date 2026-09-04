@testset "1-component diagonalisation" begin
    # Quantum harmonic oscillator: 𝐻 = -Δ/2 + 𝑥²/2 

    𝑈(x) = x^2 / 2
    xlimits = (-10, 10) .|> Float64
    stateno = 5 # will test 5th eigenstate (=4th excited state) 
    𝜓₄(x) = 1/√(2^4*factorial(4)√π) * exp(-x^2/2) * (16x^4 - 48x^2 + 12) # analytical state with 𝑛 = 4

    for basis in (:cis, :sin, :cos), kind in (:dense, :sparse, :xspace, :xspace_statevector)
        M = basis == :sin ? 63 : 64

        if kind == :xspace
            qh = XSpaceHamiltonian([xlimits], 𝑈; basis, M, δ=√0.5) # `√` because `δ` is the coefficient of ∂ₓ, not Δ
            diagonalize!(qh; nev=5)
            xs, v = make_eigenfunction(qh, stateno)
            ψ = v[1]
            ε = qh.ε
        elseif kind == :xspace_statevector
            qh = XSpaceHamiltonian([xlimits], 𝑈; basis, M, δ=√0.5)
            vals, vecs, info = QuantumHamiltonians.diagonalize_via_statevector(qh; nev=5)
            xs = qh.ft.xs
            ψ = vecs[stateno][1]
            ε = vals[1:5]
        else
            (kind == :sparse && basis != :cis) && continue # sparse is only implemented for cis
            kwargs = kind == :sparse ? (;fft_threshold=1e-2) : (;) # pass `fft_threshold` for sparse; otherwise pass an empty named tuple
            qh = PSpaceHamiltonian{kind}([xlimits], 𝑈; basis, 𝑈_iseven=true, M, δ=√0.5, kwargs...)
            # test correctness of kind parameters
            @test qh isa PSpaceHamiltonian{kind, Float64, Float64, Float64, 2, 3}
            diagonalize!(qh, nev=5)
            xs, ψ = make_eigenfunctions(qh; statenos=[stateno], nx=50)
            ε = qh.ε
        end
        
        @test ε ≈ 0.5:1:4.5 rtol=1e-5

        if basis == :cis
            @test all(@. imag(ψ) < 1e-8) # 5th eigenstate is even, hence ψ should be purely real
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
        _, ψ_nr, _ = find_stationary(qh, [𝜓₀], [g;;], μ₀; searchreal=true, abstol=5e-13)
        E = get_EμN(qh, ψ_nr, [g;;], state_is_pspace=false)[1]
        @test E ≈ -0.1581185113871 atol=1e-12 # default NonlinearSolve tolerance for Float64 is ≈ 3e-13

        # Calculate BdG and test relevant eigenvalues (calculating all eigenvalues here)
        vals, vecs = bdg_spectrum(qh, ψ_nr, g, μ₀)
        @test maximum(imag, vals) ≈ 0.3306185 atol=1e-7 # default ArnoldiMethod tolerance for Float64 is √eps ≈ 1.5e-8

        # also test BdG in p-space. Skip sin case because that requires 2M+1 harmonics for dimensions to match, but the result is then inaccurate
        if basis != :sin
            xh_half = PSpaceHamiltonian{:dense}([xlimits], 𝑈; basis, 𝑈_iseven=true, M=M÷2, δ)
            vals, vecs = QuantumHamiltonians.bdg_spectrum_pspace(xh_half, ψ_nr, g, μ₀; ψ_iseven=true)
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
    xs, ψ_db, _ = find_stationary(qh, [tanh, sech], g, μs; searchreal=true, abstol=1e-9)

    # Test against analytical solution (testing against abs because might converge to a different sign)
    𝛹 = [x -> abs( √μ₀ * tanh(D*x) ), x -> abs( η * sech(D*x) )]
    Ψ_exact = [𝛹[1].(xs); 𝛹[2].(xs)] |> vec
    @test sum(abs, Ψ_exact - abs.(ψ_db)) / length(ψ_db) < 1e-8

    ### Testing for fixed 𝑁ᵢ's

    natoms = get_EμN(qh, ψ_db, [g;;], state_is_pspace=false)[3] # get numbers of atoms from the above state
    # Get stationary state starting from a basic tanh-sech trial, this time for fixed number of atoms and using a guess for 𝜇 given by [2, 1]
    xs, ψ_db, μs_nr = find_stationary(qh, [tanh, sech], [g;;], [2.0, 1.0], natoms; searchreal=true, abstol=1e-9)
    @test μs_nr[1] ≈ μs[1] atol=1e-9
    @test μs_nr[2] ≈ μs[2] atol=1e-9

    # Again test against analytical solution (testing against abs because might converge to a different sign)
    @test sum(abs, Ψ_exact - abs.(ψ_db)) / length(ψ_db) < 1e-8

    # Calculate BdG spectrum in x-space
    vals, vecs = bdg_spectrum(qh, ψ_db, g, μs)
    smallindx = findall(x -> abs(x) < 1e-2, vals) # find indices of very small values
    @test length(smallindx) == 6 # there should be exactly 6 values "close to zero"
    @test all(x -> abs(x) < 5e-5, vals[smallindx]) # those values should be ≲ 5e-5

    # Calculate BdG spectrum in p-space
    xh_half = PSpaceHamiltonian{:dense}([xlimits], fill(nothing, 2, 2); basis, 𝑈_iseven=trues(2, 2), M=M÷2, δ=√0.5);
    vals, vecs = QuantumHamiltonians.bdg_spectrum_pspace(xh_half, [ψ_db[1:end÷2] ψ_db[end÷2+1:end]], g, μs)
    smallindx = findall(x -> abs(x) < 1e-2, vals) # find indices of very small values
    @test length(smallindx) == 6 # there should be exactly 6 values "close to zero"
    @test all(x -> abs(x) < 5e-5, vals[smallindx]) # those values should be ≲ 5e-5

    # time evolution
    nsaves = 2
    T_max = 50 |> Float
    dt = 1e-3 |> Float
    sol = propagate(qh, [ψ_db[1:end÷2], ψ_db[end÷2+1:end]], g; T_max, dt, itime=false, nsaves, solver=QuantumHamiltonians.ODE_EXP.ETDRK2())
    @test Int(sol.retcode) == 1 # test for success

    # test energy conservation
    E1 = get_EμN(qh, sol.u[1], g)[1]
    E2 = get_EμN(qh, sol.u[end], g)[1]
    @test abs(E2 - E1) < 1e-5

    # test that density remains the same
    xs, Ψ1 = make_wavefunction(qh, sol.u[1])
    xs, Ψ2 = make_wavefunction(qh, sol.u[end])
    @test sum(abs2, abs2.(Ψ1[1]) - abs2.(Ψ2[1])) < 5e-8
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

    xs, ψ_lattice, _ = find_stationary(qh, Ψ₀, g, μs; searchreal=true, abstol=1e-11)

    qs = [π/4R] # quasimomentum chosen for the test
    vals_true = [0.009, 0.016, 0.031, 0.029].*im # true values of the imaginary parts of the unstable modes
    
    # BdG in x-space
    vals, _ = bdg_spectrum(qh, ψ_lattice, g, μs, [qs], nev=100)
    indx = findall(x -> imag(x) > 0.005, vals)
    @test length(indx) == 4 # there should be 4 unstable eigenvalues
    @test isapprox.(sort(vals[indx]; by=imag), vals_true; atol=1e-2) |> all

    # BdG in p-space
    vals, _ = QuantumHamiltonians.bdg_spectrum_pspace(xh_half, [ψ_lattice[1:end÷2] ψ_lattice[end÷2+1:end]], g, μs, qs; nev=100)
    indx = findall(x -> imag(x) > 0.005, vals)
    @test length(indx) == 4 # there should be 4 unstable eigenvalues
    @test isapprox.(sort(vals[indx]; by=imag), vals_true; atol=1e-2) |> all
end

# ~ 1 min 15 s
@testset "1-component imaginary and real dynamics" begin
    # Soliton in a harmonic potential; Na-23 parameters similar to https://doi.org/10.1103/PhysRevLett.87.130402 (https://arxiv.org/abs/cond-mat/0104549)

    m = 3.8165e-26
    aₛ = 2.5e-9
    h = 6.62607015e-34
    ħ = h / 2π
    ω = 3.5 * 2π # 1D trap frequency
    a₀ = √(ħ / (m*ω)) # [1/m] -- unit of length
    α = 100 # ω⟂ / ω ratio
    τ = 1/ω # [s] unit of time

    n_atoms = 1e4
    g = 2 * α * (aₛ/a₀) * n_atoms # coefficient of nonlinearity
    R = 11.0 # trap half-length, in units of a₀

    δ = √0.5 # coefficient of the momentum term

    𝑈(x::Real) = x^2 / 2

    xlimits = (-R, R)

    # construct analytic soliton trial
    natoms = 1.0
    p = 1/√(2R) # value of wf in the bulk (= ground state solution for the free case)
    ξ = √(1/(p^2 * g)) # healing length
    𝜓₀(x) = p * tanh(x/ξ) # soliton trial
    μ₀ = 40.0

    for basis in (:cis, :sin, :cos), kind in (:dense, :sparse, :xspace)
        M = basis == :cis ? (kind == :xspace ? 64 : 62) :
            basis == :sin ? 127 : 128

        if kind == :xspace
            qh = XSpaceHamiltonian([xlimits], 𝑈; basis, M, δ)
        else
            (kind == :sparse && basis != :cis) && continue # sparse is only implemented for cis
            kwargs = kind == :sparse ? (;fft_threshold=1e-3) : (;) # pass `fft_threshold` for sparse; otherwise pass an empty named tuple
            qh = PSpaceHamiltonian{kind}([xlimits], 𝑈; basis, 𝑈_iseven=true, M, δ, kwargs...)
        end
        
        # find soliton state using Newton-Raphson
        xs, ψ_nr, _ = find_stationary(qh, [𝜓₀], [g;;], μ₀, natoms; searchreal=true)
        if kind != :xspace
            E_nr, μ_nr, N_nr = get_EμN(qh, ψ_nr, [g;;]; state_is_pspace=false)
        else
            E_nr, μ_nr, N_nr = get_EμN(qh, ψ_nr, [g;;])
        end

        ### imaginary time test: calculate a soliton, compare with Newton-Raphson

        # find soliton state using imaginary time
        T_max = 1.0
        dt = 1e-4
        sol = propagate(qh, 𝜓₀, g; T_max, dt, itime=true, solver=QuantumHamiltonians.ODE_EXP.LawsonEuler(;krylov=true, m=5))
        E_itime, μ_itime, N_itime = get_EμN(qh, sol.u[end], [g;;])

        # compare itime agains NR
        @test N_itime[1] ≈ natoms atol=1e-10
        @test E_itime ≈ E_nr rtol=1e-5
        @test μ_itime[1] ≈ μ_nr[1] rtol=5e-4

        ### real time test: calculate half-period of oscillations, check energy conservation and check that wf minimum is at roughly -5

        # get ground state
        xs, ψ, _ = find_stationary(qh, [one], [g;;], μ₀, natoms; searchreal=true)
        # create a displaced soliton
        ψ₀ = real(ψ) .* tanh.(9 .* (xs .- 5)) |> vec

        T_max = π√2 # soliton oscillation half-period, in units of 1/ω
        dt = 1e-3
        nsaves = 2
        G = kind == :xspace ? [g;;] : g # work-around for interface difference 
        sol = propagate(qh, ψ₀, G; T_max, dt, itime=false, nsaves, solver=QuantumHamiltonians.ODE_EXP.ETDRK4(;krylov=true, m=5))

        E_final = get_EμN(qh, sol.u[end], [g;;])[1]
        E_initial = get_EμN(qh, sol.u[1], [g;;])[1]
        @test E_final ≈ E_initial rtol=1e-4

        xs, ψ = make_wavefunction(qh, sol.u[end])
        window = -7 .< xs .< 7
        xs_reduced = xs[window]
        ψ²_min, ix_min = findmin(abs2, ψ[1][window]) # find minimum of the density on 𝑥 ∈ (-7, 7)
        @test xs_reduced[ix_min] ≈ -5.3 atol=0.5
        @test ψ²_min < 0.0021
    end
end
