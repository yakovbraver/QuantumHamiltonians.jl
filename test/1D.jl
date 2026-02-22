@testset "Test dense 1D 1-component diagonalisation" begin
    ########## Quantum harmonic oscillator: 𝐻 = -Δ/2 + 𝑥²/2 

    𝑈(x) = x^2 / 2
    xlimits = (-5, 5) .|> Float32

    ### Periodic
    xh = XSpaceHamiltonian{:dense}([xlimits], 𝑈; basis=:cis, 𝑈_iseven=true, M=10, δ=(√0.5f0)) # `√` because `δ` is the coefficient of ∂ₓ, not Δ
    @test xh isa XSpaceHamiltonians.DenseHamiltonian{Float32,Float32,Float32,2,3}

    # test 5 lowest eigenvalues
    diagonalize!(xh, nev=5)
    @test xh.ε ≈ 0.5:1:4.5 rtol=1e-3

    # test 5th eigenstate (=4th excited state)
    stateno = 5
    xs, ψ = make_eigenfunctions(xh; statenos=[stateno], nx=50)
    @test isapprox.(imag.(ψ), 0, atol=1f-5) |> all # 5th eigenstate is even, hence ψ should be purely real
    
    𝜓₄(x) = 1/√(2^4*factorial(4)√π) * exp(-x^2/2) * (16x^4 - 48x^2 + 12) # analytical state with n = 4
    ψ_true = 𝜓₄.(xs)
    @test abs.(ψ) ≈ abs.(ψ_true) atol=1e-2 # test abs because a sign difference is possible

    ### Nonperiodic
    xh = XSpaceHamiltonian{:dense}([xlimits], 𝑈; basis=:sin, M=15, δ=(√0.5f0)) # `√` because `δ` is the coefficient of ∂ₓ, not Δ
    @test xh isa XSpaceHamiltonians.DenseHamiltonian{Float32,Float32,Float32,2,3}

    # test 5 lowest eigenvalues
    diagonalize!(xh, nev=5)
    @test xh.ε ≈ 0.5:1:4.5 rtol=1e-3

    # test 5th eigenstate (=4th excited state)
    stateno = 5
    xs, ψ = make_eigenfunctions(xh; statenos=[stateno], nx=50)
    @test abs.(ψ) ≈ abs.(ψ_true) atol=1e-2 # test abs because a sign difference is possible
end

@testset "Test dense 1D 1-component nlsolve stationary state and BdG" begin
    ########## Soliton on a hill (https://doi.org/10.1093/oso/9780192843234.001.0001, Section 22.5) ################

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
        xh = XSpaceHamiltonian{:dense}([xlimits], 𝑈; basis, 𝑈_iseven=true, M, δ)

        # Calculate stationary state and test energy
        𝜓₀(x) = sech(x)
        g = -1 |> Float64
        μ₀ = -1 |> Float64
        _, sol = find_stationary(xh, [𝜓₀], μ₀, [g;;])
        E, μ = get_E_μ(xh, sol.u, [g;;], v_is_pspace=false) .|> real
        @test E ≈ -0.1581185113871 atol=1e-12 # default NonlinearSolve tolerance for Float64 is ≈ 3e-13

        # Calculate BdG and test relevant eigenvalues (calculating all eigenvalues here)
        vals, vecs = bdg_spectrum(xh, sol.u, g, μ₀)
        @test maximum(imag, vals) ≈ 0.3306185 atol=1e-7 # default ArnoldiMethod tolerance for Float64 is √eps ≈ 1.5e-8

        # also test BdG in p-space. Skip sin case because that requires 2M+1 harmonics for dimensions to match, but the result is then inaccurate
        if basis != :sin
            xh_double = XSpaceHamiltonian{:dense}([xlimits], 𝑈; basis, 𝑈_iseven=true, M=2M, δ)
            _, sol = find_stationary(xh_double, [𝜓₀], μ₀, [g;;])
            vals, vecs = XSpaceHamiltonians.bdg_spectrum_pspace(xh, sol.u, g, μ₀; ψ_iseven=true)
            @test maximum(imag, vals) ≈ 0.3306185 atol=1e-7
        end
    end
end