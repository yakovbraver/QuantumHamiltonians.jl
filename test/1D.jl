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