using LinearAlgebra

@testset "Block-Jacobi x-space preconditioner" begin
    xlims = [(0.0, 2π)]

    # take constant `U` so that p-space version is block-diagonal and hence inversion can be calculated exactly (using LU for each mode)
    U = [x -> 2.0  x -> 0.3;
         x -> 0.3  x -> 3.0]
    for basis in (:cis, :sin, :cos), T in (Float64, ComplexF64)
        xh = XSpaceHamiltonian(xlims, U; basis, M=(basis == :sin ? 3 : 4))
        prec = QuantumHamiltonians.JacobiPreconditioner(xh)
        b = randn(T, xh.nc * xh.B)
        x = similar(b)
        ldiv!(x, prec, b) # should have the effect of x = H⁻¹b
        Hx = similar(b)
        mul!(Hx, xh, x) # should have the effect of Hx = HH⁻¹b = b
        @test Hx ≈ b rtol=1e-12 atol=1e-12
    end

    # single-component case with constant 𝐴, which can also be inverted exactly
    xh_gauge = XSpaceHamiltonian(xlims, x -> 2.0, [x -> 0.25]; basis=:cis, M=4, Γ=0.1)
    gauge_prec = QuantumHamiltonians.JacobiPreconditioner(xh_gauge)
    gauge_b = randn(ComplexF64, xh_gauge.B)
    gauge_x = similar(gauge_b)
    ldiv!(gauge_x, gauge_prec, gauge_b)
    gauge_Hx = similar(gauge_b)
    mul!(gauge_Hx, xh_gauge, gauge_x)
    @test gauge_Hx ≈ gauge_b rtol=1e-12 atol=1e-12

    # A nonuniform gauge field is approximated by its mean but remains a valid preconditioner.
    xh_variable_gauge = XSpaceHamiltonian(xlims, x -> 2.0, [x -> 0.25 + 0.1sin(x)]; basis=:cis, M=4)
    variable_prec = QuantumHamiltonians.JacobiPreconditioner(xh_variable_gauge)
    variable_x = similar(gauge_b)
    ldiv!(variable_x, variable_prec, gauge_b)
    @test all(isfinite, variable_x)

    # Test with no potential so that preconditioner is singular -- the automatic shift should keep the zero momentum block invertible.
    xh_zero = XSpaceHamiltonian(xlims, nothing; basis=:cis, M=4)
    @test_nowarn QuantumHamiltonians.JacobiPreconditioner(xh_zero)

    # Test diagonalisation
    # ε_plain, _ = QuantumHamiltonians.diagonalize(xh; nev=2, invert=true, tol=1e-10)
    # ε_prec, _ = QuantumHamiltonians.diagonalize(xh; nev=2, invert=true, tol=1e-10, preconditioner=:block_jacobi)
    # @test ε_prec ≈ ε_plain rtol=1e-8 atol=1e-10
end

@testset "Jacobi x-space preconditioner" begin
    xlims = [(0.0, 2π)]

    # take constant `U` so that p-space version is block-diagonal and hence inversion can be calculated exactly
    U = [x -> 2.23  x -> 0.0;
         x -> 0.0  x -> 4.234]
    for basis in (:cis, :sin, :cos), T in (Float64, ComplexF64)
        xh = XSpaceHamiltonian(xlims, U; basis, M=(basis == :sin ? 3 : 4))
        prec = QuantumHamiltonians.JacobiPreconditioner(xh; type=:simple)
        b = randn(T, xh.nc * xh.B)
        x = similar(b)
        ldiv!(x, prec, b) # should have the effect of x = H⁻¹b
        Hx = similar(b)
        mul!(Hx, xh, x) # should have the effect of Hx = HH⁻¹b = b
        @test Hx ≈ b rtol=1e-12 atol=1e-12
    end
end
