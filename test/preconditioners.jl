using LinearAlgebra

@testset "Fourier-block x-space preconditioner" begin
    xlims = [(0.0, 2π)]

    # take constant `U` so that p-space version is block-diagonal and hence inversion can be calculated exactly
    U = [x -> 2.0  x -> 0.3;
         x -> 0.3  x -> 3.0]
    xh = XSpaceHamiltonian(xlims, U; basis=:cis, M=4)
    prec = QuantumHamiltonians.BlockJacobiPreconditioner(xh)
    b = randn(ComplexF64, xh.nc * xh.B)
    x = similar(b)
    ldiv!(x, prec, b) # should have the effect of x = H⁻¹b
    Hx = similar(b)
    mul!(Hx, xh, x) # should have the effect of Hx = HH⁻¹b = b
    @test Hx ≈ b rtol=1e-12 atol=1e-12

    # single-component case with constant 𝐴, which can also be inverted exactly
    xh_gauge = XSpaceHamiltonian(xlims, x -> 2.0, [x -> 0.25]; basis=:cis, M=4, Γ=0.1)
    gauge_prec = QuantumHamiltonians.BlockJacobiPreconditioner(xh_gauge)
    gauge_b = randn(ComplexF64, xh_gauge.B)
    gauge_x = similar(gauge_b)
    ldiv!(gauge_x, gauge_prec, gauge_b)
    gauge_Hx = similar(gauge_b)
    mul!(gauge_Hx, xh_gauge, gauge_x)
    @test gauge_Hx ≈ gauge_b rtol=1e-12 atol=1e-12

    # A nonuniform gauge field is approximated by its mean but remains a valid preconditioner.
    xh_variable_gauge = XSpaceHamiltonian(xlims, x -> 2.0, [x -> 0.25 + 0.1sin(x)]; basis=:cis, M=4)
    variable_prec = QuantumHamiltonians.BlockJacobiPreconditioner(xh_variable_gauge)
    variable_x = similar(gauge_b)
    ldiv!(variable_x, variable_prec, gauge_b)
    @test all(isfinite, variable_x)

    # The transform path also supports the real sine and cosine bases.
    for (basis, M) in ((:sin, 3), (:cos, 4))
        xh_real = XSpaceHamiltonian(xlims, x -> 2.0; basis, M)
        real_prec = QuantumHamiltonians.BlockJacobiPreconditioner(xh_real)
        real_b = randn(xh_real.B)
        real_x = similar(real_b)
        ldiv!(real_x, real_prec, real_b)
        real_Hx = similar(real_b)
        mul!(real_Hx, xh_real, real_x)
        @test real_Hx ≈ real_b rtol=1e-12 atol=1e-12
    end

    # The automatic shift keeps the zero momentum block invertible.
    xh_zero = XSpaceHamiltonian(xlims, nothing; basis=:cis, M=4)
    @test_nowarn QuantumHamiltonians.BlockJacobiPreconditioner(xh_zero)
    @test_throws ArgumentError QuantumHamiltonians.diagonalize(xh; nev=1, invert=false, preconditioner=:unsupported)

    ε_plain, _ = QuantumHamiltonians.diagonalize(xh; nev=2, invert=true, tol=1e-10)
    ε_prec, _ = QuantumHamiltonians.diagonalize(xh; nev=2, invert=true, tol=1e-10, preconditioner=:fourier_block)
    @test ε_prec ≈ ε_plain rtol=1e-8 atol=1e-10

    # laplace_prec = QuantumHamiltonians.LaplacePreconditioner(xh; shift=1.0)
    # laplace_b = randn(ComplexF64, xh.nc * xh.B)
    # laplace_x = similar(laplace_b)
    # ldiv!(laplace_x, laplace_prec, laplace_b)
    # laplace_expected = similar(laplace_b)
    # for c in 1:xh.nc
    #     window = (c - 1)xh.B+1:c*xh.B
    #     tmp = similar(laplace_b[window])
    #     QuantumHamiltonians.transform!(tmp, xh.ft, laplace_b[window]; direction=:forward)
    #     @views tmp .*= inv.(xh.∇² .+ 1.0)
    #     QuantumHamiltonians.transform!(@view(laplace_expected[window]), xh.ft, tmp; direction=:backward, normalise=true)
    # end
    # @test laplace_x ≈ laplace_expected rtol=1e-12 atol=1e-12

    # ε_laplace, _ = QuantumHamiltonians.diagonalize(xh; nev=2, invert=true, tol=1e-10,
    #                                                 preconditioner=:laplace, preconditioner_shift=1.0)
    # @test ε_laplace ≈ ε_plain rtol=1e-8 atol=1e-10
end
