using LinearAlgebra

@testset "Fourier-block x-space preconditioner" begin
    xlims = [(0.0, 2π)]

    U = [x -> 2.0  x -> 0.3;
         x -> 0.3  x -> 3.0]
    xh = XSpaceHamiltonian(xlims, U; basis=:cis, M=4)
    prec = QuantumHamiltonians.FourierBlockPreconditioner(xh; shift=0.0)
    b = randn(ComplexF64, xh.nc * xh.B)
    x = similar(b)
    ldiv!(x, prec, b)
    Hx = similar(b)
    mul!(Hx, xh, x)
    @test Hx ≈ b rtol=1e-12 atol=1e-12

    # Constant gauge fields are included exactly by the -2Ā·p term.
    xh_gauge = XSpaceHamiltonian(xlims, x -> 2.0, [x -> 0.25]; basis=:cis, M=4, Γ=0.1)
    gauge_prec = QuantumHamiltonians.FourierBlockPreconditioner(xh_gauge; shift=0.0)
    gauge_b = randn(ComplexF64, xh_gauge.B)
    gauge_x = similar(gauge_b)
    ldiv!(gauge_x, gauge_prec, gauge_b)
    gauge_Hx = similar(gauge_b)
    mul!(gauge_Hx, xh_gauge, gauge_x)
    @test gauge_Hx ≈ gauge_b rtol=1e-12 atol=1e-12

    # A nonuniform gauge field is approximated by its mean but remains a valid preconditioner.
    xh_variable_gauge = XSpaceHamiltonian(xlims, x -> 2.0, [x -> 0.25 + 0.1sin(x)]; basis=:cis, M=4)
    variable_prec = QuantumHamiltonians.FourierBlockPreconditioner(xh_variable_gauge; shift=0.0)
    variable_x = similar(gauge_b)
    ldiv!(variable_x, variable_prec, gauge_b)
    @test all(isfinite, variable_x)

    # The transform path also supports the real sine and cosine bases.
    for (basis, M) in ((:sin, 3), (:cos, 4))
        xh_real = XSpaceHamiltonian(xlims, x -> 2.0; basis, M)
        real_prec = QuantumHamiltonians.FourierBlockPreconditioner(xh_real; shift=0.0)
        real_b = randn(xh_real.B)
        real_x = similar(real_b)
        ldiv!(real_x, real_prec, real_b)
        real_Hx = similar(real_b)
        mul!(real_Hx, xh_real, real_x)
        @test real_Hx ≈ real_b rtol=1e-12 atol=1e-12
    end

    # The automatic shift keeps the zero momentum block invertible.
    xh_zero = XSpaceHamiltonian(xlims, nothing; basis=:cis, M=4)
    @test_nowarn QuantumHamiltonians.FourierBlockPreconditioner(xh_zero)
    @test_throws SingularException QuantumHamiltonians.FourierBlockPreconditioner(xh_zero; shift=0.0)
    @test_throws ArgumentError QuantumHamiltonians.diagonalize(xh; nev=1, invert=false, preconditioner=:unsupported)

    ε_plain, _ = QuantumHamiltonians.diagonalize(xh; nev=2, invert=true, tol=1e-10)
    ε_prec, _ = QuantumHamiltonians.diagonalize(xh; nev=2, invert=true, tol=1e-10, preconditioner=:fourier_block, preconditioner_shift=0.0)
    @test ε_prec ≈ ε_plain rtol=1e-8 atol=1e-10
end
