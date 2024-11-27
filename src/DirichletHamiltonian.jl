mutable struct DirichletHamiltonian{R<:Real,T<:Number} # in practice `T` will be `R` or `Complex{R}`
    xlims::Tuple{R, R}
    ylims::Tuple{R, R}
    Lx::R # period along 𝑥
    Ly::R # period along 𝑦
    H::Matrix{T}
    ε::Vector{R} # eigenvalues
    V::Matrix{T} # eigenvectors matrix
end

"""
Construct a `Hamiltonian` object.
`N` is the desired size of each block of the Hamiltonian, power of 2 minus one recommended.
The size of the Hamiltonian will be `N`² × `N`².
To make sure that the resulting Hamiltonian matrix is of the desired type `T`, the type of elements of `xlims`, `ylims`,
and the return type of the passed functions has to be the same. E.g., if all are `Float32`, then `T` will be `Float32` if only `𝑈` is passed,
and `ComplexF32` if `𝐴`'s are passed. Inconsistency in the types of arguments will result in widening.
"""
function DirichletHamiltonian(xlims::Tuple{<:Real,<:Real}, ylims::Tuple{<:Real,<:Real}; 𝑈::Union{Function,Nothing}=nothing, 𝐴_x::Union{Function,Nothing}=nothing,
                              𝐴_y::Union{Function,Nothing}=nothing, N=63)
    Lx, Ly = xlims[2]-xlims[1], ylims[2]-ylims[1]

    M = 2N + 2
    xs = range(xlims[1], xlims[2], M)
    ys = range(ylims[1], ylims[2], M)
    
    dx, dy = Lx/M, Ly/M
    f = dx/Lx * dy/Ly

    u = [𝑈(x, y) for x in xs, y in ys]

    F = FFTW.plan_r2r(u, FFTW.RODFT00)
    U = F * u * f |> dct_to_matrix

    Δ = Diagonal(typeof(Lx)[-π^2 * ((jx/Lx)^2 + (jy/Ly)^2) for jx in 1:N for jy in 1:N])
    
    if 𝐴_x === nothing
        H = -Δ + U
    else
        a_x = [𝐴_x(x, y) for x in xs, y in ys]
        a_y = [𝐴_y(x, y) for x in xs, y in ys]
        
        A_x = F * a_x * f |> dct_to_matrix
        A_y = F * a_y * f |> dct_to_matrix
        
        ∂_x = make_∂_x(N, Lx)
        ∂_y = make_∂_y(N, Ly)
        
        H = -Δ + im*(A_x*∂_x + A_y*∂_y + ∂_x*A_x + ∂_y*A_y) + A_x^2 + A_y^2 + U
    end
    return DirichletHamiltonian(xlims, ylims, Lx, Ly, H, typeof(Lx)[], eltype(H)[;;])
end

"""
Based on results of 2D DCT `u`, return the matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
"""
function dct_to_matrix(u)
    M = size(u, 1)
    N = (M-2) ÷ 2 # size of each block of the Hamiltonian
    H = Matrix{eltype(u)}(undef, N^2, N^2)
    for jx in 1:N, jy in 1:N, j′x in 1:N, j′y in 1:N
        j₋x = abs(j′x-jx)
        j₋y = abs(j′y-jy)
        H[(j′x-1)N+j′y, (jx-1)N+jy] = (u[j₋x+1, j₋y+1] - u[j₋x+1, j′y+jy+1] - u[j′x+jx+1, j₋y+1] + u[j′x+jx+1, j′y+jy+1]) / 4
    end
    return H
end

function make_∂_x(N, Lx)
    ∂_x = zeros(typeof(Lx), N^2, N^2)
    for jx in 1:N, jy in 1:N, j′x in 1+isodd(jx):2:N
        val = 1/(j′x+jx)
        j′x != jx && (val += 1/(j′x-jx))
        ∂_x[(j′x-1)N+jy, (jx-1)N+jy] = 2jx/Lx * val
    end
    return ∂_x
end

function make_∂_y(N, Ly)
    ∂_y = zeros(typeof(Ly), N^2, N^2)
    for jx in 1:N, jy in 1:N, j′y in 1+isodd(jy):2:N
        val = 1/(j′y+jy)
        j′y != jy && (val += 1/(j′y-jy))
        ∂_y[(jx-1)N+j′y, (jx-1)N+jy] = 2jy/Ly * val
    end
    return ∂_y
end

"""
Construct wavefunction of state number `stateno` on a grid having `nx` points in `x` and `ny` points in `y` direction.
Return (`xs`, `ys`, `ψ`).
"""
function make_wavefunction(dh::DirichletHamiltonian, stateno::Integer, nx::Integer, ny::Integer)
    (;Lx, Ly, V) = dh
    N = Int(√size(V, 1))
    xs = range(0, Lx, nx)
    ys = range(0, Ly, ny)
    ψ = Matrix{eltype(V)}(undef, nx, ny)
    for (iy, y) in enumerate(ys), (ix, x) in enumerate(xs)
        ψ[ix, iy] = sum(V[(jx-1)N+jy, stateno]sin(π*jx*x/Lx)sin(π*jy*y/Ly) for jx in 1:N for jy in 1:N) * 2 / √(Lx*Ly)
    end
    return xs, ys, ψ
end

function diagonalize!(dh::DirichletHamiltonian; nev::Integer)
    if nev == 0
        dh.ε, dh.V = eigen(Hermitian(dh.H))
    else
        S, info = partialschur(construct_linear_map(Hermitian(dh.H)); nev, which=:LM);
        @show info
        dh.V = S.Q
        dh.ε = inv.(real.(S.eigenvalues))
    end

    # S, = partialschur(Hermitian(dh.H); nev, which=:SR)
    # dh.V = S.Q
    # dh.ε = S.eigenvalues
end

function construct_linear_map(A)
    F = factorize(A)
    LinearMap{eltype(A)}((y, x) -> ldiv!(y, F, x), size(A,1), ismutating=true)
end