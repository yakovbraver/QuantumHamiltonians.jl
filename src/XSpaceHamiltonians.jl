module XSpaceHamiltonians

import FFTW
using ArnoldiMethod: partialschur
using LinearAlgebra: Hermitian, diagind, Diagonal, ldiv!, factorize
using LinearMaps: LinearMap

export DirichletHamiltonian, diagonalize!, make_wavefunction

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
    Lx, Ly = xlims[2]-xlims[1], ylims[2]-ylims[1] # area dimensions

    M = 2N + 1
    xs = range(0, Lx, M)
    ys = range(0, Ly, M)
    
    dx, dy = Lx/M, Ly/M
    f = dx/Lx * dy/Ly

    u = [𝑈(x, y) for x in xs, y in ys]

    F = FFTW.plan_r2r(u, FFTW.RODFT00)
    U = F * u * f |> dct_to_matrix

    Δ = Diagonal([-π^2 * ((jx/Lx)^2 + (jy/Ly)^2) for jx in 1:N for jy in 1:N])
    
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
    N = (M-1) ÷ 2 # size of each block of the Hamiltonian
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
    (;Lx, Ly, xlims, ylims, V) = dh
    N = Int(√size(V, 1))
    xs = range(xlims[1], xlims[2], nx)
    ys = range(ylims[1], ylims[2], ny)
    ψ = Matrix{eltype(V)}(undef, nx, ny)
    for (iy, y) in enumerate(ys), (ix, x) in enumerate(xs)
        ψ[ix, iy] = sum(V[(jx-1)N+jy, stateno]sin(π*jx*x/Lx)sin(π*jy*y/Ly) for jx in 1:N for jy in 1:N) * 2 / √(Lx*Ly)
    end
    return xs, ys, ψ
end

function diagonalize!(dh::DirichletHamiltonian; nev::Integer)
    S, = partialschur(construct_linear_map(Hermitian(dh.H)); nev, which=:LM);
    dh.V = S.Q
    dh.ε = inv.(real.(S.eigenvalues))
    
    # S, = partialschur(Hermitian(dh.H); nev, which=:SR)
    # dh.V = S.Q
    # dh.ε = S.eigenvalues
end

function construct_linear_map(A)
    F = factorize(A)
    LinearMap{eltype(A)}((y, x) -> ldiv!(y, F, x), size(A,1), ismutating=true)
end

# """
# Set to zero values of `u` that are `threshold` times smaller (by absolute magnitude) than the largest.
# Based on the resulting number of nonzero elements in `u`, count the number of values that will be stored in 𝐻.
# """
# function filter_count!(u::AbstractMatrix{<:Number}; fft_threshold::Real)
#     n_elem = 0
#     M = size(u, 2) ÷ 2
#     N = size(u, 1) # if `u` is really the result of `rfft`, then `N == M+1`, but we keep the calculation a bit more general
#     # do the first row of `u`, i.e. the diagonal blocks of 𝐻, separately
#     for c in axes(u, 2)
#         r = 1
#         if abs(u[r, c]) < fft_threshold
#             u[r, c] = 0
#         else
#             if c < M+1
#                 n_elem += (N - (r-1)) * (M+1 - (c-1)) # number of blocks in which `u[r, c]` will appear × number of times it will appear within each block
#             elseif c == M+1
#                 n_elem += 2(N - (r-1)) * (M+1 - (c-1))
#             else
#                 n_elem += (N - (r-1)) * (c - M)
#             end
#         end
#     end
#     for c in axes(u, 2), r in 2:size(u, 1)
#         if abs(u[r, c]) < fft_threshold
#             u[r, c] = 0
#         else
#             if c < M+1
#                 n_elem += 2(N - (r-1)) * (M+1 - (c-1))
#             elseif c == M+1
#                 n_elem += 4(N - (r-1)) * (M+1 - (c-1))
#             else
#                 n_elem += 2(N - (r-1)) * (c - M)
#             end
#         end
#     end
#     return n_elem
# end

# """
# Based on results of a real 2D fft `u`, return `rows, cols, vals` tuple for constructing a sparse matrix.
# Optionally, a tuple `δ` of shifts in 𝑥 and 𝑦 directions can be supplied.
# """
# function fft_to_matrix!(rows, cols, vals, u, δ::Tuple{<:Real,<:Real})
#     L = π # periodicity of the potential
#     M = size(u, 2) ÷ 2 # M + 1 gives the size of each block; `size(u, 1)` gives the number of block-rows (= number of block-cols)
#     counter = 1

#     # it is assumed that u[1, 1] == 0 -- otherwise, one would also need to prevent double pushing of the diagonal elements
#     for c_u in axes(u, 2), r_u in axes(u, 1) # iterate over columns and rows of `u`
#         u[r_u, c_u] == 0 && continue
#         e = c_u <= M+1 ? cispi(2/L*(c_u-1)*δ[1]) : cispi(2/L*(c_u-(2M+1))*δ[1])
#         val = u[r_u, c_u] * e * cispi(2/L*(r_u-1)*δ[2])
#         for r_b in r_u:size(u, 1) # a value from `r_u`th row of `u` will be put in block-rows of `H` from `r_u`th to `M+1`th. For actual applications, `size(u, 1) == M+1`
#             c_b = r_b - r_u + 1 # block-column where to place the value
#             if c_u <= M # for `c_u` ≤ `M`, the value from `c_u`th column of `u` will be put to the `c_u`th lower diagonal of the block
#                 for (r, c) in zip(c_u:M+1, 1:M+2-c_u)
#                     push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize=M+1, val)
#                     counter += 2
#                 end
#             elseif c_u == M+1 # for `c_u` = `M+1`, the value from `c_u`th column of `u` will be put to lower left and upper right corners of the block
#                 push_vals!(rows, cols, vals, counter; r_b, c_b, r=M+1, c=1, blocksize=M+1, val)
#                 counter += 2
#                 if r_b != c_b
#                     push_vals!(rows, cols, vals, counter; r_b, c_b, r=1, c=M+1, blocksize=M+1, val) # if we're in the diagonal block, then the upper right corner is conjugate to lower left and has already been pushed
#                     counter += 2
#                 end
#             else # for `c_u` ≥ `M+2`, the value from `c_u`th column of `u` will be put to the `2M+2-c_u`th upper diagonal of the block
#                 if r_b != c_b # if `r_b == c_b`, then upper diagonal of the block has already been filled by pushing the conjugate element
#                     c_u_inv = 2M+2 - c_u
#                     for (r, c) in zip(1:M+2-c_u_inv, c_u_inv:M+1)
#                         push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize=M+1, val)
#                         counter += 2
#                     end
#                 end
#             end
#         end
#     end
# end


# """
# Push value `val` stored at (`r`, `c`) in some matrix to the block (`r_b`, `c_b`) of a sparse matrix encoded in `rows`, `cols`, `vals`.
# `counter` shows where to push. The complex-conjugate element is also pushed.
# """
# function push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize, val)
#     i = (r_b-1)*blocksize + r
#     j = (c_b-1)*blocksize + c
#     rows[counter] = i
#     cols[counter] = j
#     vals[counter] = val
#     # conjugate
#     rows[counter+1] = j
#     cols[counter+1] = i
#     vals[counter+1] = val'
# end

# """
# Calculate ground state energy dispersion for a quarter of the BZ, since ℤ₄ symmetry is assumed.
# This quarter is discretised with `n_q` points in each direction.
# """
# function spectrum(gf::Hamiltonian{Float}; n_q::Integer) where {Float<:AbstractFloat}
#     E = Matrix{Float}(undef, n_q, n_q)

#     H = sparse(gf.H_rows, gf.H_cols, gf.H_vals)
#     H_vals = nonzeros(H)
#     diagidx = findall(==(0), H_vals) # find indices of diagonal elements -- we saved zeros there (see `constructH`)

#     B = Int(√size(H, 1)) # size of each block in `H`
#     j_max = (B - 1) ÷ 2  # index for each block in `H` will run in `-j_max:j_max`, giving `B` values in total
#     qs = range(0, 1, length=n_q) # BZ is (-1 ≤ 𝑞ₓ, 𝑞𝑦 ≤ 1), but it's enough to consider a triangle 0 ≤ 𝑞ₓ ≤ 1, 0 ≤ 𝑞𝑦 ≤ 𝑞ₓ
#     for (iqx, qx) in enumerate(qs), iqy in iqx:n_q
#         qy = qs[iqy]
#         for (j, jx) in enumerate(-j_max:j_max), (i, jy) in enumerate(-j_max:j_max)
#             H_vals[diagidx[(j-1)B+i]] = gf.u₀₀ + qx^2 + qy^2 + 4(qx*jx + qy*jy) + 4(jx^2 + jy^2)
#         end
#         # vals, _, _ = eigsolve(H, 1, :SR, tol=(Float == Float32 ? 1e-6 : 1e-12))
#         vals = eigvals(Hermitian(Matrix(H)))
#         E[iqy, iqx] = E[iqx, iqy] = vals[1]
#     end
#     return E
# end

# """
# Calculate energies and wavefunctions at zero quasimomenta for `nbands` lowest bands.
# """
# function q0_states(gf::Hamiltonian{Float}) where {Float<:AbstractFloat}
#     H = sparse(gf.H_rows, gf.H_cols, gf.H_vals) |> Matrix |> Hermitian
#     B = Int(√size(H, 1)) # size of each block in `H`
#     j_max = (B - 1) ÷ 2  # index for each block in `H` will run in `-j_max:j_max`, giving `B` values in total
#     H[diagind(H)] .= [gf.u₀₀ + 4(jx^2 + jy^2) for jx in -j_max:j_max for jy in -j_max:j_max]
#     f = eigen(H)
#     return f.values, f.vectors
# end

# """
# Construct wavefunction `wf` on a grid having `nx` points in `x` and `y` direction. Use odd `nx` for nice results.
# `v` is one of the vectors output from [`q0_states`](@ref).
# Return (`xs`, `wf`), where `xs` is the grid in one dimension.
# """
# function make_wavefunction(v::AbstractVector{<:Number}, nx::Integer)
#     B = Int(√length(v))
#     j_max = (B - 1) ÷ 2
#     L = π # spatial period
#     xs = range(0, L, nx)
#     wf = Matrix{eltype(v)}(undef, nx, nx)
#     for (iy, y) in enumerate(xs), (ix, x) in enumerate(xs)
#         wf[ix, iy] = sum(v[(j-1)B+i]cis(2jx*x + 2jy*y) for (j, jx) in enumerate(-j_max:j_max)
#                                                        for (i, jy) in enumerate(-j_max:j_max)) / L
#     end
#     return xs, wf
# end

# """
# Calculate energy dispersion for all pairs of quasimomenta described by `qxs` and `qys`.
# Save `nsaves` lowest states; if not passed, all states are saved.
# """
# function spectrum(gf::Hamiltonian{Float}, qxs::AbstractVector{<:Real}, qys::AbstractVector{<:Real}; nsaves::Integer=0) where {Float<:AbstractFloat}
#     H = sparse(gf.H_rows, gf.H_cols, gf.H_vals)
#     H_vals = nonzeros(H)
#     diagidx = findall(==(0), H_vals) # find indices of diagonal elements -- we saved zeros there (see `constructH`)
    
#     (nsaves == 0) && (nsaves = size(H, 1))
#     E = Array{Float}(undef, nsaves, length(qxs), length(qys))

#     M = Int(√size(H, 1)) # size of each block in `H`
#     j_max = (M - 1) ÷ 2  # index for each block in `H` will run in `-j_max:j_max`, giving `M` values in total
#     for (iqy, qy) in enumerate(qys), (iqx, qx) in enumerate(qxs)
#         for (j, jx) in enumerate(-j_max:j_max), (i, jy) in enumerate(-j_max:j_max)
#             H_vals[diagidx[(j-1)M+i]] = gf.u₀₀ + qx^2 + qy^2 + 4(qx*jx + qy*jy) + 4(jx^2 + jy^2)
#         end
#         # vals, _, _ = eigsolve(H, nsaves, :SR, tol=(Float == Float32 ? 1e-6 : 1e-12))
#         vals = eigvals(Hermitian(Matrix(H)))
#         E[:, iqx, iqy] = vals[1:nsaves]
#     end
#     return E
# end

end