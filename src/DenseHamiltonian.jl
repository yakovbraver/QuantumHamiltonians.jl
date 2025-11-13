"""
A type representing a spatial [𝑟 = (𝑥, 𝑦)], possibly quasimomentum-dependent Hamiltonian
    𝐻(𝑟) = [-i𝛿∇ + 𝑞 - 𝐴(𝑟)]² + 𝑈(𝑟)
as a dense matrix.
"""
mutable struct DenseHamiltonian{R<:Real,T<:Number} # in practice `T` shoudld be `R` or `Complex{R}` -- always check this. If this is not the case, probably your 𝑈 or 𝐴 do not return R's.
    xlims::Tuple{R, R}
    ylims::Tuple{R, R}
    Lx::R # length along 𝑥
    Ly::R # length along 𝑦
    δ::R # coefficient of the momentum term: -iδ∇
    isperiodic::Bool
    M::Int # maximum harmonic number (will use -M:M for periodic, 1:M for nonperiodic)
    𝑈::Union{Function,Nothing}
    𝐴_x::Union{Function,Nothing}
    𝐴_y::Union{Function,Nothing}
    H::Matrix{T}
    ε::Vector{R} # eigenvalues
    V::Matrix{T} # eigenvectors matrix
    ε_q::Array{R,3} # ε_q[n, iqx, iqy] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    V_q::Array{T,4} # V_q[:, n, iqx, iqy] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)
end

"""
Construct a `DenseHamiltonian` object.
`M` is the maximum harmonic number. In the periodic case, the size of the Hamiltonian will be (`2M`)² × (`2M`)².
In nonperiodic case, the size will be `(2M+1)`² × `(2M+1)`².
To make sure that the resulting Hamiltonian matrix is of the desired type `T`, the type of elements of `xlims`, `ylims`,
and the return type of the passed functions has to be the same. E.g., if all are `Float32`, then `T` will be `Float32` if only `𝑈` is passed,
and `ComplexF32` if `𝐴`'s are passed. Inconsistency in the types of arguments will result in widening.
"""
function DenseHamiltonian(xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R), 𝑈::Union{Function,Nothing}=nothing, 𝐴_x::Union{Function,Nothing}=nothing,
                          𝐴_y::Union{Function,Nothing}=nothing) where R <: Real
    Lx, Ly = xlims[2]-xlims[1], ylims[2]-ylims[1]
    
    PI = R(π) # π of the working type to prevent widening

    if isperiodic
        N = 4M # number of points for FFT. This will yield harmonics from -2M to 2M
        dx, dy = Lx/N, Ly/N
        xs = range(xlims[1], xlims[2]-dx, N)
        ys = range(ylims[1], ylims[2]-dy, N)

        f = dx/Lx * dy/Ly
        u = [𝑈(x, y) for x in xs, y in ys]

        F = FFTW.plan_rfft(u)
        U = F * u * f |> dft_to_matrix
    
        if 𝐴_x === nothing
            Δ = Diagonal([-(2PI*δ)^2 * ((jx/Lx)^2 + (jy/Ly)^2) for jx in -M:M for jy in -M:M]) # this is δ²Δ
            H = -Δ + U
        else
            a_x = [𝐴_x(x, y) for x in xs, y in ys]
            a_y = [𝐴_y(x, y) for x in xs, y in ys]
    
            A_x = F * a_x * f |> dft_to_matrix
            A_y = F * a_y * f |> dft_to_matrix
            
            ∂_x = Diagonal([2PI * δ * jx/Lx for jx in -M:M for jy in -M:M]) # this is -iδ∂ₓ
            ∂_y = Diagonal([2PI * δ * jy/Ly for jx in -M:M for jy in -M:M]) # this is -iδ∂y
            
            # H = -Δ + im*(A_x*∂_x + A_y*∂_y + ∂_x*A_x + ∂_y*A_y) + A_x^2 + A_y^2 + U
            H = (∂_x - A_x)^2 + (∂_y - A_y)^2 + U
        end
    else # non-periodic
        N = 2M + 1
        xs = range(xlims[1], xlims[2], N)
        ys = range(ylims[1], ylims[2], N)
        dx, dy = xs[2]-xs[1], ys[2]-ys[1]

        f = dx/Lx * dy/Ly
        u = [𝑈(x, y) for x in xs, y in ys]

        F = FFTW.plan_r2r!(u, FFTW.REDFT00)
        (F * u) .*= f
        U = dct_to_matrix(u)

        if 𝐴_x === nothing
            Δ = Diagonal([-(PI*δ)^2 * ((jx/Lx)^2 + (jy/Ly)^2) for jx in 1:M for jy in 1:M]) # this is δ²Δ
            H = -Δ + U
        else
            a_x = [𝐴_x(x, y) for x in xs, y in ys]
            a_y = [𝐴_y(x, y) for x in xs, y in ys]
            
            (F * a_x) .*= f
            A_x = dct_to_matrix(a_x)
            (F * a_y) .*= f
            A_y = dct_to_matrix(a_y)
            
            ∂_x = make_∂_x(M, Lx)
            ∂_y = make_∂_y(M, Ly)
            
            # H = -Δ + im*(A_x*∂_x + A_y*∂_y + ∂_x*A_x + ∂_y*A_y) + A_x^2 + A_y^2 + U
            # H = sum_parts(A_x, A_y, ∂_y, ∂_x, U, Δ)
            H = (-im*δ*∂_x - A_x)^2 + (-im*δ*∂_y - A_y)^2 + U
        end
    end
    
    return DenseHamiltonian(xlims, ylims, Lx, Ly, δ, isperiodic, M, 𝑈, 𝐴_x, 𝐴_y, H, R[], eltype(H)[;;], R[;;;], eltype(H)[;;;;])
end

# """
# More efficient way of calculating
#     H = -Δ + im*(A_x*∂_x + A_y*∂_y + ∂_x*A_x + ∂_y*A_y) + A_x^2 + A_y^2 + U
# Currently intended for non-periodic calculation.
# """
# function sum_parts(A_x::Matrix{<:Real}, A_y::Matrix{<:Real}, ∂_y::Matrix{<:Real}, ∂_x::Matrix{<:Real}, U::Matrix{<:Real}, Δ)
#     H = complex.(U)
#     H += -Δ
#     unit = one(eltype(A_x))
#     null = zero(eltype(A_x))
#     symm!('L', 'L', unit, A_x, ∂_x, null, U) # we start using `U` as a buffer
#     symm!('R', 'L', unit, A_x, ∂_x, unit, U)
#     symm!('L', 'L', unit, A_y, ∂_y, unit, U)
#     symm!('R', 'L', unit, A_y, ∂_y, unit, U)
#     H .+= U .* im
#     symm!('L', 'L', unit, A_x, A_x, null, U)
#     H += U
#     symm!('L', 'L', unit, A_y, A_y, null, U)
#     H += U
#     return H
# end

"""
Based on results of 2D DCT `u`, return the matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
"""
function dct_to_matrix(u)
    N = size(u, 1) # number of points used for FFT
    M = (N-1) ÷ 2 # maximum harmonic number
    H = Matrix{eltype(u)}(undef, M^2, M^2)
    @floop for jx in 1:M
        for jy in 1:M, j′x in 1:M, j′y in 1:M
            j₋x = abs(j′x-jx)
            j₋y = abs(j′y-jy)
            H[(j′x-1)M+j′y, (jx-1)M+jy] = (u[j₋x+1, j₋y+1] - u[j₋x+1, j′y+jy+1] - u[j′x+jx+1, j₋y+1] + u[j′x+jx+1, j′y+jy+1]) / 4
        end
    end
    return H
end

"""
Based on results of a real 2D fft `u`, return the matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
"""
function dft_to_matrix(u)
    B = size(u, 2) ÷ 2 + 1 # the size of each block; `size(u, 1)` gives the number of block-rows (= number of block-cols)
    H = zeros(eltype(u), B^2, B^2)
    H[diagind(H)] .= u[1, 1] # save the secular component
    u[1, 1] = 0 # remove because it breaks the structure of the loop below if included

    # it is assumed that u[1, 1] == 0 -- otherwise, one would also need to prevent double pushing of the diagonal elements
    @floop for c_u in axes(u, 2)
        for r_u in axes(u, 1) # iterate over columns and rows of `u`
            u[r_u, c_u] == 0 && continue
            val = u[r_u, c_u]
            for r_b in r_u:size(u, 1) # a value from `r_u`th row of `u` will be put in block-rows of `H` from `r_u`th to `M+1`th. For actual applications, `size(u, 1) == M+1`
                c_b = r_b - r_u + 1 # block-column where to place the value
                if c_u < B # for `c_u` < `B`, the value from `c_u`th column of `u` will be put to the `c_u`th lower diagonal of the block
                    for (r, c) in zip(c_u:B, 1:B+1-c_u)
                        push_vals!(H; r_b, c_b, r, c, blocksize=B, val)
                    end
                elseif c_u == B # for `c_u` = `B`, the value from `c_u`th column of `u` will be put to lower left and upper right corners of the block
                    push_vals!(H; r_b, c_b, r=B, c=1, blocksize=B, val)
                    if r_b != c_b
                        push_vals!(H; r_b, c_b, r=1, c=B, blocksize=B, val) # if we're in the diagonal block, then the upper right corner is conjugate to lower left and has already been pushed
                    end
                else # for `c_u` > `B`, the value from `c_u`th column of `u` will be put to the `2M+2-c_u`th upper diagonal of the block
                    if r_b != c_b # if `r_b == c_b`, then upper diagonal of the block has already been filled by pushing the conjugate element
                        c_u_inv = 2B - c_u
                        for (r, c) in zip(1:B+1-c_u_inv, c_u_inv:B)
                            push_vals!(H; r_b, c_b, r, c, blocksize=B, val)
                        end
                    end
                end
            end
        end
    end
    return H
end

"""
Push value `val` to element (`r`, `c`) of the block (`r_b`, `c_b`) of `H`, with block size being `blocksize`.
The complex-conjugate element is also pushed.
"""
function push_vals!(H; r_b, c_b, r, c, blocksize, val)
    i = (r_b-1)*blocksize + r
    j = (c_b-1)*blocksize + c
    H[i, j] = val
    H[j, i] = val'
end

function make_∂_x(M, Lx)
    ∂_x = zeros(typeof(Lx), M^2, M^2)
    @floop for jx in 1:M
        for jy in 1:M, j′x in 1+isodd(jx):2:M
            val = 1/(j′x+jx)
            j′x != jx && (val += 1/(j′x-jx))
            ∂_x[(j′x-1)M+jy, (jx-1)M+jy] = 2jx/Lx * val
        end
    end
    return ∂_x
end

function make_∂_y(M, Ly)
    ∂_y = zeros(typeof(Ly), M^2, M^2)
    @floop for jx in 1:M
        for jy in 1:M, j′y in 1+isodd(jy):2:M
            val = 1/(j′y+jy)
            j′y != jy && (val += 1/(j′y-jy))
            ∂_y[(jx-1)M+j′y, (jx-1)M+jy] = 2jy/Ly * val
        end
    end
    return ∂_y
end

"""
Construct wavefunction of state number `stateno` on a grid having `nx` points in `x` and `ny` points in `y` direction.
Return (`xs`, `ys`, `ψ`). If `qx` and `qy` are passed, then construct `ψ` at the corresponding quasimomenta.
"""
function make_wavefunction(dh::DenseHamiltonian, stateno::Integer, nx::Integer, ny::Integer, iqx::Integer=0, iqy::Integer=0)
    (;Lx, Ly, xlims, ylims, M, V, V_q) = dh
    xs = range(0, Lx, nx) # these are the differences `x - xlims[1]`, with `x ∈ xlims`
    ys = range(0, Ly, ny)
    ψ = Matrix{eltype(V)}(undef, nx, ny)
    if dh.isperiodic
        B = 2M + 1
        if iqx != 0 # if quasimomentum index has been passed
            @floop for (iy, y) in enumerate(ys)
                for (ix, x) in enumerate(xs)
                    ψ[ix, iy] = sum(V_q[(j-1)B+i, stateno, iqx, iqy]cis(2π*jx*x/Lx + 2π*jy*y/Ly) for (j, jx) in enumerate(-M:M)
                                                                                                 for (i, jy) in enumerate(-M:M)) / √(Lx*Ly)
                end
            end
        else # no quasimomentum index
            @floop for (iy, y) in enumerate(ys)
                for (ix, x) in enumerate(xs)
                    ψ[ix, iy] = sum(V[(j-1)B+i, stateno]cis(2π*jx*x/Lx + 2π*jy*y/Ly) for (j, jx) in enumerate(-M:M)
                                                                                     for (i, jy) in enumerate(-M:M)) / √(Lx*Ly)
                end
            end
        end
    else # nonperiodic
        @floop for (iy, y) in enumerate(ys)
            for (ix, x) in enumerate(xs)
                ψ[ix, iy] = sum(V[(jx-1)M+jy, stateno]sin(π*jx*x/Lx)sin(π*jy*y/Ly) for jx in 1:M for jy in 1:M) * 2 / √(Lx*Ly)
            end
        end
    end
    return xs .+ xlims[1], ys .+ ylims[1], ψ # return "normal" coordinates, in `x ∈ xlims` and `y ∈ ylims`
end

"""
Calculate `nev` lowest eigenvectors and eigenvalues using `ArnoldiMethod`.
Pass `nev=0` for full diagonalisation using `LinearAlgebra`.
"""
function diagonalize!(dh::DenseHamiltonian; nev::Integer)
    if nev == 0
        dh.ε, dh.V = eigen(Hermitian(dh.H))
    else
        S, info = partialschur(dense_linear_map(Hermitian(dh.H)); nev, which=:LM, tol=1e-7); # `which=:SR` does not converge, so we use "shift-invert" (although shift is zero)
        @show info
        dh.V = S.Q
        dh.ε = inv.(real.(S.eigenvalues)) # invert back
    end
end

"Helper function for diagonalisation. It encodes the in-place multiplication by the inverse (required for shift-invert)."
function dense_linear_map(A)
    F = factorize(A)
    LinearMap{eltype(A)}((y, x) -> ldiv!(y, F, x), size(A, 1), ismutating=true)
end

"""
Calculate eigenenergies for all pairs of quasimomenta in `qxs` and `qys`.
Calculate `nev` lowest bands using `ArnoldiMethod`.
If `nev=0` or not passed, then full diagonalisation using `LinearAlgebra` is performed.
Note that `dh.H` is modified in the process.
"""
function diagonalize!(dh::DenseHamiltonian{R,T}, qxs::AbstractVector{<:Real}, qys::AbstractVector{<:Real}; nev::Integer=0) where {R<:Real, T<:Number}
    (;M, xlims, ylims, Lx, Ly, δ, 𝑈, 𝐴_x, 𝐴_y) = dh

    nsaves = nev == 0 ? (2M+1)^2 : nev # number of eigenvalues and eigenvectors to allocate
    dh.ε_q = Array{R,3}(undef, nsaves, length(qxs), length(qys))
    dh.V_q = Array{T,4}(undef, (2M+1)^2, nsaves, length(qxs), length(qys))
    
    if 𝐴_x === nothing
        H_diag = diagview(dh.H)
        U_diag = H_diag[(end+1) ÷ 2] # generally, `H = -Δ + U`, but this element is purely `U`, since Laplace is zero (see construction of Δ in `DenseHamiltonian` constructor)
    else
        N = 4M # number of points for FFT. This will yield harmonics from -2M to 2M
        dx, dy = Lx/N, Ly/N
        xs = range(xlims[1], xlims[2]-dx, N)
        ys = range(ylims[1], ylims[2]-dy, N)

        f = dx/Lx * dy/Ly
        u = [𝑈(x, y) for x in xs, y in ys]

        F = FFTW.plan_rfft(u)
        U = F * u * f |> dft_to_matrix
    
        a_x = [𝐴_x(x, y) for x in xs, y in ys]
        a_y = [𝐴_y(x, y) for x in xs, y in ys]

        D_x = F * a_x * -f |> dft_to_matrix # this is -𝐴ₓ
        D_y = F * a_y * -f |> dft_to_matrix # this is -𝐴y
        
        D_x += Diagonal(typeof(Lx)[2π * δ * jx/Lx for jx in -M:M for jy in -M:M]) # this adds -iδ∂ₓ and results in -iδ∂ₓ-𝐴ₓ
        D_y += Diagonal(typeof(Lx)[2π * δ * jy/Ly for jx in -M:M for jy in -M:M]) # this adds -iδ∂y and results in -iδ∂y-𝐴y
    end
    
    # iterate quasimomenta
    for (iqy, qy) in enumerate(qys), (iqx, qx) in enumerate(qxs)
        # update diagonal
        if 𝐴_x === nothing
            H_diag = [(2π*δ*jx/Lx + qx)^2 + (2π*δ*jy/Ly + qy)^2 + U_diag for jx in -M:M for jy in -M:M]
        else
            dh.H = (D_x + LA.I*qx)^2 + (D_y + LA.I*qy)^2 + U
        end

        # diagonalise
        if nev == 0
            dh.ε_q[:, iqx, iqy], dh.V_q[:, :, iqx, iqy] = eigen(Hermitian(dh.H))
        else
            S, info = partialschur(dense_linear_map(Hermitian(dh.H)); nev, which=:LM, tol=1e-7); # `which=:SR` does not converge, so we use "shift-invert" (although shift is zero)
            @show info
            dh.V_q[:, :, iqx, iqy] = S.Q
            dh.ε_q[:, iqx, iqy] = inv.(real.(S.eigenvalues)) # invert back
        end
    end
end