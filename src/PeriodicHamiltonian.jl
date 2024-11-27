mutable struct PeriodicHamiltonian{R<:Real,T<:Number} # in practice `T` will be `R` or `Complex{R}`
    xlims::Tuple{R, R}
    ylims::Tuple{R, R}
    Lx::R # period along 𝑥
    Ly::R # period along 𝑦
    H::SparseMatrixCSC{T, Int}
    ε::Vector{R} # eigenvalues
    V::Matrix{T} # eigenvectors matrix
end

"""
Construct a `PeriodicHamiltonian` object.
Coordinate functions will be FFT'ed using harmonics from `-M` to `M`, yielding `M=2N` points.
The size of the Hamiltonian will be `(M+1)`² × `(M+1)`².
To make sure that the resulting Hamiltonian matrix is of the desired type `T`, the type of elements of `xlims`, `ylims`,
and the return type of the passed functions has to be the same. E.g., if all are `Float32`, then `T` will be `Float32` if only `𝑈` is passed,
and `ComplexF32` if `𝐴`'s are passed. Inconsistency in the types of arguments will result in widening.
"""
function PeriodicHamiltonian(xlims::Tuple{<:Real,<:Real}, ylims::Tuple{<:Real,<:Real}; 𝑈::Union{Function,Nothing}=nothing, 𝐴_x::Union{Function,Nothing}=nothing,
                              𝐴_y::Union{Function,Nothing}=nothing, M=64)
    Lx, Ly = xlims[2]-xlims[1], ylims[2]-ylims[1] # area dimensions
    if isodd(M)
        @warn "`M` must be even. Reducing `M` by one."
        M -= 1
    end
    N = 2M
    dx, dy = Lx/N, Ly/N
    f = dx/Lx * dy/Ly
    
    xs = range(0, Lx-dx, N)
    ys = range(0, Ly-dy, N)
    
    u = [𝑈(x, y) for x in xs, y in ys]
    F = FFTW.plan_rfft(u)
    U = F * u * f |> fft_to_matrix!

    m = M÷2
    Δ = Diagonal([-(2π)^2 * ((jx/Lx)^2 + (jy/Ly)^2) for jx in -m:m for jy in -m:m])
    
    if 𝐴_x === nothing
        H = -Δ + U
    else
        a_x = [𝐴_x(x, y) for x in xs, y in ys]
        a_y = [𝐴_y(x, y) for x in xs, y in ys]

        A_x = F * a_x * f |> fft_to_matrix!
        A_y = F * a_y * f |> fft_to_matrix!
        
        ∂_x = Diagonal([2π*im * jx/Lx for jx in -m:m for jy in -m:m])
        ∂_y = Diagonal([2π*im * jy/Ly for jx in -m:m for jy in -m:m])
        
        H = -Δ + im*(A_x*∂_x + A_y*∂_y + ∂_x*A_x + ∂_y*A_y) + A_x^2 + A_y^2 + U
    end
    return PeriodicHamiltonian(xlims, ylims, Lx, Ly, H, typeof(Lx)[], eltype(H)[;;])
end

"""
Set to zero values of `u` that are `threshold` times smaller (by absolute magnitude) than the largest.
Based on the resulting number of nonzero elements in `u`, count the number of values that will be stored in 𝐻.
"""
function filter_count!(u::AbstractMatrix{<:Number}; fft_threshold::Real=1e-5)
    n_elem = 0
    M = size(u, 2) ÷ 2
    N = size(u, 1) # if `u` is really the result of `rfft`, then `N == M+1`, but we keep the calculation a bit more general
    # do the first row of `u`, i.e. the diagonal blocks of 𝐻, separately
    for c in axes(u, 2)
        r = 1
        if abs(u[r, c]) < fft_threshold
            u[r, c] = 0
        else
            if c < M+1
                n_elem += (N - (r-1)) * (M+1 - (c-1)) # number of blocks in which `u[r, c]` will appear × number of times it will appear within each block
            elseif c == M+1
                n_elem += 2(N - (r-1)) * (M+1 - (c-1))
            else
                n_elem += (N - (r-1)) * (c - M)
            end
        end
    end
    for c in axes(u, 2), r in 2:size(u, 1)
        if abs(u[r, c]) < fft_threshold
            u[r, c] = 0
        else
            if c < M+1
                n_elem += 2(N - (r-1)) * (M+1 - (c-1))
            elseif c == M+1
                n_elem += 4(N - (r-1)) * (M+1 - (c-1))
            else
                n_elem += 2(N - (r-1)) * (c - M)
            end
        end
    end
    return n_elem
end

"""
Based on results of a real 2D fft `u`, return `rows, cols, vals` tuple for constructing a sparse matrix.
Optionally, a tuple `δ` of shifts in 𝑥 and 𝑦 directions can be supplied.
"""
function fft_to_matrix!(u, δ::Tuple{<:Real,<:Real}=(0, 0))
    n_elem = filter_count!(u) # filter small values and calculate the number of elements in the final Hamiltonian

    rows = Vector{Int32}(undef, n_elem)
    cols = Vector{Int32}(undef, n_elem)
    vals = Vector{eltype(u)}(undef, n_elem)

    u₀₀ = u[1, 1] # save the secular component
    u[1, 1] = 0 # remove because it breaks the structure of the loop below if included

    L = π # periodicity of the potential, TO BE REVISED
    M = size(u, 2) ÷ 2 # M + 1 gives the size of each block; `size(u, 1)` gives the number of block-rows (= number of block-cols)
    counter = 1

    # it is assumed that u[1, 1] == 0 -- otherwise, one would also need to prevent double pushing of the diagonal elements
    for c_u in axes(u, 2), r_u in axes(u, 1) # iterate over columns and rows of `u`
        u[r_u, c_u] == 0 && continue
        e = c_u <= M+1 ? cispi(2/L*(c_u-1)*δ[1]) : cispi(2/L*(c_u-(2M+1))*δ[1])
        val = u[r_u, c_u] * e * cispi(2/L*(r_u-1)*δ[2])
        for r_b in r_u:size(u, 1) # a value from `r_u`th row of `u` will be put in block-rows of `H` from `r_u`th to `M+1`th. For actual applications, `size(u, 1) == M+1`
            c_b = r_b - r_u + 1 # block-column where to place the value
            if c_u <= M # for `c_u` ≤ `M`, the value from `c_u`th column of `u` will be put to the `c_u`th lower diagonal of the block
                for (r, c) in zip(c_u:M+1, 1:M+2-c_u)
                    push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize=M+1, val)
                    counter += 2
                end
            elseif c_u == M+1 # for `c_u` = `M+1`, the value from `c_u`th column of `u` will be put to lower left and upper right corners of the block
                push_vals!(rows, cols, vals, counter; r_b, c_b, r=M+1, c=1, blocksize=M+1, val)
                counter += 2
                if r_b != c_b
                    push_vals!(rows, cols, vals, counter; r_b, c_b, r=1, c=M+1, blocksize=M+1, val) # if we're in the diagonal block, then the upper right corner is conjugate to lower left and has already been pushed
                    counter += 2
                end
            else # for `c_u` ≥ `M+2`, the value from `c_u`th column of `u` will be put to the `2M+2-c_u`th upper diagonal of the block
                if r_b != c_b # if `r_b == c_b`, then upper diagonal of the block has already been filled by pushing the conjugate element
                    c_u_inv = 2M+2 - c_u
                    for (r, c) in zip(1:M+2-c_u_inv, c_u_inv:M+1)
                        push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize=M+1, val)
                        counter += 2
                    end
                end
            end
        end
    end
    n_diag = (M+1)^2 # number of diagonal elements in 𝐻
    # fill positions of the diagonal elements
    rows[end-n_diag+1:end] .= 1:n_diag
    cols[end-n_diag+1:end] .= 1:n_diag
    vals[end-n_diag+1:end] .= u₀₀

    return sparse(rows, cols, vals)
end


"""
Push value `val` stored at (`r`, `c`) in some matrix to the block (`r_b`, `c_b`) of a sparse matrix encoded in `rows`, `cols`, `vals`.
`counter` shows where to push. The complex-conjugate element is also pushed.
"""
function push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize, val)
    i = (r_b-1)*blocksize + r
    j = (c_b-1)*blocksize + c
    rows[counter] = i
    cols[counter] = j
    vals[counter] = val
    # conjugate
    rows[counter+1] = j
    cols[counter+1] = i
    vals[counter+1] = val'
end

function diagonalize!(dh::PeriodicHamiltonian; nev::Integer)
    S, info = partialschur(make_linear_map(Hermitian(dh.H)); nev, which=:LM);
    @show info
    dh.V = S.Q
    dh.ε = inv.(real.(S.eigenvalues))
end

function make_linear_map(A)
    LDL = ldl_analyze(A)
    ldl_factorize!(A, LDL) # mutates (updates) `LDL`, does not alter `A`
    LinearMap{eltype(A)}((y, x) -> ldiv!(y, LDL, x), size(A,1), ismutating=true)
end

"""
Construct wavefunction of state number `stateno` on a grid having `nx` points in `x` and `ny` points in `y` direction.
Return (`xs`, `ys`, `ψ`).
"""
function make_wavefunction(dh::PeriodicHamiltonian, stateno::Integer, nx::Integer, ny::Integer)
    (;Lx, Ly, xlims, ylims, V) = dh
    B = Int(√size(V, 1))
    j_max = (B - 1) ÷ 2
    xs = range(xlims[1], xlims[2], nx)
    ys = range(ylims[1], ylims[2], ny)
    ψ = Matrix{Complex{typeof(Lx)}}(undef, nx, ny)
    for (iy, y) in enumerate(ys), (ix, x) in enumerate(xs)
        ψ[ix, iy] = sum(V[(j-1)B+i, stateno]cis(2π*jx*x/Lx + 2π*jy*y/Ly) for (j, jx) in enumerate(-j_max:j_max)
                                                                         for (i, jy) in enumerate(-j_max:j_max)) / √(Lx*Ly)
    end
    return xs, ys, ψ
end