"""
A type representing a spatial [𝑟 = (𝑥, 𝑦)], 𝑛-component, possibly quasimomentum-dependent Hamiltonian (𝐻ᵢⱼ)
    𝐻ᵢᵢ(𝑟) = [-i𝛿∇ + 𝑞 - 𝐴(𝑟)]² + 𝑈ᵢᵢ(𝑟)
    𝐻ᵢⱼ(𝑟) = 𝑈ᵢⱼ(𝑟)
as a sparse matrix.
"""
mutable struct SparseHamiltonian2D{R<:Real,T<:Number,S<:Number} <: XSpaceHamiltonian2D{:sparse} # in practice `T` shoudld be `R` or `Complex{R}` (and same for `S`) -- always check this. If this is not the case, probably your 𝑈 or 𝐴 do not return R's.
    xlims::Tuple{R, R}
    ylims::Tuple{R, R}
    Lx::R # length along 𝑥
    Ly::R # length along 𝑦
    M::Int # maximum harmonic number (will use -M:M for periodic, 1:M for nonperiodic)
    δ::R # coefficient of the momentum term: -iδ∇
    nc::Int # number of components
    isperiodic::Bool
    ishermitian::Bool # `H` is nonhermitian if decays Γ are present
    𝑈::Matrix{<:Union{Function,Nothing}} # nc-component Hamiltonian matrix containing coordinate-space functions
    𝐴_x::Union{Function,Nothing}
    𝐴_y::Union{Function,Nothing}
    H::SparseMatrixCSC{T, Int64} # momentum-space Hamiltonian used for diagonalisation (UMFPACKFactorization only supports Int64-type indices)
    ε::Vector{S} # eigenvalues, can be complex for nonhermitian `H`, hence additional type `S`
    V::Matrix{T} # eigenvectors matrix
    ε_q::Array{S,3} # ε_q[n, iqx, iqy] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    V_q::Array{T,4} # V_q[:, n, iqx, iqy] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)
end

"""
Construct a `SparseHamiltonian2D` object using the coordinate-space functions stored in `𝑈`, decay rates `Γ`, and gauge field (same for all components) 𝐴_x, 𝐴_y.
`M` is the maximum harmonic number. In the periodic case, the Hamiltonian will be `nc*(2M+1)²`-by-`nc*(2M+1)²` where `nc` is the number of components.
In nonperiodic case, the size will be `nc*M²`-by-`nc*M²`.
`𝑈_iseven[i, j]` matters only if `isperiodic=true` and shows whether `𝑈[i, j]` is an even function (i.e. whether 𝑢(𝑥, 𝑦) = 𝑢(-𝑥, -𝑦)). If it is, then Fourier transform is real, which is used for better accuracy.
If *all* functions are even (and real), then the resulting Fourier-space Hamiltonian is real (provided also there is no 𝐴 and Γ), giving a speed-up and better accuracy (compared to complex diagonalisation).
If `𝑈[i, j] === nothing` or it is complex, then the value of `𝑈_iseven[i, j]` does not matter.
"""
function SparseHamiltonian2D(𝑈::AbstractMatrix{<:Union{Function,Nothing}}, xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                             𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1)),
                             𝐴_x::Union{Function,Nothing}=nothing, 𝐴_y::Union{Function,Nothing}=nothing, fft_threshold::R=√eps(R)) where R <: Real
    Lx, Ly = xlims[2]-xlims[1], ylims[2]-ylims[1]
    
    PI = R(π) # π of the working type to prevent widening

    nc = size(𝑈, 1) # number of components

    # `isreal` will show if the resulting `H` will be real
    isreal = all( 𝑢(xlims[1], ylims[1]) isa Real for 𝑢 in 𝑈 if !isnothing(𝑢)) & # check if all functions in 𝑈 are real
             isnothing(𝐴_x) & isnothing(𝐴_y) & all(==(0), Γ)
    if isperiodic # for periodic potential, also check if functions are even 
        isreal &= all(𝑈_iseven[𝑈 .!== nothing])
    end

    # allocate a matrix of dimensions like `𝑈`. `H_temp[i, j]` will hold the Fourier-transformed sparse matrix corresponding to 𝑈ᵢⱼ
    T = isreal ? R : Complex{R}
    H_temp = [SparseMatrixCSC{T, Int64}(undef, 0, 0) for _ in 1:nc, _ in 1:nc]

    if isperiodic
        N = 4M + 1 # number of points for FFT. This will yield harmonics from -2M to 2M
        dx, dy = Lx/N, Ly/N
        xs = range(xlims[1], xlims[2]-dx, N)
        ys = range(ylims[1], ylims[2]-dy, N)

        fft_buff = Matrix{Complex{R}}(undef, N, N) # a buffer for all (in-place) FFTs
        F = FFTW.plan_fft!(fft_buff) # the savings of rfft are negligible, and the output is much less convenient to handle in `fft_to_matrix`, so using fft. Also, this way we can do FFT in-place

        # iterate over `𝑈` and populate `H_temp`
        for jH in axes(𝑈, 2)
            for iH in 1:jH # only upper triangle is scanned. The lower triangle is filled only if Γ is present
                𝑢 = 𝑈[iH, jH]

                # calculate and store FFT of 𝑢
                if isnothing(𝑢)
                    H_temp[iH, jH] = spzeros(T, Int64, (2M+1)^2, (2M+1)^2)
                else
                    𝑢_isrealeven = (𝑢(xlims[1], ylims[1]) isa Real) & 𝑈_iseven[iH, jH]
                    fft_buff .= 𝑢.(xs, ys')
                    F * fft_buff # in-place FFT, weird syntax
                    fft_buff ./= N^2
                    H_temp[iH, jH] = fft_to_matrix_sparse!(fft_buff; make_real=𝑢_isrealeven, fft_threshold)
                end

                # for a diagonal block, add Laplacian, Γ, and 𝐴
                if iH == jH
                    if Γ[iH] != 0
                        H_temp[iH, jH] -= im*Γ[iH]/2 * LA.I # H_temp[iH, jH][3] are the values of 𝑈ᵢⱼ, the last elements are the diagonal elements
                    end
                    # if there is no 𝐴, then add Laplacian. Otherwise it will be added together with 𝐴 components
                    if isnothing(𝐴_x) && isnothing(𝐴_y)
                        H_temp[iH, jH] += Diagonal([(2PI*δ)^2 * ((jx/Lx)^2 + (jy/Ly)^2) for jx in -M:M for jy in -M:M]) # this is -δ²Δ
                    else
                        # if 𝐴 is present, we have to construct the matrix explicitly
                        if 𝐴_x !== nothing
                            fft_buff .= 𝐴_x.(xs, ys')
                            F * fft_buff
                            fft_buff ./= N^2
                            A_i = fft_to_matrix_sparse!(fft_buff; fft_threshold)
                            ∂_i = Diagonal([2PI * δ * jx/Lx for jx in -M:M for jy in -M:M]) # this is -iδ∂ₓ
                            H_temp[iH, jH] += (∂_i - A_i)^2
                            # if there is no 𝐴𝑦, then add ∂ₓ². Otherwise it will be added together with 𝐴𝑦 in the next `if` clause
                            isnothing(𝐴_y) && (H_temp[iH, jH] += Diagonal([(2PI * δ * jy/Ly)^2 for jx in -M:M for jy in -M:M]))
                        end
                        if 𝐴_y !== nothing
                            fft_buff .= 𝐴_y.(xs, ys')
                            F * fft_buff
                            fft_buff ./= N^2
                            A_i = fft_to_matrix_sparse!(fft_buff; fft_threshold)
                            ∂_i = Diagonal([2PI * δ * jy/Ly for jx in -M:M for jy in -M:M]) # this is -iδ∂y
                            H_temp[iH, jH] += (∂_i - A_i)^2
                            # if there is no 𝐴ₓ, then add ∂𝑦². Otherwise it was added together with 𝐴ₓ in the preceding `if` clause
                            isnothing(𝐴_x) && (H_temp[iH, jH] += Diagonal([(2PI * δ * jx/Lx)^2 for jx in -M:M for jy in -M:M]))
                        end
                    end
                else # non-diagonal block
                    H_temp[jH, iH] = H_temp[iH, jH]' # fill the conjugate block of 𝑈
                    # Could potentially be avoided if all(iszero, Γ) because then we could use a Hermitian view.
                    # But factorisation is LU anyway, so a non-hermitian workspace is needed. 
                end
            end
        end
    else # non-periodic
        # not implemented
    end
    
    H = hvcat(nc, transpose(H_temp)...) # construct the final Hamiltonian

    # determine the type of eigenvalues 
    ishermitian = all(==(0), Γ) # if all `Γ`s are zeros, then Hamiltonian is Hermitian and the eigenvalues real
    S = ishermitian ? R : Complex{R} # type of eigenvalues
    return SparseHamiltonian2D(xlims, ylims, Lx, Ly, M, δ, nc, isperiodic, ishermitian, 𝑈, 𝐴_x, 𝐴_y, H, S[], T[;;], S[;;;], T[;;;;])
end

"""
Set to zero values of `u` that are smaller by magnitude than `threshold`.
Based on the resulting number of nonzero elements in `u`, count the number of values that will be stored in 𝑈.
"""
function filter_count_fft!(u::AbstractMatrix{<:Number}; fft_threshold::Real=0)
    n_elem = 0
    N = size(u, 2) # number of points used for FFT
    M = (N-1) ÷ 4 # maximum harmonic number (recall that N = 4M + 1 in the constructor)
    B = 2M + 1 # the size of each block

    # roughly, r controls the block-diagonal on which u[r, c] will be placed, while c controls the diagonal inside all those blocks
    for c in axes(u, 2), r in axes(u, 1)
        if abs(u[r, c]) ≤ fft_threshold
            u[r, c] = 0
        else
            # the block-diagonal into which u[r, c] will be placed: 0 is main block-diagonal, 1 is the first lower or upper block-diagonal, etc.
            b_d = r ≤ B ? r - 1 : B - (r-B)
            # the diagonal (of a given block) into which u[r, c] will be placed: 0 is main diagonal, 1 is the first lower or upper diagonal, etc.
            d = c ≤ B ? c - 1 : B - (c-B)

            n_elem += (B - b_d) * (B - d)
        end
    end
    return n_elem
end

"""
Based on results of 2D `fft` output `u`, return `rows, cols, vals` tuple for constructing a sparse matrix.
`make_real=true` will take real parts of elements of `u`.
"""
function fft_to_matrix_sparse!(u::Matrix{<:Number}; fft_threshold::Real=0, make_real=false)
    make_real && (u .= real.(u))
    n_elem = filter_count_fft!(u; fft_threshold)

    N = size(u, 2) # number of points used for FFT
    M = (N-1) ÷ 4 # maximum harmonic number (recall that N = 4M + 1 in the constructor)
    B = 2M + 1 # the size of each block

    rows = Vector{Int64}(undef, n_elem)
    cols = Vector{Int64}(undef, n_elem)
    vals = Vector{make_real ? real(eltype(u)) : eltype(u)}(undef, n_elem)

    counter = 1

    for c_u in axes(u, 2), r_u in axes(u, 1) # iterate over columns and rows of `u`
        u[r_u, c_u] == 0 && continue
        val = u[r_u, c_u]
        if r_u ≤ B # when using rows 1 through B of `u` to fill the lower block-triangle of H, including the main block-diagonal
            d = 1 - r_u # (negative) block-diagonal number, where 0 is the main block-diagonal, -1 is first lower block-diagonal, etc.
            r_b_range = r_u:B  # a value from `r_u`th row of `u` will be put in block-rows of `H` from `r_u`th to `B`th
        else # when using rows B+1 through end of `u` to fill the upper block-triangle of H
            d = B - (r_u-B) # (positive) block-diagonal number, where 0 is the main block-diagonal, +1 is first upper block-diagonal, etc.
            r_b_range = 1:r_u-B # a value from `r_u`th row of `u` will be put in block-rows of `H` from `r_u`th to `B`th
        end
        for r_b in r_b_range # block-rows where to place the value
            c_b = r_b + d # block-column where to place the value
            # fill the lower triangle of the block, including the main diagonal
            if c_u ≤ B # for `c_u` ≤ `B`, the value from `c_u`th column of `u` will be put to the `c_u`th lower diagonal of the block (`c_u=1` means main diagonal)
                for (r, c) in zip(c_u:B, 1:B+1-c_u)
                    counter = push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize=B, val)
                end
            # fill the upper triangle of the block
            else # for `c_u` > `B`, the value from `c_u`th column of `u` will be put to the `2B-c_u`th upper diagonal of the block
                c_u_inv = 2B-c_u+1 
                for (r, c) in zip(1:B+1-c_u_inv, c_u_inv:B)
                    counter = push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize=B, val)
                end
            end
        end
    end

    return sparse(rows, cols, vals)
end

"""
Push value `val` to element (`r`, `c`) of the block (`r_b`, `c_b`) of a sparse matrix encoded in `rows`, `cols`, `vals`; block size being `blocksize`.
`counter` shows where to push and the updated value is returned.
If `conjugate=true`, then the complex-conjugate element is also pushed.
"""
function push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize, val, conjugate=false)
    i = (r_b-1)*blocksize + r
    j = (c_b-1)*blocksize + c
    rows[counter] = i
    cols[counter] = j
    vals[counter] = val
    counter += 1
    if conjugate
        rows[counter+1] = j
        cols[counter+1] = i
        vals[counter+1] = val'
        counter += 1
    end
    return counter
end

"Calculate `nev` lowest eigenvectors and eigenvalues."
function diagonalize!(sh::SparseHamiltonian2D{R,T,S}; nev::Integer, verbose::Bool=false) where {R<:Real,T<:Number,S<:Number}
    prob = LS.LinearProblem(sh.H, similar(sh.H, size(sh.H, 1)))
    linsolve = LS.init(prob, LS.UMFPACKFactorization())
    linmap = LinSolveLinMap{T, typeof(linsolve)}(linsolve, size(sh.H))
    ps, info = partialschur(linmap; nev, which=:LM);
    verbose && @show info
    ε, sh.V = partialeigen(ps)
    if sh.ishermitian # if sh.H is Hermitian but complex, the solver returns complex eigenvalues
        sh.ε = real(inv.(ε)) # so we make them real manually (no copy is made if already real)
    else
        sh.ε = inv.(ε)
    end
end

"A linear map holding a `LinearSolve` object, used for applying the inverse map."
struct LinSolveLinMap{T,L} <: LinearMaps.LinearMap{T}
    linsolve::L
    size::Dims{2}
end

Base.size(lm::LinSolveLinMap) = lm.size

function LinearMaps._unsafe_mul!(y, lm::LinSolveLinMap, x::AbstractVector)
    copy!(lm.linsolve.b, x)
    copy!(y, LS.solve!(lm.linsolve).u) # `solve!` allocates up to 50 KiB :(
end

##### Unused but correct and tested functions

"""
Set to zero values of `u` that are smaller by magnitude than `threshold`.
Based on the resulting number of nonzero elements in `u`, count the number of values that will be stored in 𝑈.
"""
function _filter_count_rfft!(u::AbstractMatrix{<:Number}; fft_threshold::Real=0)
    n_elem = 0
    M = size(u, 2) ÷ 2
    N = size(u, 1) # if `u` is really the result of `rfft`, then `N == M+1`, but we keep the calculation a bit more general
    # do the first row of `u`, i.e. the diagonal blocks of 𝑈, separately
    for c in axes(u, 2)
        r = 1
        if abs(u[r, c]) ≤ fft_threshold
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
        if abs(u[r, c]) ≤ fft_threshold
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
Based on results of 2D `rfft` output `u`, return `rows, cols, vals` tuple for constructing a sparse matrix.
Optionally: `d` is a tuple of shifts in 𝑥 and 𝑦 directions, divided by the corresponding periods.
`make_real=true` will take real parts of elements of `u`.
"""
function _rft_to_matrix_sparse!(u::Matrix{<:Number}; fft_threshold::Real=0, make_real=false, d::Tuple{<:Real,<:Real}=(0, 0))
    n_elem = filter_count_fft!(u; fft_threshold)

    rows = Vector{Int64}(undef, n_elem)
    cols = Vector{Int64}(undef, n_elem)
    vals = Vector{make_real ? real(eltype(u)) : eltype(u)}(undef, n_elem)

    u₀₀ = u[1, 1] # save the secular component
    u[1, 1] = 0 # remove because it breaks the structure of the loop below if included
    make_real && (u .= real.(u))

    N = size(u, 2) # number of points used for FFT
    M = (N-1) ÷ 4 # maximum harmonic number (recall that N = 4M + 1 in the constructor)
    B = 2M + 1 # the size of each block

    counter = 1

    # it is assumed that u[1, 1] == 0 -- otherwise, one would also need to prevent double pushing of the diagonal elements
    for c_u in axes(u, 2), r_u in axes(u, 1) # iterate over columns and rows of `u`
        u[r_u, c_u] == 0 && continue
        e = c_u <= M+1 ? cispi(2*(c_u-1)*d[1]) : cispi(2*(c_u-size(u, 2))*d[1]) # the factor is exp(2πi/L n) but division by `L` is absorbed in `d`
        val = u[r_u, c_u] * e * cispi(2*(r_u-1)*d[2])
        for r_b in r_u:B # a value from `r_u`th row of `u` will be put in block-rows of `H` from `r_u`th to `B`th.
            c_b = r_b - r_u + 1 # block-column where to place the value
            if c_u ≤ B # for `c_u` ≤ `B`, the value from `c_u`th column of `u` will be put to the `c_u`th lower diagonal of the block (`c_u=1` means main diagonal)
                for (r, c) in zip(c_u:B, 1:B+1-c_u)
                    push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize=B, val)
                    counter += 2
                end
            elseif r_b != c_b # for `c_u` > `B`, the value from `c_u`th column of `u` will be put to the `2B-c_u`th upper diagonal of the block,
                c_u_inv = 2B-c_u+1 # but this is not needed for a diagonal block (`r_b == c_b`), because then the upper triangle of the block has already been filled by pushing the conjugate element
                for (r, c) in zip(1:B+1-c_u_inv, c_u_inv:B)
                    push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize=B, val)
                    counter += 2
                end
            end
        end
    end
    n_diag = B^2 # number of diagonal elements in 𝑈
    # fill positions of the diagonal elements
    rows[end-n_diag+1:end] .= 1:n_diag
    cols[end-n_diag+1:end] .= 1:n_diag
    vals[end-n_diag+1:end] .= u₀₀

    return sparse(rows, cols, vals)
end