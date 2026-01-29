mutable struct FourierTransformer{R,T,Plan,D} # T is the type of buffer, real for sin and cos, complex for cis
    xs::Matrix{R} # coordinates matrix: 1st column contains 𝑥's, second contains 𝑦's, etc.
    M::Int # maximum harmonic number (will use -M:M for cis basis, 1:M for sin, 0:M for cos)
    normalisation::R # factor by which the transform should be *multiplied*
    basis::Symbol
    buff::Array{T,D} # buffer for the result of the transform
    buff_im::Array{T,D} # additional buffer for storing the DST/DCT of the imaginary part of a function
    did_complex_rxdft::Bool # a flag that is true if the last-performed transformation was a DST/DCT ("FFTW.RODFT"/"FFTW.REDFT") of a *complex* function. The only reason why the type is mutable.
    plan::Plan
end

"""
`target_real` is only used in the sin and cos case: set to false if you plan to calculate DST or DCT for complex functions.
Set `target_rank=1` if you are making the transform for building a Fourier-space vector (using `fft_to_vector`),
set `target_rank=2` if you are making the transform for building a Fourier-space matrix (using `fft_to_matrix`). The latter needs twice the number of harmonics.
"""
function FourierTransformer(xlims::AbstractVector{Tuple{R, R}}, M::Integer; basis::Symbol, target_real::Bool=true, target_rank::Integer=2, forward::Bool=true) where R <: AbstractFloat
    D = length(xlims) # number of spatial dimensions
    L = Vector{R}(undef, D) # periods in each dimension. Only needed here, to calculate the normalisation factor
    dx = Vector{R}(undef, D) # dx's in each dimension
    if basis == :cis
        N = target_rank*2M + 1 # number of points for FFT (the same for each dimension). This will yield harmonics from `-target_rank*2M` to `target_rank*2M`
        xs = Matrix{R}(undef, N, D)
        for i in 1:D
            L[i] = xlims[i][2] - xlims[i][1]
            dx[i] = L[i] / N
            xs[:, i] .= range(xlims[i][1], xlims[i][2]-dx[i], N)
        end
        buff = Array{Complex{R}}(undef, ntuple(Returns(N), D)) # a buffer for all (in-place) FFTs
        buff_im = similar(buff, ntuple(Returns(0), D)) # this buffer is not needed in the cis case; make it 0x0 (in `D` dimesions)
        # the time savings of rfft are negligible for us, and the output is much less convenient to handle in `fft_to_matrix`, so using fft. Also, this way we can do FFT in-place
        plan = forward ? FFTW.plan_fft!(buff) : FFTW.plan_bfft!(buff) # using unnormalised `bfft` because the "1/N" used in `ifft` is not right for our use case
    else # sin/cos
        if basis == :cos || target_rank == 2 # for `target_rank == 2` we always need DCT, even if the basis is sin
            N = target_rank*M + 1
            xs = Matrix{R}(undef, N, D)
            for i in 1:D
                L[i] = xlims[i][2] - xlims[i][1]
                dx[i] = L[i] / (N-1)
                xs[:, i] .= range(xlims[i][1], xlims[i][2], N)
            end
            buff = Array{R}(undef, ntuple(Returns(N), D)) # a buffer for all (in-place) FFTs
            buff_im = similar(buff, ntuple(Returns(target_real ? 0 : N), D)) # if `target_real`, then this buffer is not needed; make it 0x0 (in `D` dimesions)
            plan = FFTW.plan_r2r!(buff, FFTW.REDFT00) # note that `REDFT00` is its own inverse
        else # basis == :sin && target_rank == 1 # the only case when we need DST
            N = M
            xs = Matrix{R}(undef, N, D)
            for i in 1:D
                L[i] = xlims[i][2] - xlims[i][1]
                dx[i] = L[i] / (N+1)
                xs[:, i] .= range(xlims[i][1]+dx[i], xlims[i][2]-dx[i], N)
            end
            buff = Array{R}(undef, ntuple(Returns(N), D)) # a buffer for all (in-place) FFTs
            buff_im = similar(buff, ntuple(Returns(target_real ? 0 : N), D)) # if `target_real`, then this buffer is not needed; make it 0x0 (in `D` dimesions)
            plan = FFTW.plan_r2r!(buff, FFTW.RODFT00) # note that `RODFT00` is its own inverse
        end
        target_rank == 1 && (L .*= 2) # just because the normalisation factors feature 1/√(2𝐿) in the case (target_rank == 1 && basis != :cos)
    end

    if forward
        normalisation = target_rank == 1 ? prod(dx ./ sqrt.(L)) : prod(dx./L) # the latter equals (1/N)^D, (1/(N+1)), (1/(N-1)) in the cis, sin, cos cases respectively
    else
        normalisation = target_rank == 1 ? prod(inv.(sqrt.(L))) : prod(inv.(L))
    end
    
    did_complex_rxdft = false # value does not matter before any transform is performed
    return FourierTransformer(xs, M, normalisation, basis, buff, buff_im, did_complex_rxdft, plan)
end

"Perform the transformation of a callable function `𝑓`."
function transform!(ft::FourierTransformer, 𝑓::Function)
    (;xs, normalisation, basis, buff, buff_im, plan) = ft
    D = size(xs, 2)

    if basis == :cis || 𝑓(xs[1, 1:D]...) isa Real
        if D == 1
            buff .= 𝑓.(xs) .* normalisation
        elseif D == 2
            @views buff .= 𝑓.(xs[:, 1], xs[:, 2]') .* normalisation
        end
        plan * buff # in-place transform, weird syntax
        ft.did_complex_rxdft = false
    else # if basis is sin/cos and 𝑓 is complex
        # `FFTW.RxDFT00` can only handle real input. So we transform Re and Im separately.
        if D == 1
            for (ix, x) in enumerate(xs)
                buff[ix], buff_im[ix] = reim(𝑓(x)) .* normalisation
            end
        elseif D == 2
            for iy in axes(xs, 1), ix in axes(xs, 1)
                buff[ix, iy], buff_im[ix, iy] = reim(𝑓(xs[ix, 1], xs[iy, 2])) .* normalisation
            end
        end
        plan * buff
        plan * buff_im
        ft.did_complex_rxdft = true # will be used in fft_to_matrix_*D! to inclue `buff_im` when constructing the matrix
    end
    return
end

"Perform the transformation of a discretised function `f`."
function transform!(ft::FourierTransformer, f::AbstractArray{<:Number})
    (;normalisation, basis, buff, buff_im, plan) = ft
    if basis == :cis || eltype(f) <: Real
        buff .= f .* normalisation
        plan * buff # in-place transform, weird syntax
        ft.did_complex_rxdft = false
    else # if basis is sin/cos and `f` is complex
        # `FFTW.RxDFT00` can only handle real input. So we transform Re and Im separately.
        for i in eachindex(f)
            buff[i], buff_im[i] = reim(f[i]) .* normalisation
        end
        plan * buff
        plan * buff_im
        ft.did_complex_rxdft = true # will be used in fft_to_matrix_*D! to inclue `buff_im` when constructing the matrix
    end
    return
end

################ Dense ################

"""
Use the result of the transform to construct a vector indexed by (𝑗ₓ𝑗y⋯).
If `makesparse=true`, a sparse vector is returned, with values below `threshold` in magnitude filtered out. By default, a dense vector is returned.
If `makereal=true`, a real vector (of type `R`) is returned, which is useful in the cis case if you wish to drop the imaginary part of ft.buff.
"""
function fft_to_vector(ft::FourierTransformer{R,T}; makesparse::Bool=false, makereal=false, threshold::Real=√(eps(R))) where {R <: AbstractFloat, T <: Number}
    (;M, buff, basis) = ft
    D = ndims(buff)
    if basis == :cis
        B = (2M+1)^D
        if makereal
            v_type = R
            buff .= real.(buff)
        else
            v_type = T
        end
    else
        B = basis == :sin ? M^D : (M+1)^D
        v_type = ft.did_complex_rxdft ? Complex{T} : T
    end

    if makesparse
        # if basis == :cis
        #     n_elem = filter_count!(ft; threshold)
        #     rows = Vector{Int64}(undef, n_elem)
        #     cols = Vector{Int64}(undef, n_elem)
        #     vals = Vector{v_type}(undef, n_elem)
        #     fft_to_matrix_sparse!(rows, cols, vals, ft)
        #     v = sparse(rows, cols, vals)
        # else
        #     error("Sparse not available for basis = $basis. Only available for basis = :cis.")
        # end
    else # dense
        v = Vector{v_type}(undef, B)
        fft_to_vector!(v, ft) # we do not pass `makereal` because already performed this above
    end
    return v
end

"""
Use the result of the transform to fill `v` as a vector indexed by (𝑗ₓ𝑗y⋯). 
`makereal=true` is useful in the cis case if you wish to drop the imaginary part of `ft.buff`.
"""
function fft_to_vector!(v::AbstractVector{<:Number}, ft::FourierTransformer; makereal=false)
    (;buff, buff_im, basis) = ft
    D = ndims(buff)
    makereal && (buff .= real.(buff))
    if D == 1
        if basis == :cis
            FFTW.fftshift!(v, buff) # we also do fftshift in `fft_to_matrix`, and hence go over the harmonics in the order -M:M when constructing x-space wf's.
        else # sin/cos
            copy!(v, buff)
            if ft.did_complex_rxdft
                v .+= im.*buff_im
            end
        end
    else
        error("fft_to_vector! not implemented in $(D)D.")
    end
    return
end

"""
Use the result of the transform to construct a matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
If `makesparse=true`, a sparse matrix is returned, with values below `threshold` in magnitude filtered out. By default, a dense matrix is returned.
If `makereal=true`, a real matrix (of type `R`) is returned, which is useful in the cis case if you wish to drop the imaginary part of `ft.buff`.
"""
function fft_to_matrix(ft::FourierTransformer{R,T}; makesparse::Bool=false, makereal=false, threshold::Real=√(eps(R))) where {R <: AbstractFloat, T <: Number}
    (;M, buff, basis) = ft
    D = ndims(buff)
    if basis == :cis
        B = (2M+1)^D
        if makereal
            A_type = R
            buff .= real.(buff)
        else
            A_type = T
        end
    else
        B = basis == :sin ? M^D : (M+1)^D
        A_type = ft.did_complex_rxdft ? Complex{T} : T
    end

    if makesparse
        if basis == :cis
            n_elem = filter_count!(ft; threshold)
            rows = Vector{Int64}(undef, n_elem)
            cols = Vector{Int64}(undef, n_elem)
            vals = Vector{A_type}(undef, n_elem)
            fft_to_matrix_sparse!(rows, cols, vals, ft)
            A = sparse(rows, cols, vals)
        else
            error("Sparse not available for basis = $basis. Only available for basis = :cis.")
        end
    else # dense
        A = Matrix{A_type}(undef, B, B)
        fft_to_matrix!(A, ft) # we do not pass `makereal` because already performed this above
    end
    return A
end

"""
Use the result of the transform to fill `A` as a matrix indexed by (𝑗′ₓ𝑗′y⋯, 𝑗ₓ𝑗y⋯). 
`makereal=true` is useful in the cis case if you wish to drop the imaginary part of `ft.buff`.
"""
function fft_to_matrix!(A::AbstractMatrix{<:Number}, ft::FourierTransformer; makereal=false)
    D = ndims(ft.buff)
    makereal && (ft.buff .= real.(ft.buff))
    if D == 1
        fft_to_matrix_1D!(A, ft)
    elseif D == 2
        fft_to_matrix_2D!(A, ft)
    else
        error("fft_to_matrix_$(D)D! not implemented.")
    end
end

"""
Use the result of the 1D transform to fill `A` as a matrix indexed by (𝑗′ₓ, 𝑗ₓ).
"""
function fft_to_matrix_1D!(A::AbstractMatrix{<:Number}, ft::FourierTransformer)
    (;M, basis, buff, buff_im) = ft
    if basis == :cis
        A[diagind(A)] .= buff[1]
        for i in 2:M+1
            A[diagind(A, 1-i)] .= buff[i]       # fill lower triangle
            A[diagind(A, i-1)] .= buff[end-i+2] # fill upper triangle
        end
    elseif basis == :sin
        if ft.did_complex_rxdft
            for jx in 1:M # not enough work for @floop, slows down execution (checked for M up to 300)
                for j′x in 1:M
                    j₋x = abs(j′x-jx)
                    A[j′x, jx] = ( (buff[j₋x+1] - buff[j′x+jx+1]) + im*(buff_im[j₋x+1] - buff_im[j′x+jx+1]) ) / 2
                end
            end
        else
            for jx in 1:M
                for j′x in 1:M
                    j₋x = abs(j′x-jx)
                    A[j′x, jx] = (buff[j₋x+1] - buff[j′x+jx+1]) / 2
                end
            end
        end
    else # basis == :cos
        if ft.did_complex_rxdft
            for jx in 0:M
                ζₓ = ifelse(iszero(jx), 2, 1)
                for j′x in 0:M
                    ζ′ₓ = ifelse(iszero(j′x), 2, 1)
                    j₋x = abs(j′x-jx)
                    A[j′x+1, jx+1] = ( (buff[j₋x+1] + buff[j′x+jx+1]) + im*(buff_im[j₋x+1] + buff_im[j′x+jx+1]) ) / 2√(ζₓ*ζ′ₓ)
                end
            end
        else
            for jx in 0:M
                ζₓ = ifelse(iszero(jx), 2, 1)
                for j′x in 0:M
                    ζ′ₓ = ifelse(iszero(j′x), 2, 1)
                    j₋x = abs(j′x-jx)
                    A[j′x+1, jx+1] = (buff[j₋x+1] + buff[j′x+jx+1]) / 2√(ζₓ*ζ′ₓ)
                end
            end
        end
    end
    return
end

"""
Use the result of the 2D transform to fill `A` as a matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
"""
function fft_to_matrix_2D!(A::AbstractMatrix{<:Number}, ft::FourierTransformer)
    (;M, basis, buff, buff_im) = ft
    if basis == :cis
        B = 2M + 1
        # TODO add check size(A) .== B^2
        buff_shifted = FFTW.fftshift(buff) # indexing into `u` is more convenient if we shift TODO rewrite without relying on this
        # We sipmly go over each element of `A`, assigning an appropriate element of `u`.
        # In the dense case it is preferred over (since it's faster than) [`fft_to_matrix_sparse!`](@ref)
        # because even if `buffer[i, j]=0`, the corresponding elements of `A` still must be accessed to be set to zero.
        @floop for jx in 1:B
            for jy in 1:B, j′x in 1:B, j′y in 1:B
                j₋x = j′x - jx
                j₋y = j′y - jy
                A[(j′x-1)B+j′y, (jx-1)B+jy] = buff_shifted[j₋x+B, j₋y+B]
            end
        end
    elseif basis == :sin
        # TODO add check size(A) .== M^2
        if ft.did_complex_rxdft
            @floop for jx in 1:M
                for jy in 1:M, j′x in 1:M, j′y in 1:M
                    j₋x = abs(j′x-jx)
                    j₋y = abs(j′y-jy)
                    A[(j′x-1)M+j′y, (jx-1)M+jy] = (buff[j₋x+1, j₋y+1] - buff[j₋x+1, j′y+jy+1] - buff[j′x+jx+1, j₋y+1] + buff[j′x+jx+1, j′y+jy+1]) / 4 +
                                 im * (buff_im[j₋x+1, j₋y+1] - buff_im[j₋x+1, j′y+jy+1] - buff_im[j′x+jx+1, j₋y+1] + buff_im[j′x+jx+1, j′y+jy+1]) / 4
                end
            end
        else
            @floop for jx in 1:M
                for jy in 1:M, j′x in 1:M, j′y in 1:M
                    j₋x = abs(j′x-jx)
                    j₋y = abs(j′y-jy)
                    A[(j′x-1)M+j′y, (jx-1)M+jy] = (buff[j₋x+1, j₋y+1] - buff[j₋x+1, j′y+jy+1] - buff[j′x+jx+1, j₋y+1] + buff[j′x+jx+1, j′y+jy+1]) / 4
                end
            end
        end
    else # basis == :cos
        b = M + 1 # not `B` to preven Core.Box :(
        if ft.did_complex_rxdft
            @floop for jx in 0:M
                for jy in 0:M, j′x in 0:M, j′y in 0:M
                    j₋x = abs(j′x-jx)
                    j₋y = abs(j′y-jy)
                    A[j′x*b+j′y+1, jx*b+jy+1] = (buff[j₋x+1, j₋y+1] + buff[j₋x+1, j′y+jy+1] + buff[j′x+jx+1, j₋y+1] + buff[j′x+jx+1, j′y+jy+1]) / 4 +
                               im * (buff_im[j₋x+1, j₋y+1] + buff_im[j₋x+1, j′y+jy+1] + buff_im[j′x+jx+1, j₋y+1] + buff_im[j′x+jx+1, j′y+jy+1]) / 4
                end
            end
        else
            @floop for jx in 0:M
                for jy in 0:M, j′x in 0:M, j′y in 0:M
                    j₋x = abs(j′x-jx)
                    j₋y = abs(j′y-jy)
                    A[j′x*b+j′y+1, jx*b+jy+1] = (buff[j₋x+1, j₋y+1] + buff[j₋x+1, j′y+jy+1] + buff[j′x+jx+1, j₋y+1] + buff[j′x+jx+1, j′y+jy+1]) / 4
                end
            end
        end
    end
end

################ Sparse ################

function filter_count!(ft::FourierTransformer; threshold::Real=0)
    D = ndims(ft.buff)
    if D == 1
        n_elem = filter_count_1D!(ft; threshold)
    elseif D == 2
        n_elem = filter_count_2D!(ft; threshold)
    else
        error("filter_count_$(D)D! not implemented.")
    end
    return n_elem
end

function fft_to_matrix_sparse!(rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}, vals::AbstractVector{<:Number}, ft::FourierTransformer)
    D = ndims(ft.buff)
    if D == 1
        fft_to_matrix_sparse_1D!(rows, cols, vals, ft)
    elseif D == 2
        fft_to_matrix_sparse_2D!(rows, cols, vals, ft)
    else
        error("fft_to_matrix_sparse_$(D)D! not implemented.")
    end
end

######## 1D ########

"""
Set to zero values of `ft.buff` that are smaller by magnitude than `threshold`.
Based on the resulting number of nonzero elements in `ft.buff`, count and return the number of values that will be stored in the matrix indexed by (𝑗′ₓ, 𝑗ₓ).
"""
function filter_count_1D!(ft::FourierTransformer; threshold::Real=0)
    (;M, buff) = ft
    n_elem = 0
    B = 2M + 1 # Hamiltonian size

    # roughly, c controls the diagonal on which buff[c] will be placed
    for c in eachindex(buff)
        if abs(buff[c]) ≤ threshold
            buff[c] = 0
        else
            # the diagonal into which buff[c] will be placed: 0 is main diagonal, 1 is the first lower or upper diagonal, etc.
            d = c ≤ B ? c - 1 : B - (c-B)

            n_elem += B - d # the number of elements on the `d`th diagonal is `B - d`
        end
    end
    return n_elem
end

"""
Use the result of the transform to construct a sparse matrix indexed by (𝑗′ₓ, 𝑗ₓ).
The type of `vals` might differ from the type of `ft.buff` since one may want to drop the imaginary part.
"""
function fft_to_matrix_sparse_1D!(rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}, vals::AbstractVector{<:Number}, ft::FourierTransformer)
    B = 2ft.M + 1 # the size of Hamiltonian
    counter = 1
    for (c_u, val) in enumerate(ft.buff)
        val == 0 && continue
        # fill the lower triangle
        if c_u ≤ B # for `c_u` ≤ `B`, the value from `c_u`th column of `u` will be put to the `c_u`th lower diagonal (`c_u=1` means main diagonal)
            for (r, c) in zip(c_u:B, 1:B+1-c_u)
                counter = push_vals!(rows, cols, vals, counter, r, c, val)
            end
        # fill the upper triangle
        else # for `c_u` > `B`, the value from `c_u`th column of `u` will be put to the `2B-c_u`th upper diagonal
            c_u_inv = 2B-c_u+1 
            for (r, c) in zip(1:B+1-c_u_inv, c_u_inv:B)
                counter = push_vals!(rows, cols, vals, counter, r, c, val)
            end
        end
    end
end

"""
Push value `val` to element (`r`, `c`) of a sparse matrix encoded in `rows`, `cols`, `vals`.
`counter` shows where to push and the updated value is returned.
"""
function push_vals!(rows, cols, vals, counter, r, c, val)
    rows[counter] = r
    cols[counter] = c
    vals[counter] = val
    counter += 1
    return counter
end

######## 2D ########

"""
Set to zero values of `ft.buff` that are smaller by magnitude than `threshold`.
Based on the resulting number of nonzero elements in `ft.buff`, count and return the number of values that will be stored in the matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
"""
function filter_count_2D!(ft::FourierTransformer; threshold::Real=0)
    (;M, buff) = ft
    n_elem = 0
    B = 2M + 1 # the size of each block

    # roughly, r controls the block-diagonal on which buff[r, c] will be placed, while c controls the diagonal inside all those blocks
    for c in axes(buff, 2), r in axes(buff, 1)
        if abs(buff[r, c]) ≤ threshold
            buff[r, c] = 0
        else
            # the block-diagonal into which buff[r, c] will be placed: 0 is main block-diagonal, 1 is the first lower or upper block-diagonal, etc.
            b_d = r ≤ B ? r - 1 : B - (r-B)
            # the diagonal (of a given block) into which buff[r, c] will be placed: 0 is main diagonal, 1 is the first lower or upper diagonal, etc.
            d = c ≤ B ? c - 1 : B - (c-B)

            n_elem += (B - b_d) * (B - d)
        end
    end
    return n_elem
end

"""
Use the result of the transform to construct a sparse matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
The type of `vals` might differ from the type of `ft.buff` since one may want to drop the imaginary part.
"""
function fft_to_matrix_sparse_2D!(rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}, vals::AbstractVector{<:Number}, ft::FourierTransformer)
    B = 2ft.M + 1 # the size of each block
    u = ft.buff

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
                    counter = push_vals!(rows, cols, vals, counter, r_b, c_b, r, c, B, val)
                end
            # fill the upper triangle of the block
            else # for `c_u` > `B`, the value from `c_u`th column of `u` will be put to the `2B-c_u`th upper diagonal of the block
                c_u_inv = 2B-c_u+1 
                for (r, c) in zip(1:B+1-c_u_inv, c_u_inv:B)
                    counter = push_vals!(rows, cols, vals, counter, r_b, c_b, r, c, B, val)
                end
            end
        end
    end
end

"""
Push value `val` to element (`r`, `c`) of the block (`r_b`, `c_b`) of a sparse matrix encoded in `rows`, `cols`, `vals`; block size being `B`.
`counter` shows where to push and the updated value is returned.
"""
function push_vals!(rows, cols, vals, counter, r_b, c_b, r, c, B, val)
    i = (r_b-1)*B + r
    j = (c_b-1)*B + c
    rows[counter] = i
    cols[counter] = j
    vals[counter] = val
    counter += 1
    return counter
end

########## Unused but correct and tested functions

# """
# Based on results of a real 2D RFT `u`, return the matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
# `make_real=true` will mutate `u`, taking the real parts of elements, which is useful if the original function is even and hence the transform is known to be real.
# This version is for dense matrices, but it is slower than [`fft_to_matrix_2D!`](@ref); used only for testing purposes.
# """
# function _rfft_to_matrix!(u::AbstractMatrix{<:Number}; make_real=false)
#     N = size(u, 2) # number of points used for FFT
#     M = (N-1) ÷ 4 # maximum harmonic number (recall that N = 4M + 1 in the constructor)
#     B = 2M + 1 # the size of each block
#     H = make_real ? zeros(real(eltype(u)), B^2, B^2) : zeros(eltype(u), B^2, B^2)
#     H[diagind(H)] .= real(u[1, 1]) # store the secular component manually
#     u[1, 1] = 0 # remove because it breaks the structure of the loop below if included
#     make_real && (u .= real.(u))

#     # it is assumed that u[1, 1] == 0 -- otherwise, one would also need to prevent double pushing of the diagonal elements
#     @floop for c_u in axes(u, 2)
#         for r_u in axes(u, 1) # iterate over columns and rows of `u`
#             u[r_u, c_u] == 0 && continue
#             val = u[r_u, c_u]
#             for r_b in r_u:B # a value from `r_u`th row of `u` will be put in block-rows of `H` from `r_u`th to `B`th
#                 c_b = r_b - r_u + 1 # block-column where to place the value
#                 # fill the lower triangle of the block, including the main diagonal
#                 if c_u ≤ B # for `c_u` ≤ `B`, the value from `c_u`th column of `u` will be put to the `c_u`th lower diagonal of the block (`c_u=1` means main diagonal)
#                     for (r, c) in zip(c_u:B, 1:B+1-c_u)
#                         push_vals!(H; r_b, c_b, r, c, blocksize=B, val, conjugate=true)
#                     end
#                 # fill the upper triangle of the block, but this is not needed for a diagonal block (`r_b == c_b`), because then the upper triangle has already been filled by pushing the conjugate element
#                 elseif r_b != c_b # for `c_u` > `B`, the value from `c_u`th column of `u` will be put to the `2B-c_u`th upper diagonal of the block,
#                     c_u_inv = 2B-c_u+1
#                     for (r, c) in zip(1:B+1-c_u_inv, c_u_inv:B)
#                         push_vals!(H; r_b, c_b, r, c, blocksize=B, val, conjugate=true)
#                     end
#                 end
#             end
#         end
#     end
#     return H
# end

# """
# Based on results of a 2D FFT `u`, return the matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
# `make_real=true` will mutate `u`, taking the real parts of elements, which is useful if the original function is even and hence the transform is known to be real.
# For dense matrices, this is slower than [`fft_to_matrix_naive`](@ref); used only for testing purposes.
# """
# function _fft_to_matrix(u::AbstractMatrix; make_real=false)
#     N = size(u, 2) # number of points used for FFT
#     M = (N-1) ÷ 4 # maximum harmonic number (recall that N = 4M + 1 in the constructor)
#     B = 2M + 1 # the size of each block

#     H = make_real ? zeros(real(eltype(u)), B^2, B^2) : zeros(eltype(u), B^2, B^2)
#     H[diagind(H)] .= real(u[1, 1]) # store the secular component manually
#     make_real && (u .= real.(u))
    
#     @floop for c_u in axes(u, 2)
#         for r_u in axes(u, 1) # iterate over columns and rows of `u`
#             u[r_u, c_u] == 0 && continue
#             val = u[r_u, c_u]
#             if r_u ≤ B # when using rows 1 through B of `u` to fill the lower block-triangle of H, including the main block-diagonal
#                 d = 1 - r_u # (negative) block-diagonal number, where 0 is the main block-diagonal, -1 is first lower block-diagonal, etc.
#                 r_b_range = r_u:B  # a value from `r_u`th row of `u` will be put in block-rows of `H` from `r_u`th to `B`th
#             else # when using rows B+1 through end of `u` to fill the upper block-triangle of H
#                 d = B - (r_u-B) # (positive) block-diagonal number, where 0 is the main block-diagonal, +1 is first upper block-diagonal, etc.
#                 r_b_range = 1:r_u-B # a value from `r_u`th row of `u` will be put in block-rows of `H` from `r_u`th to `B`th
#             end
#             for r_b in r_b_range # block-rows where to place the value
#                 c_b = r_b + d # block-column where to place the value
#                 # fill the lower triangle of the block, including the main diagonal
#                 if c_u ≤ B # for `c_u` ≤ `B`, the value from `c_u`th column of `u` will be put to the `c_u`th lower diagonal of the block (`c_u=1` means main diagonal)
#                     for (r, c) in zip(c_u:B, 1:B+1-c_u)
#                         push_vals!(H; r_b, c_b, r, c, blocksize=B, val)
#                     end
#                 # fill the upper triangle of the block
#                 else # for `c_u` > `B`, the value from `c_u`th column of `u` will be put to the `2B-c_u`th upper diagonal of the block
#                     c_u_inv = 2B-c_u+1 
#                     for (r, c) in zip(1:B+1-c_u_inv, c_u_inv:B)
#                         push_vals!(H; r_b, c_b, r, c, blocksize=B, val)
#                     end
#                 end
#             end
#         end
#     end
#     return H
# end

# """
# Push value `val` to element (`r`, `c`) of the block (`r_b`, `c_b`) of `H`, with block size being `blocksize`.
# If `conjugate=true`, then the complex-conjugate element is also pushed.
# """
# function push_vals!(H; r_b, c_b, r, c, blocksize, val, conjugate=false)
#     i = (r_b-1)*blocksize + r
#     j = (c_b-1)*blocksize + c
#     H[i, j] = val
#     conjugate && (H[j, i] = val')
# end

# """
# Set to zero values of `u` that are smaller by magnitude than `threshold`.
# Based on the resulting number of nonzero elements in `u`, count the number of values that will be stored in 𝑈.
# """
# function _filter_count_rfft!(u::AbstractMatrix{<:Number}; fft_threshold::Real=0)
#     n_elem = 0
#     M = size(u, 2) ÷ 2
#     N = size(u, 1) # if `u` is really the result of `rfft`, then `N == M+1`, but we keep the calculation a bit more general
#     # do the first row of `u`, i.e. the diagonal blocks of 𝑈, separately
#     for c in axes(u, 2)
#         r = 1
#         if abs(u[r, c]) ≤ fft_threshold
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
#         if abs(u[r, c]) ≤ fft_threshold
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
# Based on results of 2D `rfft` output `u`, return `rows, cols, vals` tuple for constructing a sparse matrix.
# Optionally: `d` is a tuple of shifts in 𝑥 and 𝑦 directions, divided by the corresponding periods.
# `make_real=true` will take real parts of elements of `u`.
# """
# function _rft_to_matrix_sparse!(u::Matrix{<:Number}; fft_threshold::Real=0, make_real=false, d::Tuple{<:Real,<:Real}=(0, 0))
#     n_elem = filter_count_fft!(u; fft_threshold)

#     rows = Vector{Int64}(undef, n_elem)
#     cols = Vector{Int64}(undef, n_elem)
#     vals = Vector{make_real ? real(eltype(u)) : eltype(u)}(undef, n_elem)

#     u₀₀ = u[1, 1] # save the secular component
#     u[1, 1] = 0 # remove because it breaks the structure of the loop below if included
#     make_real && (u .= real.(u))

#     N = size(u, 2) # number of points used for FFT
#     M = (N-1) ÷ 4 # maximum harmonic number (recall that N = 4M + 1 in the constructor)
#     B = 2M + 1 # the size of each block

#     counter = 1

#     # it is assumed that u[1, 1] == 0 -- otherwise, one would also need to prevent double pushing of the diagonal elements
#     for c_u in axes(u, 2), r_u in axes(u, 1) # iterate over columns and rows of `u`
#         u[r_u, c_u] == 0 && continue
#         e = c_u <= M+1 ? cispi(2*(c_u-1)*d[1]) : cispi(2*(c_u-size(u, 2))*d[1]) # the factor is exp(2πi/L n) but division by `L` is absorbed in `d`
#         val = u[r_u, c_u] * e * cispi(2*(r_u-1)*d[2])
#         for r_b in r_u:B # a value from `r_u`th row of `u` will be put in block-rows of `H` from `r_u`th to `B`th.
#             c_b = r_b - r_u + 1 # block-column where to place the value
#             if c_u ≤ B # for `c_u` ≤ `B`, the value from `c_u`th column of `u` will be put to the `c_u`th lower diagonal of the block (`c_u=1` means main diagonal)
#                 for (r, c) in zip(c_u:B, 1:B+1-c_u)
#                     push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize=B, val)
#                     counter += 2
#                 end
#             elseif r_b != c_b # for `c_u` > `B`, the value from `c_u`th column of `u` will be put to the `2B-c_u`th upper diagonal of the block,
#                 c_u_inv = 2B-c_u+1 # but this is not needed for a diagonal block (`r_b == c_b`), because then the upper triangle of the block has already been filled by pushing the conjugate element
#                 for (r, c) in zip(1:B+1-c_u_inv, c_u_inv:B)
#                     push_vals!(rows, cols, vals, counter; r_b, c_b, r, c, blocksize=B, val)
#                     counter += 2
#                 end
#             end
#         end
#     end
#     n_diag = B^2 # number of diagonal elements in 𝑈
#     # fill positions of the diagonal elements
#     rows[end-n_diag+1:end] .= 1:n_diag
#     cols[end-n_diag+1:end] .= 1:n_diag
#     vals[end-n_diag+1:end] .= u₀₀

#     return sparse(rows, cols, vals)
# end