mutable struct FourierTransformer{R,T,Plan,D} # T is the type of buffer, real for sin and cos, complex for cis
    xs::Matrix{R} # coordinates matrix: 1st columns contains 𝑥's, second contains 𝑦's, etc.
    M::Int # maximum harmonic number (will use -M:M for cis basis, 1:M for sin/cos)
    basis::Symbol
    buff::Array{T,D} # buffer for the result of the transform
    buff_im::Array{T,D} # additional buffer for storing the DCT of the imaginary part of a function
    did_complex_redft::Bool # a flag that is true if the last-performed transformation was a DST/DCT ("FFTW.REDFT") of a *complex* function. The only reason why the type is mutable.
    plan::Plan
end

"`target_real` is only used in the sin and cos case: set to false if you plan to calculate DST or DCT for complex functions."
function FourierTransformer(xlims::AbstractVector{Tuple{R, R}}, M::Integer; basis::Symbol, target_real::Bool=true) where R <: AbstractFloat
    D = length(xlims)
    if basis == :cis
        N = 4M + 1 # number of points for FFT. This will yield harmonics from -2M to 2M
        xs = Matrix{R}(undef, N, D)
        for i in 1:D
            Lᵢ = xlims[i][2] - xlims[i][1]
            dxᵢ = Lᵢ/N
            xs[:, i] .= range(xlims[i][1], xlims[i][2]-dxᵢ, N)
        end
        buff = Array{Complex{R}}(undef, ntuple(Returns(N), D)) # a buffer for all (in-place) FFTs
        buff_im = similar(buff, ntuple(Returns(0), D)) # this buffer is not needed in the cis case; make it 0x0 (in `D` dimesions)
        plan = FFTW.plan_fft!(buff) # the savings of rfft are negligible, and the output is much less convenient to handle in `fft_to_matrix`, so using fft. Also, this way we can do FFT in-place
    else
        N = 2M + 1
        xs = Matrix{R}(undef, N, D)
        for d in 1:D
            xs[:, d] .= range(xlims[d][1], xlims[d][2], N)
        end
        buff = Array{R}(undef, ntuple(Returns(N), D)) # a buffer for all (in-place) FFTs
        buff_im = similar(buff, ntuple(Returns(target_real ? 0 : N), D)) # if `target_real`, then this buffer is not needed; make it 0x0 (in `D` dimesions)
        plan = FFTW.plan_r2r!(buff, FFTW.REDFT00)
    end

    did_complex_redft = false # value does not matter before any transform is performed
    return FourierTransformer(xs, M, basis, buff, buff_im, did_complex_redft, plan)
end

function transform!(ft::FourierTransformer, 𝑓::Function)
    (;xs, basis, buff, buff_im, plan) = ft
    N, D = size(xs)
    npoints = ft.basis == :cis ? N^D : (N-1)^D # total number of points, used for proper normalisation

    if basis == :cis || 𝑓(xs[1, 1:D]...) isa Real
        if D == 1
            buff .= 𝑓.(xs) ./ npoints
        elseif D == 2
            @views buff .= 𝑓.(xs[:, 1], xs[:, 2]') ./ npoints
        end
        plan * buff # in-place transform, weird syntax
        ft.did_complex_redft = false
    else # if basis is sin/cos and 𝑓 is complex
        # `FFTW.REDFT00` can only handle real ones. So we transform Re and Im separately.
        if D == 1
            for (ix, x) in enumerate(xs)
                buff[ix], buff_im[ix] = reim(𝑓(x)) ./ npoints
            end
        elseif D == 2
            for iy in axes(xs, 1), ix in axes(xs, 1)
                buff[ix, iy], buff_im[ix, iy] = reim(𝑓(xs[ix, 1], xs[iy, 2])) ./ npoints
            end
        end
        plan * buff
        plan * buff_im
        ft.did_complex_redft = true # will be used in fft_to_matrix_*D! to inclue `buff_im` when constructing the matrix
    end
end

"""
Use the result of the transform to fill `A` as a matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
"""
function fft_to_matrix!(A::AbstractMatrix{<:Number}, ft::FourierTransformer)
    D = ndims(ft.buff)
    if D == 1
        fft_to_matrix_1D!(A, ft)
    elseif D == 2
        fft_to_matrix_2D!(A, ft)
    end
end

"""
Construct from the result of 1D FFT `u` the matrix `U` indexed by (𝑗′ₓ, 𝑗ₓ).
"""
function fft_to_matrix_1D!(A::AbstractMatrix{<:Number}, ft::FourierTransformer)
    (;M, basis, buff, buff_im) = ft
    if basis == :cis
        A[diagind(A)] .= buff[1]
        for i in 2:M+1
            A[diagind(A, 1-i)] .= buff[i]       # fill lower triangle (including the diagonal)
            A[diagind(A, i-1)] .= buff[end-i+2] # fill upper triangle (including the diagonal)
        end
    elseif basis == :sin
        if ft.did_complex_redft
            @floop for jx in 1:M
                for j′x in 1:M
                    j₋x = abs(j′x-jx)
                    A[j′x, jx] = ( (buff[j₋x+1] - buff[j′x+jx+1]) + im*(buff_im[j₋x+1] - buff_im[j′x+jx+1]) ) / 2
                end
            end
        else
            @floop for jx in 1:M
                for j′x in 1:M
                    j₋x = abs(j′x-jx)
                    A[j′x, jx] = (buff[j₋x+1] - buff[j′x+jx+1]) / 2
                end
            end
        end
    end
end

"""
Use the result of 2D transform to fill `A` as a matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
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
        if ft.did_complex_redft
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
    end
end

########## Unused but correct and tested functions

"""
Based on results of a real 2D RFT `u`, return the matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
`make_real=true` will mutate `u`, taking the real parts of elements, which is useful if the original function is even and hence the transform is known to be real.
For dense matrices, this is slower than [`fft_to_matrix_naive`](@ref); used only for testing purposes.
"""
function _rfft_to_matrix!(u::AbstractMatrix{<:Number}; make_real=false)
    N = size(u, 2) # number of points used for FFT
    M = (N-1) ÷ 4 # maximum harmonic number (recall that N = 4M + 1 in the constructor)
    B = 2M + 1 # the size of each block
    H = make_real ? zeros(real(eltype(u)), B^2, B^2) : zeros(eltype(u), B^2, B^2)
    H[diagind(H)] .= real(u[1, 1]) # store the secular component manually
    u[1, 1] = 0 # remove because it breaks the structure of the loop below if included
    make_real && (u .= real.(u))

    # it is assumed that u[1, 1] == 0 -- otherwise, one would also need to prevent double pushing of the diagonal elements
    @floop for c_u in axes(u, 2)
        for r_u in axes(u, 1) # iterate over columns and rows of `u`
            u[r_u, c_u] == 0 && continue
            val = u[r_u, c_u]
            for r_b in r_u:B # a value from `r_u`th row of `u` will be put in block-rows of `H` from `r_u`th to `B`th
                c_b = r_b - r_u + 1 # block-column where to place the value
                # fill the lower triangle of the block, including the main diagonal
                if c_u ≤ B # for `c_u` ≤ `B`, the value from `c_u`th column of `u` will be put to the `c_u`th lower diagonal of the block (`c_u=1` means main diagonal)
                    for (r, c) in zip(c_u:B, 1:B+1-c_u)
                        push_vals!(H; r_b, c_b, r, c, blocksize=B, val, conjugate=true)
                    end
                # fill the upper triangle of the block, but this is not needed for a diagonal block (`r_b == c_b`), because then the upper triangle has already been filled by pushing the conjugate element
                elseif r_b != c_b # for `c_u` > `B`, the value from `c_u`th column of `u` will be put to the `2B-c_u`th upper diagonal of the block,
                    c_u_inv = 2B-c_u+1
                    for (r, c) in zip(1:B+1-c_u_inv, c_u_inv:B)
                        push_vals!(H; r_b, c_b, r, c, blocksize=B, val, conjugate=true)
                    end
                end
            end
        end
    end
    return H
end

"""
Based on results of a 2D FFT `u`, return the matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
`make_real=true` will mutate `u`, taking the real parts of elements, which is useful if the original function is even and hence the transform is known to be real.
For dense matrices, this is slower than [`fft_to_matrix_naive`](@ref); used only for testing purposes.
"""
function _fft_to_matrix(u::AbstractMatrix; make_real=false)
    N = size(u, 2) # number of points used for FFT
    M = (N-1) ÷ 4 # maximum harmonic number (recall that N = 4M + 1 in the constructor)
    B = 2M + 1 # the size of each block

    H = make_real ? zeros(real(eltype(u)), B^2, B^2) : zeros(eltype(u), B^2, B^2)
    H[diagind(H)] .= real(u[1, 1]) # store the secular component manually
    make_real && (u .= real.(u))
    
    @floop for c_u in axes(u, 2)
        for r_u in axes(u, 1) # iterate over columns and rows of `u`
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
                        push_vals!(H; r_b, c_b, r, c, blocksize=B, val)
                    end
                # fill the upper triangle of the block
                else # for `c_u` > `B`, the value from `c_u`th column of `u` will be put to the `2B-c_u`th upper diagonal of the block
                    c_u_inv = 2B-c_u+1 
                    for (r, c) in zip(1:B+1-c_u_inv, c_u_inv:B)
                        push_vals!(H; r_b, c_b, r, c, blocksize=B, val)
                    end
                end
            end
        end
    end
    return H
end

"""
Push value `val` to element (`r`, `c`) of the block (`r_b`, `c_b`) of `H`, with block size being `blocksize`.
If `conjugate=true`, then the complex-conjugate element is also pushed.
"""
function push_vals!(H; r_b, c_b, r, c, blocksize, val, conjugate=false)
    i = (r_b-1)*blocksize + r
    j = (c_b-1)*blocksize + c
    H[i, j] = val
    conjugate && (H[j, i] = val')
end