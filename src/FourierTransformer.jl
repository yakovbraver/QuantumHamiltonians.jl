mutable struct FourierTransformer{R,T,Plan,D} # T is the type of buffer, real for sin and cos, complex for cis
    xs::Matrix{R} # coordinates matrix: 1st columns contains 𝑥's, second contains 𝑦's, etc.
    M::Int # maximum harmonic number (will use -M:M for cis basis, 1:M for sin/cos)
    basis::Symbol
    buff::Array{T,D} # buffer for the result of the transform
    buff_im::Array{T,D} # additional buffer for storing the DCT of the imaginary part of a function
    did_complex_redft::Bool # a flag that is true if the last-performed transformation was a DST/DCT ("FFTW.REDFT") of a *complex* function
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