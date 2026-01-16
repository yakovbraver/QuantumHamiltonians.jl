struct FourierTransformer{R,T,PLAN} # T is the type of buffer, real for sin and cos, complex for cis
    xs::Vector{R}
    ys::Vector{R}
    M::Int # maximum harmonic number (will use -M:M for cis basis, 1:M for sin/cos)
    basis::Symbol
    buff::Matrix{T} # buffer for the result of the transform
    buff_im::Matrix{T} # additional buffer for storing the DCT of the imaginary part of a function
    plan::PLAN
end

"`target_real` is only used in the sin and cos case. Set to false if you plan to transform complex functions."
function FourierTransformer(xlims::Tuple{R, R}, ylims::Tuple{R, R}, M::Integer; basis::Symbol, target_real::Bool=true) where R <: Real
    Lx, Ly = xlims[2]-xlims[1], ylims[2]-ylims[1]

    if basis == :cis
        N = 4M + 1 # number of points for FFT. This will yield harmonics from -2M to 2M
        dx, dy = Lx/N, Ly/N
        xs = range(xlims[1], xlims[2]-dx, N)
        ys = range(ylims[1], ylims[2]-dy, N)

        buff = Matrix{Complex{R}}(undef, N, N) # a buffer for all (in-place) FFTs
        plan = FFTW.plan_fft!(buff) # the savings of rfft are negligible, and the output is much less convenient to handle in `fft_to_matrix`, so using fft. Also, this way we can do FFT in-place
    else
        N = 2M + 1
        xs = range(xlims[1], xlims[2], N)
        ys = range(ylims[1], ylims[2], N)

        buff = Matrix{R}(undef, N, N)
        buff_im = target_real ? R[;;] : Matrix{R}(undef, N, N) # if `target_real`, then this buffer is not needed; make it 0x0
        plan = FFTW.plan_r2r!(buff, FFTW.REDFT00)
    end

    return FourierTransformer(xs, ys, M, basis, buff, buff_im, plan)
end

function transform!(ft::FourierTransformer, 𝑓::Function)
    (;xs, ys, basis, buff, buff_im, plan) = ft
    N = length(ft.xs)
    npoints = ft.basis == :cis ? N^2 : (N-1)^2 # total number of points, used for proper normalisation

    if basis == :cis || 𝑓(xs[1], ys[1]) isa Real
        buff .= 𝑓.(xs, ys') ./ npoints
        plan * buff # in-place transform, weird syntax
    else # if basis is sin/cos and 𝑓 is complex
        # `FFTW.REDFT00` can only handle real ones. So we transform Re and Im separately.
        buff, buff_im = reim.(𝑓.(xs, ys')) # TODO make sure this does not allocate new matrices

        buff .= 𝑓.(xs, ys') ./ npoints
        plan * buff # in-place transform, weird syntax

        buff_im .= 𝑓.(xs, ys') ./ npoints
        plan * buff_im # in-place transform, weird syntax
    end
end

"""
Use the result of 2D FT to fill `A` as a matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
"""
function fft_to_matrix!(A::Matrix{<:Number}, ft::FourierTransformer)
    (;M, basis, buff, buff_im) = ft
    if basis == :cis
        B = 2M + 1
        # TODO add check size(A) .== B^2
        buff = FFTW.fftshift(buff) # indexing into `u` is more convenient if we shift TODO rewrite without relying on this
        # We sipmly go over each element of `A`, assigning an appropriate element of `u`.
        # In the dense case it is preferred over (since it's faster than) [`fft_to_matrix_sparse!`](@ref)
        # because even if `buffer[i, j]=0`, the corresponding elements of `A` still must be accessed to be set to zero.
        @floop for jx in 1:B
            for jy in 1:B, j′x in 1:B, j′y in 1:B
                j₋x = j′x - jx
                j₋y = j′y - jy
                A[(j′x-1)B+j′y, (jx-1)B+jy] = buff[j₋x+B, j₋y+B]
            end
        end
    elseif basis == :sin
        # TODO add check size(A) .== M^2
        if isempty(buff_im)
            @floop for jx in 1:M
                for jy in 1:M, j′x in 1:M, j′y in 1:M
                    j₋x = abs(j′x-jx)
                    j₋y = abs(j′y-jy)
                    A[(j′x-1)M+j′y, (jx-1)M+jy] = (buff[j₋x+1, j₋y+1] - buff[j₋x+1, j′y+jy+1] - buff[j′x+jx+1, j₋y+1] + buff[j′x+jx+1, j′y+jy+1]) / 4
                end
            end
        else
            @floop for jx in 1:M
                for jy in 1:M, j′x in 1:M, j′y in 1:M
                    j₋x = abs(j′x-jx)
                    j₋y = abs(j′y-jy)
                    A[(j′x-1)M+j′y, (jx-1)M+jy] = (buff[j₋x+1, j₋y+1] - buff[j₋x+1, j′y+jy+1] - buff[j′x+jx+1, j₋y+1] + buff[j′x+jx+1, j′y+jy+1]) / 4 +
                                             im * (buff_im[j₋x+1, j₋y+1] - buff_im[j₋x+1, j′y+jy+1] - buff_im[j′x+jx+1, j₋y+1] + buff_im[j′x+jx+1, j′y+jy+1]) / 4
                end
            end
        end
    end
    return A
end