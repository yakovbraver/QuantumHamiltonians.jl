"An object used to perform Fourier transformations for `XSpaceHamiltonian`."
struct FourierTransformerX{R, D, PlanForward, PlanBackward, PlanBothways!}
    xs::Matrix{R} # grid points: 1st column contains 𝑥's, second contains 𝑦's, etc.
    basis::Symbol
    normalisation::Int # normalisation used in the sin and cos cases. Integer because this is essentially just the number of points
    buff_re::Array{R, D} # buffer for DST/DCT of the real part of a function
    buff_im::Array{R, D} # buffer for DST/DCT of the imaginary part of a function
    plan_forward::PlanForward # plan for forward transform
    plan_backward::PlanBackward # plan for backward transform
    plan_bothways!::PlanBothways! # in-place plan, used in the sin/cos case when acting on complex functions to avoid two additional buffers
end

"Construct a `FourierTransformerX` object for a `basis` having maximum harmonic number `M`."
function FourierTransformerX(xlims::AbstractVector{Tuple{R, R}}, M::Integer; basis::Symbol) where R <: AbstractFloat
    D = length(xlims) # number of spatial dimensions
    if basis == :cis
        N = 2M # number of points for FFT (the same for each dimension). This will yield harmonics -M:M-1. User is recommended to pass M = 2ⁿ; N = 2⋅2ⁿ is optimal for cis-transform
        xs = Matrix{R}(undef, N, D) # grid points: 1st column contains 𝑥's, second contains 𝑦's, etc.
        for i in 1:D
            L = xlims[i][2] - xlims[i][1]
            dx = L / N
            xs[:, i] .= range(xlims[i][1], xlims[i][2]-dx, N)
        end
        # buffers are not needed in the cis case; make them 0x0 (in `D` dimesions)
        buff_re = Array{R}(undef, ntuple(Returns(0), D))
        buff_im = similar(buff_re)
        # a buffer used only for creating the plans
        buff = Array{Complex{R}}(undef, ntuple(Returns(N), D)) 
        plan_forward = FFTW.plan_fft(buff)
        plan_backward = FFTW.plan_ifft(buff)
        # in-place map is not needed in the cis case; just make a reference
        plan_bothways! = plan_forward
    elseif basis == :cos
        N = M + 1 # This will yield harmonics 0:M. User is recommended to pass M = 2ⁿ; N = 2ⁿ+1 is optimal for cos-transform
        xs = Matrix{R}(undef, N, D)
        for i in 1:D
            xs[:, i] .= range(xlims[i][1], xlims[i][2], N)
        end
        buff_re = Array{R}(undef, ntuple(Returns(N), D)) # a buffer for all (in-place) FFTs
        buff_im = similar(buff_re)
        plan_forward = FFTW.plan_r2r(buff_re, FFTW.REDFT00) # note that `REDFT00` is its own inverse
        plan_backward = plan_forward # same plan for backward
        # in-place map needed when acting on a complex vector
        plan_bothways! = FFTW.plan_r2r!(buff_re, FFTW.REDFT00) # note that `REDFT00` is its own inverse
    else # basis == :sin
        N = M # This will yield harmonics 1:M. User is recommended to pass M = 2ⁿ-1; N = 2ⁿ-1 is optimal for sin-transform
        xs = Matrix{R}(undef, N, D)
        for i in 1:D
            L = xlims[i][2] - xlims[i][1]
            dx = L / (N+1)
            xs[:, i] .= range(xlims[i][1]+dx, xlims[i][2]-dx, N)
        end
        buff_re = Array{R}(undef, ntuple(Returns(N), D)) # a buffer for all (in-place) FFTs
        buff_im = similar(buff_re)
        plan_forward = FFTW.plan_r2r(buff_re, FFTW.RODFT00) # note that `RODFT00` is its own inverse
        plan_backward = plan_forward # same plan for backward
        # in-place map needed when acting on a complex vector
        plan_bothways! = FFTW.plan_r2r!(buff_re, FFTW.RODFT00) # note that `RODFT00` is its own inverse
    end

    # normalisation used in the sin and cos cases
    normalisation = basis == :sin ? (2(N+1))^D : (2(N-1))^D

    return FourierTransformerX(xs, basis, normalisation, buff_re, buff_im, plan_forward, plan_backward, plan_bothways!)
end

"""
Transform a discretised function `f_in`, which can be either in x-space or p-space, writing the result to `f_out`.
The transformation is forward or backward depending on the `direction` keyword argument.
`normalise` will normalise the transform in the sin/cos case. In the cis case, `normalise` has no effect; the backward transform automatically includes normalisation.
"""
function transform!(f_out::AbstractArray{<:Number}, ft::FourierTransformerX, f_in::AbstractArray{<:Number}; direction::Symbol=:forward, normalise::Bool=false)
    (;basis, normalisation, buff_re, buff_im) = ft    
    # transform              
    if basis == :cis || eltype(f_in) <: Real
        if direction == :forward
            mul!(f_out, ft.plan_forward, f_in)
        else
            mul!(f_out, ft.plan_backward, f_in)
        end
        if basis != :cis && normalise
            @turbo f_out ./= normalisation
        end
    else # if basis is sin/cos and `f_in` is complex
        # `FFTW.RxDFT00` can only handle real input. So we transform Re and Im separately.
        for i in eachindex(f_in)
            buff_re[i], buff_im[i] = reim(f_in[i])
        end
        if normalise
            @turbo buff_re ./= normalisation
            @turbo buff_im ./= normalisation
        end
        # apply plans; forward is same as backward
        ft.plan_bothways! * buff_re
        ft.plan_bothways! * buff_im
        f_out .= Complex.(buff_re, buff_im)
    end
    return
end