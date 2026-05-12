"An object used to perform Fourier transformations for `PSpaceHamiltonian`."
mutable struct FourierTransformerP{R, T, PlanForward, PlanBackward, D} # T is the type of buffer, real for sin and cos, complex for cis
    xs::Matrix{R} # coordinates matrix: 1st column contains 𝑥's, second contains 𝑦's, etc.
    M::Int # maximum harmonic number (will use -M:M for cis basis, 1:M for sin, 0:M for cos)
    norm_forward::R # normalisation factor for forward transform
    norm_backward::R # normalisation factor for backward transform
    basis::Symbol
    buff::Array{T, D} # buffer for the result of the transform
    buff_im::Array{T, D} # additional buffer: in the sin/cos cases it stores the imaginary part of a function. In the cis cases it is used for fftshift
    did_complex_rxdft::Bool # a flag that is true if the last-performed transformation was a DST/DCT ("FFTW.RODFT"/"FFTW.REDFT") of a *complex* function. The only reason why the type is mutable.
    plan_forward::PlanForward # plan for forward transform
    plan_backward::PlanBackward # plan for backward transform
end

"""
Set `target_rank=1` if you are making the transform for building a Fourier-space state (using `fft_to_state`),
set `target_rank=2` if you are making the transform for building a Fourier-space operator (using `fft_to_operator`). The latter needs twice the number of harmonics.
"""
function FourierTransformerP(xlims::AbstractVector{Tuple{R, R}}, M::Integer; basis::Symbol, target_rank::Integer=2) where R <: AbstractFloat
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
        buff_im = similar(buff, ntuple(Returns(N), D)) # allocate full buffer for backward transform fftshift
        # the time savings of rfft are negligible for us, and the output is much less convenient to handle in `fft_to_operator`, so using fft. Also, this way we can do FFT in-place
        plan_forward = FFTW.plan_fft!(buff) # using unnormalised plan
        plan_backward = FFTW.plan_bfft!(buff) # using unnormalised `bfft` because the "1/N" used in `ifft` is not right for our use case
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
            buff_im = similar(buff)
            plan_forward = FFTW.plan_r2r!(buff, FFTW.REDFT00) # note that `REDFT00` is its own inverse
            plan_backward = plan_forward # same plan for backward
        else # basis == :sin && target_rank == 1 # the only case when we need DST
            N = M
            xs = Matrix{R}(undef, N, D)
            for i in 1:D
                L[i] = xlims[i][2] - xlims[i][1]
                dx[i] = L[i] / (N+1)
                xs[:, i] .= range(xlims[i][1]+dx[i], xlims[i][2]-dx[i], N)
            end
            buff = Array{R}(undef, ntuple(Returns(N), D)) # a buffer for all (in-place) FFTs
            buff_im = similar(buff)
            plan_forward = FFTW.plan_r2r!(buff, FFTW.RODFT00) # note that `RODFT00` is its own inverse
            plan_backward = plan_forward # same plan for backward
        end
        L .*= 2 # for proper calculation of `normalisation`, to cancel the 2 included in FFTW DST/DCT
    end

    # Calculate forward normalisation
    norm_forward = target_rank == 1 ? prod(dx ./ sqrt.(L)) : prod(dx./L) # the latter equals 1/N^D, 1/(N+1)^D, 1/(N-1)^D in the cis, sin, cos cases respectively
    
    # Calculate backward normalisation
    norm_backward = target_rank == 1 ? prod(inv.(sqrt.(L))) : prod(inv.(L))
    
    did_complex_rxdft = false # value does not matter before any transform is performed
    return FourierTransformerP(xs, M, norm_forward, norm_backward, basis, buff, buff_im, did_complex_rxdft, plan_forward, plan_backward)
end

# TODO consider moving sample! to some other file

"""
Sample analytic D-argument function `𝑓` at points in `xs`, where first column is 𝑥, second is 𝑦, etc.
Result is written into `buff`, which can be either D-dimensional or flattened.
"""
function sample!(buff::AbstractArray{<:Number}, 𝑓::Function, xs::AbstractVecOrMat{<:Number})
    D = size(xs, 2)
    buff_shaped = reshape(buff, ntuple(Returns(size(xs, 1)), D))
    
    if D == 1
        buff_shaped .= 𝑓.(xs)
    elseif D == 2
        @views buff_shaped .= 𝑓.(xs[:, 1], xs[:, 2]') # structurally corresponds to f[i, j] = 𝑓(xs[i], ys[j]), i.e. first index is 𝑥, second is 𝑦. 𝑥↓ 𝑦→
    elseif D == 3
        for iz in axes(buff_shaped, 3), iy in axes(buff_shaped, 2), ix in axes(buff_shaped, 1)
            buff_shaped[ix, iy, iz] = 𝑓(xs[ix, 1], xs[iy, 2], xs[iz, 3])
        end
    else
        for I in Iterators.product(axes(buff_shaped)...)
            buff_shaped[I...] = 𝑓(ntuple(d -> xs[I[d], d], D)...)
        end
    end
    return buff
end

"""
Sample analytic D-argument function `𝑓` at points in `xs`, where first column is 𝑥, second is 𝑦, etc.
Real part of the result is written into `buff_re`, and imaginary to `buff_im`. They can be either D-dimensional or flattened.
"""
function sample!(buff_re::AbstractArray, buff_im::AbstractArray, 𝑓::Function, xs::AbstractVecOrMat{<:Number})
    D = size(xs, 2)
    buff_re_shaped = reshape(buff_re, ntuple(Returns(size(xs, 1)), D))
    buff_im_shaped = reshape(buff_im, ntuple(Returns(size(xs, 1)), D))

    if D == 1
        for ix in axes(buff_re_shaped, 1)
            buff_re_shaped[ix], buff_im_shaped[ix] = reim(𝑓(xs[ix]))
        end
    elseif D == 2
        for iy in axes(buff_re_shaped, 2), ix in axes(buff_re_shaped, 1)
            buff_re_shaped[ix, iy], buff_im_shaped[ix, iy] = reim(𝑓(xs[ix, 1], xs[iy, 2]))
        end
    elseif D == 3
        for iz in axes(buff_re_shaped, 3), iy in axes(buff_re_shaped, 2), ix in axes(buff_re_shaped, 1)
            buff_re_shaped[ix, iy, iz], buff_im_shaped[ix, iy, iz] = reim(𝑓(xs[ix, 1], xs[iy, 2], xs[iz, 3]))
        end
    else
        for I in Iterators.product(axes(buff_re_shaped)...)
            buff_re_shaped[I...], buff_im_shaped[I...] = reim(𝑓(ntuple(d -> xs[I[d], d], D)...))
        end
    end
    return buff_re, buff_im
end

"Transform a callable function `𝑓` given in x-space to p-space or reverse."
function transform!(ft::FourierTransformerP, 𝑓::Function; direction::Symbol=:forward)
    (;xs, norm_forward, norm_backward, basis, buff, buff_im, plan_forward, plan_backward) = ft
    normalisation = direction == :forward ? norm_forward : norm_backward
    plan = direction == :forward ? plan_forward : plan_backward
    D = size(xs, 2)

    if basis == :cis || 𝑓(xs[1, 1:D]...) isa Real
        sample!(buff, 𝑓, xs)
        buff .*= normalisation
        plan * buff # in-place transform, weird syntax
        ft.did_complex_rxdft = false
    else # if basis is sin/cos and `𝑓` is complex
        # `FFTW.RxDFT00` can only handle real input. So we transform Re and Im separately.
        sample!(buff, buff_im, 𝑓, xs)
        @turbo buff .*= normalisation
        @turbo buff_im .*= normalisation
        plan * buff
        plan * buff_im
        ft.did_complex_rxdft = true # will be used in fft_to_operator_*D! to include `buff_im` when constructing the matrix
    end
    return
end

"""
Transform a discretised function `f`, which can be either in x-space or p-space. Store the result in `ft.buff` and `ft.buff_im`.

The transformation is forward (= to p-space) or backward (= to x-space) depending on the `direction` keyword argument.
For `direction=:forward`, `f` is assumed to be an x-space D-dimensional tensor indexed by (x, y, …). It is transformed and written directly to the buffer (shapes match).
Forward transformation is used both for transforming states and operators.

For `direction=:backward`, `f` is assumed to be a p-space state, either a D-dimensional tensor indexed by (𝑗ˣ, 𝑗ʸ, …) or a 1D vector indexed by (⋯𝑗ʸ𝑗ˣ). 
In the latter case, prior to transforming, the input tensor is reshaped to a D-dimensional tensor like the buffer.
Backward transformation is used only for transforming states -- we never need to transform operators back to x-space.
"""
function transform!(ft::FourierTransformerP, f::AbstractArray{<:Number}; direction::Symbol=:forward)
    (;norm_forward, norm_backward, basis, buff, buff_im, plan_forward, plan_backward) = ft
    normalisation = direction == :forward ? norm_forward : norm_backward
    plan = direction == :forward ? plan_forward : plan_backward
    
    D = ndims(buff)
    # preparation of the input if going to x-space
    if direction == :backward
        f_reshaped = reshape(f, ntuple(Returns(size(buff, 1)), D)) # make `f` same shape as the buffer. It might be already, but this goes through anyway.
        if basis == :cis # then do `ifftshift` because `f` is stored in -M:M ordering, while FFT assumes 0:M,-M:-1
            FFTW.ifftshift!(buff_im, f_reshaped) # using `buff_im` as a convenient buffer (specifically made for this case)
            f_input = buff_im
        elseif basis == :sin
            f_input = f_reshaped
        else # basis == :cos
            f_input = copy(f_reshaped)
            # proper normalisation of the zeroth and last harmonic
            if D == 1
                f_input[1] *= √2; f_input[end] *= √2
            elseif D == 2
                # edges (without corners)
                f_input[1, 2:end-1] *= √2; f_input[end, 2:end-1] *= √2; f_input[2:end-1, 1] *= √2; f_input[2:end-1, end] *= √2;
                # corners 
                f_input[1, 1] *= 2; f_input[end, 1] *= 2; f_input[1, end] *= 2; f_input[end, end] *= 2;
            else
                error("transform! with basis=:cos and direction=:backward not implemented in $(D)D.")
            end
        end
    else # going to p-space, no setup required
        f_input = f
    end

    # transform              
    if basis == :cis || eltype(f) <: Real
        buff .= f_input .* normalisation
        plan * buff # in-place transform, weird syntax
        ft.did_complex_rxdft = false
    else # if basis is sin/cos and `f` is complex
        # `FFTW.RxDFT00` can only handle real input. So we transform Re and Im separately.
        for i in eachindex(f)
            buff[i], buff_im[i] = reim(f_input[i]) .* normalisation
        end
        plan * buff
        plan * buff_im
        ft.did_complex_rxdft = true # will be used in `fft_to_state` and `fft_to_operator!` to include `buff_im`
    end
    return
end

"""
Use the result of the D-dimensional transform contained in `ft.buff` to construct a p-space or x-space state.
For `direction=:forward`, use reshaping to construct a p-space state as a 1D vector indexed by (⋯𝑗ʸ𝑗ˣ). 
For `direction=:backward`, construct an x-space state as a D-dimensional array indexed by (x, y, …). 
Pass `makereal=true` to drop the imaginary part of `ft.buff` -- useful in the cis case when constructing x-space state if you know the state must be real.
"""
function fft_to_state(ft::FourierTransformerP{R, T}; makereal=false, direction::Symbol=:forward) where {R, T}
    (;buff, basis) = ft

    if basis == :cis
        if makereal
            ψ_type = R
            buff .= real.(buff)
        else
            ψ_type = T
        end
    else # sin/cos
        ψ_type = ft.did_complex_rxdft ? Complex{T} : T
    end

    if direction == :forward
        ψ = Vector{ψ_type}(undef, length(buff)) # a p-space state that is a linearised 1D vector, constructed from the D-dimensional `buffer` (`length` gives total number of elements)
    else
        ψ = similar(buff, ψ_type) # an x-space state that is a D-dimensional tensor, like `buff`
    end

    fft_to_state!(ψ, ft; direction) # we do not pass `makereal` because already performed this above

    return ψ
end

"""
Use the result of the D-dimensional transform contained in `ft.buff` to construct (fill) a p-space or x-space state `ψ`.
For `direction=:forward`, use reshaping to construct a p-space state `ψ` as a 1D vector indexed by (⋯𝑗ʸ𝑗ˣ).
For `direction=:backward`, construct an x-space state `ψ`. If it is a D-dimensional array, then it will be filled as an array indexed by (x, y, ⋯). If it is a vector, if will be a flattened version of this array.
Pass `makereal=true` to drop the imaginary part of `ft.buff` -- useful in the cis case when constructing x-space state if you know the state must be real.
"""
function fft_to_state!(ψ::AbstractArray{<:Number, D}, ft::FourierTransformerP; makereal=false, direction::Symbol=:forward) where D
    (;buff, buff_im, basis) = ft
    makereal && (buff .= real.(buff))
    if basis == :cis
        if direction == :forward # `ψ` is in p-space, and is a 1D vector, so the buffer must be reshaped (linearised)
            FFTW.fftshift!(buff_im, buff) # using `buff_im` as a convenient buffer (specifically made for this case)
            copyto!(ψ, buff_im) # `copyto!` copies contiguously even though shapes are different (`ψ` is 1D vector, `buff_im` is D-dimensional); this is like `ψ = buff_im[:]`
        else # `ψ` is in x-space, so just copy (contiguously)
            copyto!(ψ, buff)
        end
    else # sin/cos
        if direction == :forward # `ψ` is in p-space, and is a 1D vector, so the buffer must be reshaped (linearised)
            # proper normalisation of the zeroth and last harmonic; do this for the D-dimensional buffer (more convenient than for linearised ψ)
            if basis == :cos
                if D == 1
                    buff[1] /= √2; buff[end] /= √2
                    ft.did_complex_rxdft && (buff_im[1] /= √2; buff_im[end] /= √2)
                elseif D == 2
                    # edges (without corners)
                    buff[1, 2:end-1] /= √2; buff[end, 2:end-1] /= √2; buff[2:end-1, 1] /= √2; buff[2:end-1, end] /= √2;
                    # corners 
                    buff[1, 1] /= 2; buff[end, 1] /= 2; buff[1, end] /= 2; buff[end, end] /= 2;
                    # repeat for `buff_im`                    
                    if ft.did_complex_rxdft
                        buff_im[1, 2:end-1] /= √2; buff_im[end, 2:end-1] /= √2; buff_im[2:end-1, 1] /= √2; buff_im[2:end-1, end] /= √2;
                        buff_im[1, 1] /= 2; buff_im[end, 1] /= 2; buff_im[1, end] /= 2; buff_im[end, end] /= 2;
                    end
                else
                    error("fft_to_state! with basis=:cos and direction=:forward not implemented in $(D)D.")
                end
            end
            copyto!(ψ, buff) # `copyto!` copies contiguously even though shapes are different; this is like `ψ = buff[:]`
            ft.did_complex_rxdft && (ψ .+= im .* reshape(buff_im, :))
        else # `ψ` is in x-space, so just copy (contiguously)
            copyto!(ψ, buff)
            ft.did_complex_rxdft && (ψ .+= im .* reshape(buff_im, :))
        end
    end
    return
end

"""
Use the result of the transform to construct a matrix indexed by (⋯𝑗ʸ′𝑗ˣ′, ⋯𝑗ʸ𝑗ˣ).
If `makesparse=true`, a sparse matrix is returned, with values below `threshold` in magnitude filtered out. By default, a dense matrix is returned.
If `makereal=true`, a real matrix (of type `R`) is returned, which is useful in the cis case if you wish to drop the imaginary part of `ft.buff`.
"""
function fft_to_operator(ft::FourierTransformerP{R, T}; makesparse::Bool=false, makereal=false, threshold::Real=√(eps(R))) where {R, T}
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
            fft_to_operator_sparse!(rows, cols, vals, ft)
            A = sparse(rows, cols, vals)
        else
            error("Sparse fft_to_operator not implemented for basis = $basis. Only implemented for basis = :cis.")
        end
    else # dense
        A = Matrix{A_type}(undef, B, B)
        fft_to_operator!(A, ft) # we do not pass `makereal` because already performed this above
    end
    return A
end

################ Dense ################

"""
Use the result of the transform to fill `A` as a matrix indexed by (⋯𝑗ʸ′𝑗ˣ′, ⋯𝑗ʸ𝑗ˣ). 
`makereal=true` is useful in the cis case if you wish to drop the imaginary part of `ft.buff`.
"""
function fft_to_operator!(A::AbstractMatrix{<:Number}, ft::FourierTransformerP; makereal=false)
    D = ndims(ft.buff)
    makereal && (ft.buff .= real.(ft.buff))
    if D == 1
        fft_to_operator_1D!(A, ft)
    elseif D == 2
        fft_to_operator_2D!(A, ft)
    else
        error("fft_to_operator_$(D)D! not implemented.")
    end
end

"""
Use the result of the 1D transform to fill `A` as a matrix indexed by (𝑗ˣ′, 𝑗ˣ).
"""
function fft_to_operator_1D!(A::AbstractMatrix{<:Number}, ft::FourierTransformerP)
    (;M, basis, buff, buff_im) = ft
    if basis == :cis
        A[diagind(A)] .= buff[1]
        for i in 2:2M+1
            A[diagind(A, 1-i)] .= buff[i]       # fill lower triangle
            A[diagind(A, i-1)] .= buff[end-i+2] # fill upper triangle
        end
    elseif basis == :sin
        if ft.did_complex_rxdft
            for jˣ in 1:M # not enough work for @floop, slows down execution (checked for M up to 300)
                for jˣ′ in 1:M
                    jˣ⁻ = abs(jˣ′-jˣ)
                    A[jˣ′, jˣ] = buff[jˣ⁻+1] - buff[jˣ′+jˣ+1] + im*(buff_im[jˣ⁻+1] - buff_im[jˣ′+jˣ+1])
                end
            end
        else
            @turbo for jˣ in 1:M
                for jˣ′ in 1:M
                    jˣ⁻ = abs(jˣ′-jˣ)
                    A[jˣ′, jˣ] = buff[jˣ⁻+1] - buff[jˣ′+jˣ+1]
                end
            end
        end
    else # basis == :cos
        if ft.did_complex_rxdft
            for jˣ in 0:M
                ζˣ = ifelse(jˣ == 0, 2, 1)
                for jˣ′ in 0:M
                    ζˣ′ = ifelse(jˣ′ == 0, 2, 1)
                    jˣ⁻ = abs(jˣ′-jˣ)
                    A[jˣ′+1, jˣ+1] = ( (buff[jˣ⁻+1] + buff[jˣ′+jˣ+1]) + im*(buff_im[jˣ⁻+1] + buff_im[jˣ′+jˣ+1]) ) / √(ζˣ*ζˣ′)
                end
            end
        else
            @turbo for jˣ in 0:M
                ζˣ = ifelse(jˣ == 0, 2, 1)
                for jˣ′ in 0:M
                    ζˣ′ = ifelse(jˣ′ == 0, 2, 1)
                    jˣ⁻ = abs(jˣ′-jˣ)
                    A[jˣ′+1, jˣ+1] = (buff[jˣ⁻+1] + buff[jˣ′+jˣ+1]) / √(ζˣ*ζˣ′)
                end
            end
        end
    end
    return
end

"""
Use the result of the 2D transform to fill `A` as a matrix indexed by (𝑗ʸ′𝑗ˣ′, 𝑗ʸ𝑗ˣ).
"""
function fft_to_operator_2D!(A::AbstractMatrix{<:Number}, ft::FourierTransformerP)
    (;M, basis, buff, buff_im) = ft
    # We simply go over each element of `A`, assigning an appropriate element of `u`.
    # In the dense case it is preferred over (since it's faster than) [`fft_to_operator_sparse!`](@ref)
    # because even if `buff[i, j]=0`, the corresponding elements of `A` still must be accessed to be set to zero.
    # In `A`, the x-index must be fastest, like in `buff` so that when an eigenvector of `A` is reshaped into a 2D matrix, and FFT is calculated, the first index is x and second is y.
    if basis == :cis
        B = 2M + 1
        FFTW.fftshift!(buff_im, buff) # indexing into `u` is more convenient if we shift. We shift into `buff_im`
        @floop for jʸ in 1:B
            for jˣ in 1:B
                for jʸ′ in 1:B
                    jʸ⁻ = jʸ′ - jʸ + B
                    for jˣ′ in 1:B
                        jˣ⁻ = jˣ′ - jˣ + B
                        A[(jʸ′-1)B+jˣ′, (jʸ-1)B+jˣ] = buff_im[jˣ⁻, jʸ⁻]
                    end
                end
            end
        end
    elseif basis == :sin
        if ft.did_complex_rxdft
            @floop for jʸ in 1:M
                for jˣ in 1:M
                    for jʸ′ in 1:M
                        jʸ⁻ = abs(jʸ′ - jʸ) + 1 # +1 because of 1-based indexing
                        jʸ⁺ =     jʸ′ + jʸ  + 1 # +1 because of 1-based indexing
                        for jˣ′ in 1:M
                            jˣ⁻ = abs(jˣ′ - jˣ) + 1
                            jˣ⁺ =     jˣ′ + jˣ  + 1
                            A[(jʸ′-1)M+jˣ′, (jʸ-1)M+jˣ] = buff[jˣ⁻, jʸ⁻] -    buff[jˣ⁻, jʸ⁺] -    buff[jˣ⁺, jʸ⁻] +    buff[jˣ⁺, jʸ⁺] +
                                                 im * (buff_im[jˣ⁻, jʸ⁻] - buff_im[jˣ⁻, jʸ⁺] - buff_im[jˣ⁺, jʸ⁻] + buff_im[jˣ⁺, jʸ⁺])
                        end
                    end
                end
            end
        else
            @floop for jʸ in 1:M
                for jˣ in 1:M
                    for jʸ′ in 1:M
                        jʸ⁻ = abs(jʸ′ - jʸ) + 1
                        jʸ⁺ =     jʸ′ + jʸ  + 1
                        for jˣ′ in 1:M
                            jˣ⁻ = abs(jˣ′ - jˣ) + 1
                            jˣ⁺ =     jˣ′ + jˣ  + 1
                            A[(jʸ′-1)M+jˣ′, (jʸ-1)M+jˣ] = buff[jˣ⁻, jʸ⁻] - buff[jˣ⁻, jʸ⁺] - buff[jˣ⁺, jʸ⁻] + buff[jˣ⁺, jʸ⁺]
                        end
                    end
                end
            end
        end
    else # basis == :cos
        b = M + 1 # not `B` to prevent Core.Box :(
        if ft.did_complex_rxdft
            @floop for jʸ in 0:M
                ζʸ = ifelse(jʸ == 0, 2, 1)
                for jˣ in 0:M
                    ζˣ = ifelse(jˣ == 0, 2, 1)
                    for jʸ′ in 0:M
                        ζʸ′ = ifelse(jʸ′ == 0, 2, 1)
                        jʸ⁻ = abs(jʸ′ - jʸ) + 1
                        jʸ⁺ =     jʸ′ + jʸ  + 1
                        for jˣ′ in 0:M
                            ζˣ′ = ifelse(jˣ′ == 0, 2, 1)
                            jˣ⁻ = abs(jˣ′ - jˣ) + 1
                            jˣ⁺ =     jˣ′ + jˣ  + 1
                            A[jʸ′*b + jˣ′+1, jʸ*b + jˣ+1] = (buff[jˣ⁻, jʸ⁻] +    buff[jˣ⁻, jʸ⁺] +    buff[jˣ⁺, jʸ⁻] +    buff[jˣ⁺, jʸ⁺] +
                                                    im * (buff_im[jˣ⁻, jʸ⁻] + buff_im[jˣ⁻, jʸ⁺] + buff_im[jˣ⁺, jʸ⁻] + buff_im[jˣ⁺, jʸ⁺]) ) / √(ζˣ*ζˣ′*ζʸ*ζʸ′)
                        end
                    end
                end
            end
        else
            @floop for jʸ in 0:M # @floop gives ~x6 speedup (on 8 CPU cores) for M=128. @turbo slows down, @tturbo gives ~x3 speedup
                ζʸ = ifelse(jʸ == 0, 2, 1)
                for jˣ in 0:M
                    ζˣ = ifelse(jˣ == 0, 2, 1)
                    for jʸ′ in 0:M
                        ζʸ′ = ifelse(jʸ′ == 0, 2, 1)
                        jʸ⁻ = abs(jʸ′ - jʸ) + 1
                        jʸ⁺ =     jʸ′ + jʸ  + 1
                        for jˣ′ in 0:M
                            ζˣ′ = ifelse(jˣ′ == 0, 2, 1)
                            jˣ⁻ = abs(jˣ′ - jˣ) + 1
                            jˣ⁺ =     jˣ′ + jˣ  + 1
                            A[jʸ′*b + jˣ′+1, jʸ*b + jˣ+1] = (buff[jˣ⁻, jʸ⁻] + buff[jˣ⁻, jʸ⁺] + buff[jˣ⁺, jʸ⁻] + buff[jˣ⁺, jʸ⁺]) / √(ζˣ*ζˣ′*ζʸ*ζʸ′)
                        end
                    end
                end
            end
        end
    end
end

################ Sparse ################

function filter_count!(ft::FourierTransformerP; threshold::Real=0)
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

function fft_to_operator_sparse!(rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}, vals::AbstractVector{<:Number}, ft::FourierTransformerP)
    D = ndims(ft.buff)
    if D == 1
        fft_to_operator_sparse_1D!(rows, cols, vals, ft)
    elseif D == 2
        fft_to_operator_sparse_2D!(rows, cols, vals, ft)
    else
        error("fft_to_operator_sparse_$(D)D! not implemented.")
    end
end

######## 1D ########

"""
Set to zero values of `ft.buff` that are smaller by magnitude than `threshold`.
Based on the resulting number of nonzero elements in `ft.buff`, count and return the number of values that will be stored in the matrix indexed by (𝑗ˣ′, 𝑗ˣ).
"""
function filter_count_1D!(ft::FourierTransformerP; threshold::Real=0)
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
Use the result of the transform to construct a sparse matrix indexed by (𝑗ˣ′, 𝑗ˣ).
The type of `vals` might differ from the type of `ft.buff` since one may want to drop the imaginary part.
"""
function fft_to_operator_sparse_1D!(rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}, vals::AbstractVector{<:Number}, ft::FourierTransformerP)
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
Based on the resulting number of nonzero elements in `ft.buff`, count and return the number of values that will be stored in the matrix indexed by (𝑗ˣ′𝑗ʸ′, 𝑗ˣ𝑗ʸ).
"""
function filter_count_2D!(ft::FourierTransformerP; threshold::Real=0)
    (;M, buff) = ft
    n_elem = 0
    B = 2M + 1 # the size of each block

    # roughly, r controls the block-diagonal on which buff[r, c] will be placed, while c controls the diagonal inside all those blocks
    for r in axes(buff, 2), c in axes(buff, 1) # SWAPPED r and c according to swapping of x and y, but didn't rename, hence r/c ("row"/"column") have swapped names
        if abs(buff[c, r]) ≤ threshold # SWAPPED
            buff[c, r] = 0 # SWAPPED
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
Use the result of the transform to construct a sparse matrix indexed by (𝑗ˣ′𝑗ʸ′, 𝑗ˣ𝑗ʸ).
The type of `vals` might differ from the type of `ft.buff` since one may want to drop the imaginary part.
"""
function fft_to_operator_sparse_2D!(rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}, vals::AbstractVector{<:Number}, ft::FourierTransformerP)
    B = 2ft.M + 1 # the size of each block
    u = ft.buff

    counter = 1

    # SWAPPED r and c according to swapping of x and y, but didn't rename, hence r/c ("row"/"column") have swapped names
    for r_u in axes(u, 2), c_u in axes(u, 1) # iterate over columns and rows of `u`
        val = u[c_u, r_u] # SWAPPED 
        val == 0 && continue
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
# Based on results of a real 2D RFT `u`, return the matrix indexed by (𝑗ˣ′𝑗ʸ′, 𝑗ˣ𝑗ʸ).
# `make_real=true` will mutate `u`, taking the real parts of elements, which is useful if the original function is even and hence the transform is known to be real.
# This version is for dense matrices, but it is slower than [`fft_to_operator_2D!`](@ref); used only for testing purposes.
# """
# function _rfft_to_operator!(u::AbstractMatrix{<:Number}; make_real=false)
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
# Based on results of a 2D FFT `u`, return the matrix indexed by (𝑗ˣ′𝑗ʸ′, 𝑗ˣ𝑗ʸ).
# `make_real=true` will mutate `u`, taking the real parts of elements, which is useful if the original function is even and hence the transform is known to be real.
# For dense matrices, this is slower than [`fft_to_operator_naive`](@ref); used only for testing purposes.
# """
# function _fft_to_operator(u::AbstractMatrix; make_real=false)
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
