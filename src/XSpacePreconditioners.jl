"""
An FFT-space, constant-coefficient approximation of an `XSpaceHamiltonian`.

The spatially varying terms are replaced by their grid averages while the
kinetic term remains diagonal in momentum space. Each momentum mode is thus a
small `nc × nc` dense system. `ldiv!` applies the inverse approximation and is
used as a left preconditioner by iterative linear solvers.
"""
struct FourierBlockPreconditioner{R, T, FourierTransformer, Factorization}
    ft::FourierTransformer
    nc::Int
    B::Int
    factorizations::Vector{Factorization}
    # buffers for applying the map, of size `nc*B`
    buff_real::Vector{R}
    buff_complex::Vector{Complex{R}}
    buff_complex2::Vector{Complex{R}}
end

Base.eltype(::Type{<:FourierBlockPreconditioner{R, T}}) where {R, T} = T
Base.size(prec::FourierBlockPreconditioner) = (prec.nc * prec.B, prec.nc * prec.B)

"Fill the constant-coefficient approximation at linearised momentum index `j`."
function fill_fourier_block!(block::AbstractMatrix{T}, xh::XSpaceHamiltonian{R, T}, U_average::AbstractMatrix{T}, A_average::AbstractMatrix, j::Integer, shifts::AbstractVector{R}) where {R, T}
    copyto!(block, U_average)
    for c in axes(block, 1)
        block[c, c] += xh.∇²[j] + shifts[c]
        if xh.basis == :cis # 𝐴 is not supported for sin/cos
            for i in axes(A_average, 2)
                block[c, c] -= 2A_average[c, i] * xh.∇[i][j]
            end
        end
    end
    return block
end

"""
Construct a block-Jacobi preconditioner for `xh` by retaining the diagonal of each block of p-space `xh`.
If a given component has no 𝑈 and no 𝐴, then the diagonal element is shifted by `shift`. If not provided, then a scale-aware shift is used.
"""
function FourierBlockPreconditioner(xh::XSpaceHamiltonian{R, T}; shift::R=zero(R)) where {R, T}
    (;nc, B, U, A, ∇, ∇², ft) = xh

    # `T` is the element type of `xh` in real-space. Here we need the type for `xh` in p-space.
    # The Laplacian is real in p-space, while the zeroth harmonic of 𝑈 will be of the same type as `U`. So the required type is just `T`.
    U_average = zeros(T, nc, nc)
    for c in axes(U, 1), b in axes(U, 2)
        !isempty(U[c, b]) && (U_average[c, b] = sum(U[c, b]) / B)
    end

    A_average = zeros(R, nc, length(∇))
    for c in axes(A, 1), i in axes(A, 2)
        !isempty(A[c, i]) && (A_average[c, i] = sum(A[c, i]) / B)
    end

    # determine the `scale`, used if `shift` is needed but was not provided
    scale = max(one(R), maximum(abs, ∇²), maximum(abs, U_average))
    if xh.basis == :cis
        for i in axes(A_average, 2)
            scale += 2 * maximum(abs, @view(A_average[:, i])) * maximum(abs, ∇[i])
        end
    end
    # calculate the shifts for each component
    shifts = map(1:nc) do c 
        if isempty(U[c, c]) && !row_has_something(xh.𝐴, c) # no U or A, so must shift
            iszero(shift) ? √eps(R) * scale : R(shift) # is `shift` is not provided (or is zero), then use scale × ϵ
        else
            zero(R)
        end
    end

    factorizations = map(1:B) do j # iterate over all momenta, where `j` is a linearised momentum index
        block = Matrix{T}(undef, nc, nc)
        fill_fourier_block!(block, xh, U_average, A_average, j, shifts)
        LA.lu!(block; check=true) # this in-place verion will alias `block` -- this is why we create new `block` at each iteration
    end

    return FourierBlockPreconditioner{R, T, typeof(ft), eltype(factorizations)}(ft, nc, B, factorizations,
        Vector{R}(undef, nc*B), Vector{Complex{R}}(undef, nc*B), Vector{Complex{R}}(undef, nc*B))
end

@views function ldiv!(f′::AbstractVector, prec::FourierBlockPreconditioner{R, T}, f::AbstractVector) where {R, T}
    (;ft, nc, B, factorizations) = prec
    length(f) == nc * B || throw(DimensionMismatch("preconditioner input has length $(length(f)); expected $(nc * B)"))
    length(f′) == nc * B || throw(DimensionMismatch("preconditioner output has length $(length(f′)); expected $(nc * B)"))

    f_isreal  = eltype(f) <: Real
    f′_isreal = eltype(f′) <: Real

    fₚ_isreal = f_isreal && ft.basis != :cis
    buff = fₚ_isreal ? prec.buff_real : prec.buff_complex
    buff2 = prec.buff_complex2

    # transform `f` to p-space, by every component
    for c in 1:nc
        window = (c-1)B+1:c*B
        if ft.basis == :cis && f_isreal
            copyto!(buff2[window], f[window]) # `ft` can only act on complex vectors, so need to copy real `f` into a complex buffer
            transform!(buff[window], ft, buff2[window]; direction=:forward)
        else
            transform!(buff[window], ft, f[window]; direction=:forward)
        end
    end
    
    if fₚ_isreal && T <: Real # both `buff` and `T` are real, so can write buff in-place. `T` indicates the underlying type of `factorization`
        mode_buffer = similar(buff, nc)
        Jₚ⁻¹fₚ = buff
    else
        mode_buffer = Vector{Complex{R}}(undef, nc)
        Jₚ⁻¹fₚ = buff2
    end
       
    # apply Jₚ⁻¹ using the factorization object to every Fourier mode
    for j in 1:B
        for c in 1:nc
            mode_buffer[c] = buff[(c-1)B + j]
        end
        ldiv!(factorizations[j], mode_buffer) # we would like to use `buff[j:B:end]` instead of this `mode_buffer`, but `ldiv!` does not work on noncontiguos views
        for c in 1:nc
            Jₚ⁻¹fₚ[(c-1)B + j] = mode_buffer[c]
        end
    end

    for c in 1:nc
        window = (c-1)B+1:c*B
        if f′_isreal && ft.basis == :cis
            # cannot write directly to `f′` because it is real. Write into `buff`, which is complex in this case, while Jₚ⁻¹fₚ is aliased with `buff2`
            transform!(buff[window], ft, Jₚ⁻¹fₚ[window]; direction=:backward, normalise=true)
            @. f′[window] = real(buff[window])
        else
            transform!(f′[window], ft, Jₚ⁻¹fₚ[window]; direction=:backward, normalise=true)
        end
    end
    return f′
end

ldiv!(prec::FourierBlockPreconditioner, x::AbstractVector) = ldiv!(x, prec, x)

# """
# An FFT-space approximation of an `XSpaceHamiltonian` retaining only its
# diagonal Laplacian term. `ldiv!` applies the inverse independently to every
# component using a single elementwise multiplication in Fourier space.
# """
# mutable struct LaplacePreconditioner{R, T, FourierTransformer}
#     ft::FourierTransformer
#     nc::Int
#     B::Int
#     D⁻¹::Vector{T} # inverse diagonal, which contains Laplacian part + average potential
#     # buffers for applying the map, of size `nc*B`
#     buff_real::Vector{R}
#     buff_complex::Vector{Complex{R}}
#     buff_complex2::Vector{Complex{R}}
# end

# Base.eltype(::Type{<:LaplacePreconditioner{T}}) where {T} = T
# Base.size(prec::LaplacePreconditioner) = (prec.nc * prec.B, prec.nc * prec.B)

# """
#     LaplacePreconditioner(xh; shift=nothing)

# Construct a preconditioner retaining only the Fourier-diagonal Laplacian of
# `xh`. All potentials, couplings, gauge fields, and decay rates are ignored.
# When `shift` is `nothing`, a scale-aware positive diagonal shift is added to
# avoid singular zero-momentum modes.
# """
# function LaplacePreconditioner(xh::XSpaceHamiltonian{R, T}; shift=nothing) where {R, T}
#     (;nc, B, ∇², U, ft) = xh
#     if !isnothing(shift) && (!(shift isa Real) || !isfinite(shift) || shift < zero(shift))
#         throw(ArgumentError("preconditioner shift must be a nonnegative real number"))
#     end

#     # `T` is the element type of `xh` in real-space. Here we need the type for `xh` in p-space.
#     # The Laplacian is real in p-space, while the zeroth harmonic of 𝑈 will be of the same type as `U`. So the required type is just `T`.
#     U_average = zeros(T, nc)
#     for c in eachindex(U_average)
#         !isempty(U[c, c]) && (U_average[c] = sum(U[c, c]) / B)
#     end

#     scale = max(one(R), maximum(abs, ∇²), maximum(abs, U_average))
#     regularisation = isnothing(shift) ? √eps(R) * scale : R(shift)

#     D⁻¹ = 

#     return LaplacePreconditioner{T, WorkT, R, typeof(ft)}(ft, nc, B, D⁻¹, Vector{WorkT}(undef, nc * B), Vector{WorkT}(undef, nc * B))
# end

# @views function ldiv!(y::AbstractVector, prec::LaplacePreconditioner, x::AbstractVector)
#     (;ft, nc, B, D⁻¹, pspace_buffer, xspace_buffer) = prec
#     length(x) == nc * B || throw(DimensionMismatch("preconditioner input has length $(length(x)); expected $(nc * B)"))
#     length(y) == nc * B || throw(DimensionMismatch("preconditioner output has length $(length(y)); expected $(nc * B)"))

#     for c in 1:nc
#         window = (c - 1)B+1:c*B
#         if ft.basis == :cis && eltype(x) <: Real
#             copyto!(xspace_buffer[window], x[window])
#             transform!(pspace_buffer[window], ft, xspace_buffer[window]; direction=:forward)
#         else
#             transform!(pspace_buffer[window], ft, x[window]; direction=:forward)
#         end
#     end

#     for c in 1:nc
#         window = (c - 1)B+1:c*B
#         pspace_buffer[window] .*= D⁻¹
#     end

#     for c in 1:nc
#         window = (c - 1)B+1:c*B
#         transform!(xspace_buffer[window], ft, pspace_buffer[window]; direction=:backward, normalise=true)
#         if eltype(y) <: Real
#             @. y[window] = real(xspace_buffer[window])
#         else
#             copyto!(y[window], xspace_buffer[window])
#         end
#     end
#     return y
# end

# ldiv!(prec::LaplacePreconditioner, x::AbstractVector) = ldiv!(x, prec, x)
