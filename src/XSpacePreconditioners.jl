"""
An FFT-space, constant-coefficient approximation of an `XSpaceHamiltonian`.

The spatially varying terms are replaced by their grid averages while the
kinetic term remains diagonal in momentum space. Each momentum mode is thus a
small `nc × nc` dense system. `ldiv!` applies the inverse approximation and is
used as a left preconditioner by iterative linear solvers.
"""
mutable struct FourierBlockPreconditioner{T, WorkT, F, Factorization}
    ft::F
    nc::Int
    B::Int
    factorizations::Vector{Factorization}
    pspace_buffer::Vector{WorkT}
    xspace_buffer::Vector{WorkT}
    mode_buffer::Vector{WorkT}
end

Base.eltype(::Type{<:FourierBlockPreconditioner{T}}) where {T} = T
Base.size(prec::FourierBlockPreconditioner) = (prec.nc * prec.B, prec.nc * prec.B)

"Fill the constant-coefficient approximation at momentum index `k`."
function fill_fourier_block!(block::AbstractMatrix{T}, xh::XSpaceHamiltonian,
                             U_average::AbstractMatrix{T}, A_average::AbstractMatrix,
                             k::Integer, shift) where {T}
    copyto!(block, U_average)
    for c in axes(block, 1)
        block[c, c] += xh.∇²[k] + shift
        if xh.basis == :cis
            for i in axes(A_average, 2)
                block[c, c] -= 2A_average[c, i] * xh.∇[i][k]
            end
        end
    end
    return block
end

"""
    FourierBlockPreconditioner(xh; shift=nothing)

Construct a preconditioner for `xh` by retaining its exact momentum-space
kinetic term and replacing every coordinate-space field by its grid average.
When `shift` is `nothing`, a scale-aware positive diagonal shift is added to
the approximation to avoid singular momentum blocks. Pass `shift=zero(R)` to
disable that regularisation explicitly.
"""
function FourierBlockPreconditioner(xh::XSpaceHamiltonian{R, T}; shift=nothing) where {R, T}
    (;nc, B, U, A, ∇, ∇², ft) = xh
    if !isnothing(shift) && (!(shift isa Real) || !isfinite(shift) || shift < zero(shift))
        throw(ArgumentError("preconditioner shift must be a nonnegative real number"))
    end

    # A periodic Hamiltonian can be real, but its Fourier coefficients are still complex.
    WorkT = xh.basis == :cis ? Complex{R} : T
    U_average = zeros(WorkT, nc, nc)
    for c in axes(U, 1), b in axes(U, 2)
        !isempty(U[c, b]) && (U_average[c, b] = sum(U[c, b]) / B)
    end

    A_average = zeros(R, nc, length(∇))
    for c in axes(A, 1), i in axes(A, 2)
        !isempty(A[c, i]) && (A_average[c, i] = sum(A[c, i]) / B)
    end

    scale = max(one(R), maximum(abs, ∇²), maximum(abs, U_average))
    if xh.basis == :cis
        for i in axes(A_average, 2)
            scale += 2 * maximum(abs, @view(A_average[:, i])) * maximum(abs, ∇[i])
        end
    end
    regularisation = isnothing(shift) ? √eps(R) * scale : R(shift)

    first_block = Matrix{WorkT}(undef, nc, nc)
    fill_fourier_block!(first_block, xh, U_average, A_average, 1, regularisation)
    first_factorization = LA.lu!(first_block; check=true)
    factorizations = Vector{typeof(first_factorization)}(undef, B)
    factorizations[1] = first_factorization

    for k in 2:B
        block = Matrix{WorkT}(undef, nc, nc)
        fill_fourier_block!(block, xh, U_average, A_average, k, regularisation)
        factorizations[k] = LA.lu!(block; check=true)
    end

    return FourierBlockPreconditioner{T, WorkT, typeof(ft), typeof(first_factorization)}(
        ft, nc, B, factorizations,
        Vector{WorkT}(undef, nc * B), Vector{WorkT}(undef, nc * B), Vector{WorkT}(undef, nc),
    )
end

function ldiv!(y::AbstractVector, prec::FourierBlockPreconditioner, x::AbstractVector)
    (;ft, nc, B, factorizations, pspace_buffer, xspace_buffer, mode_buffer) = prec
    length(x) == nc * B || throw(DimensionMismatch("preconditioner input has length $(length(x)); expected $(nc * B)"))
    length(y) == nc * B || throw(DimensionMismatch("preconditioner output has length $(length(y)); expected $(nc * B)"))

    for c in 1:nc
        window = (c - 1)B+1:c*B
        if ft.basis == :cis && eltype(x) <: Real
            copyto!(@view(xspace_buffer[window]), @view(x[window]))
            transform!(@view(pspace_buffer[window]), ft, @view(xspace_buffer[window]); direction=:forward)
        else
            transform!(@view(pspace_buffer[window]), ft, @view(x[window]); direction=:forward)
        end
    end

    for k in 1:B
        for c in 1:nc
            mode_buffer[c] = pspace_buffer[(c - 1)B + k]
        end
        ldiv!(factorizations[k], mode_buffer)
        for c in 1:nc
            pspace_buffer[(c - 1)B + k] = mode_buffer[c]
        end
    end

    for c in 1:nc
        window = (c - 1)B+1:c*B
        transform!(@view(xspace_buffer[window]), ft, @view(pspace_buffer[window]); direction=:backward, normalise=true)
        if eltype(y) <: Real
            @. y[window] = real(xspace_buffer[window])
        else
            copyto!(@view(y[window]), @view(xspace_buffer[window]))
        end
    end
    return y
end

ldiv!(prec::FourierBlockPreconditioner, x::AbstractVector) = ldiv!(x, prec, x)
