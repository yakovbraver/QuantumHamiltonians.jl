"""
A Jacobi/block-Jacobi preconditioner for `XSpaceHamiltonian`, used for linear solving during diagonalisation.
In the block-Jacobi case, "blocks" refer to the component blocks of the Hamiltonian, and the preconditioner is block-diagonal both in x- and p-space.
The p-space elements are reordered so that the preconditioner contains small `nc` × `nc` blocks. Each of them is LU-factorised and then its inverse is applied.
`Jₚ` is then an array of LU factorization blocks.
In the simple (non-block) case, `Jₚ` is just the diagonal of p-space Hamiltonian matrix.
`T` is the type of the object as a linear map -- basically same as the `T` parameter of `XSpaceHamiltonian`.
Currently 𝐴 is supported only in the cis case.
"""
struct JacobiPreconditioner{R, T, FourierTransformer, Jₚ_T}
    ft::FourierTransformer
    nc::Int
    B::Int
    Jₚ::Vector{Jₚ_T} # vector of scalars of type `T` in the simple case, and a vector of LU factorisation objects in the block-Jacobi case
    # buffers for applying the map, of size `nc*B`
    buff_real::Vector{R}
    buff_complex::Vector{Complex{R}}
    buff_complex2::Vector{Complex{R}}
end

Base.eltype(::Type{<:JacobiPreconditioner{R, T}}) where {R, T} = T
Base.size(prec::JacobiPreconditioner) = (prec.nc * prec.B, prec.nc * prec.B)

"Fill the constant-coefficient approximation at linearised momentum index `j`."
function fill_jacobi_block!(block::AbstractMatrix{T}, xh::XSpaceHamiltonian{R, T}, U_average::AbstractMatrix{T}, A_average::AbstractMatrix, j::Integer, shifts::AbstractVector{R}) where {R, T}
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
Construct a block-Jacobi (`type=:block`) or simple Jacobi (`type=:simple`) preconditioner for `xh`.
If a given component has no 𝑈 and no 𝐴, then the diagonal element is shifted by `shift`. If not passed (or zero is passed), then a scale-aware shift is used.
"""
function JacobiPreconditioner(xh::XSpaceHamiltonian{R, T}; shift::R=zero(R), type::Symbol=:block) where {R, T}
    (;nc, B, U, A, ∇, ∇², ft) = xh

    # Here `T` is the element type of `xh` in real-space. Meanwhile, we need the type of `xh` in p-space.
    # But the Laplacian is real in p-space, and the zeroth harmonic of 𝑈 will be of the same type as `U` (whose type is `T`). So the required type is just `T`.
    U_average = zeros(T, nc, nc)
    for c in axes(U, 1), b in axes(U, 2)
        !isempty(U[c, b]) && (U_average[c, b] = sum(U[c, b]) / B)
    end

    A_average = zeros(R, nc, length(∇)) # `𝐴` is assumed real in `xh`
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

    if type == :block # block-Jacobi: will construct nc × nc blocks for each mode
        Jₚ = map(1:B) do j # iterate over all momenta, where `j` is a linearised momentum index
            block = Matrix{T}(undef, nc, nc)
            fill_jacobi_block!(block, xh, U_average, A_average, j, shifts)
            LA.lu!(block; check=true) # this in-place verion will alias `block` -- this is why we create new `block` at each iteration
        end
    else # simple Jacobi: will construct the diagonal
        Jₚ = Vector{T}(undef, nc*B)
        for c in 1:nc
            window = (c-1)B+1:c*B
            @. Jₚ[window] = xh.∇² + shifts[c]
            U_average[c, c] != 0 && (Jₚ[window] .+= U_average[c, c])
            if xh.basis == :cis # 𝐴 is not supported for sin/cos
                for i in axes(A_average, 2)
                    A_average[c, i] != 0 && (@. Jₚ[window] -= 2A_average[c, i] * xh.∇[i])
                end
            end
        end
    end

    return JacobiPreconditioner{R, T, typeof(ft), eltype(Jₚ)}(ft, nc, B, Jₚ,
        Vector{R}(undef, nc*B), Vector{Complex{R}}(undef, nc*B), Vector{Complex{R}}(undef, nc*B))
end

"Apply inverse of preconditioner `prec` to an x-space wave function `f`."
@views function ldiv!(f′::AbstractVector, prec::JacobiPreconditioner{R, T}, f::AbstractVector) where {R, T}
    (;ft, nc, B, Jₚ) = prec
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
      
    if eltype(Jₚ) <: Number # simple Jacobi: just divide the transformed vector by the diagonal stored in `Jₚ`
         buff ./= Jₚ
         # references so that correct buffers are used for FFT
         Jₚ⁻¹fₚ = buff # p-space source
         J⁻¹f = buff2 # x-space destination
    else # block-Jacobi: apply the factorization object to every Fourier mode
        if fₚ_isreal && T <: Real # both `buff` and `T` are real, so can write buff in-place. `T` indicates the underlying type of `factorization`
            mode_buffer = similar(buff, nc) # `mode_buffer` is allocated every time. Could be a field of `prec`, but the type depends on `f`, so a real and a complex version would be needed
            Jₚ⁻¹fₚ = buff
        else
            mode_buffer = Vector{Complex{R}}(undef, nc)
            Jₚ⁻¹fₚ = buff2
        end
        J⁻¹f = buff
     
        for j in 1:B
            for c in 1:nc
                mode_buffer[c] = buff[(c-1)B + j]
            end
            ldiv!(Jₚ[j], mode_buffer) # we would like to use `buff[j:B:end]` instead of this `mode_buffer`, but `ldiv!` does not work on noncontiguos views
            for c in 1:nc
                Jₚ⁻¹fₚ[(c-1)B + j] = mode_buffer[c]
            end
        end
    end

    for c in 1:nc
        window = (c-1)B+1:c*B
        if f′_isreal && ft.basis == :cis
            # cannot write directly to `f′` because it is real. Write into `J⁻¹f`, which is complex in this case. `J⁻¹f` and Jₚ⁻¹fₚ reference different buffers
            transform!(J⁻¹f[window], ft, Jₚ⁻¹fₚ[window]; direction=:backward, normalise=true)
            @. f′[window] = real(J⁻¹f[window])
        else
            transform!(f′[window], ft, Jₚ⁻¹fₚ[window]; direction=:backward, normalise=true)
        end
    end
    return f′
end

"Apply inverse of preconditioner `prec` to an x-space wave function `f` in-place."
ldiv!(prec::JacobiPreconditioner, f::AbstractVector) = ldiv!(f, prec, f)