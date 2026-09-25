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

"Fill the preconditioner block at momentum index `j`."
function fill_jacobi_block!(block::AbstractMatrix{T}, xh::XSpaceHamiltonian{R, T}, U_avg::AbstractMatrix{T}, A_avg::AbstractMatrix, j::Integer, shifts::AbstractVector{R}) where {R, T}
    copyto!(block, U_avg)
    for c in axes(block, 1)
        block[c, c] += xh.∇²[j] + shifts[c]
        if xh.basis == :cis # 𝐴 is not supported for sin/cos
            for i in axes(A_avg, 2)
                block[c, c] -= 2A_avg[c, i] * xh.∇[i][j]
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
    U_avg = zeros(T, nc, nc)
    for c in axes(U, 1), b in axes(U, 2)
        !isempty(U[c, b]) && (U_avg[c, b] = sum(U[c, b]) / B)
    end

    A_avg = zeros(R, nc, length(∇)) # `𝐴` is assumed real in `xh`
    for c in axes(A, 1), i in axes(A, 2)
        !isempty(A[c, i]) && (A_avg[c, i] = sum(A[c, i]) / B)
    end

    # determine the `scale`, used if `shift` is needed but was not provided
    scale = max(one(R), maximum(abs, ∇²), maximum(abs, U_avg))
    if xh.basis == :cis
        for i in axes(A_avg, 2)
            scale += 2 * maximum(abs, @view(A_avg[:, i])) * maximum(abs, ∇[i])
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
            fill_jacobi_block!(block, xh, U_avg, A_avg, j, shifts)
            lu!(block; check=true) # this in-place verion will alias `block` -- this is why we create new `block` at each iteration
        end
    else # simple Jacobi: will construct the diagonal
        Jₚ = Vector{T}(undef, nc*B)
        for c in 1:nc
            window = (c-1)B+1:c*B
            @. Jₚ[window] = xh.∇² + shifts[c]
            U_avg[c, c] != 0 && (Jₚ[window] .+= U_avg[c, c])
            if xh.basis == :cis # 𝐴 is not supported for sin/cos
                for i in axes(A_avg, 2)
                    A_avg[c, i] != 0 && (@. Jₚ[window] -= 2A_avg[c, i] * xh.∇[i])
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

"""
A block-Jacobi preconditioner for `XSpaceHamiltonian`, used for linear solving during GPE stationary state search using nonlinear solve.
The preconditioner depends on the state being optimised, so the obejct is updated on each nonlinear solve iteration using `update!(prec, u)`.
`T` is the type of the object as a linear map -- basically same as the `T` parameter of `XSpaceHamiltonian`.
Currently 𝐴 is supported only in the cis case.
"""
mutable struct GPEJacobiPreconditioner{R, T, FourierTransformer, Jₚ_T}
    nc::Int
    nc_physical::Int
    B::Int
    searchreal::Bool
    N_isfixed::Bool # true if total number of atoms is fixed
    μs::Vector{R} # fixed 𝜇s of each component. Will contain zeros if 𝜇s are not fixed
    g::Matrix{R}
    Jₚ::Vector{Jₚ_T}
    U_avg::Matrix{T}
    A_avg::Matrix{R}
    ∇::Vector{Vector{R}}
    ∇²::Vector{R}
    ft::FourierTransformer
    buff_real::Vector{R}
    buff_complex::Vector{Complex{R}}
    buff_complex2::Vector{Complex{R}}
    mode_buffer::Vector{Complex{R}}
end

# Two methods to avoid error when printing nonlinear solving trace using `show_trace=Val(true)`.
# NonlinearSolve's trace formatter recursively walks struct fields when it prints the algorithm, entering the linear solver object, then `GPEJacobiPreconditioner`, then `ft`, and breaks trying to pring the FFTW plans.
# So just print the generic name, in fact we don't care at all.
NLS.NonlinearSolveBase.Utils.clean_sprint_struct(::GPEJacobiPreconditioner) = "GPEJacobiPreconditioner()"
NLS.NonlinearSolveBase.Utils.clean_sprint_struct(::GPEJacobiPreconditioner, ::Int) = "GPEJacobiPreconditioner()"

Base.eltype(::Type{<:GPEJacobiPreconditioner{R}}) where R = R

"Construct a `GPEJacobiPreconditioner` object. If 𝜇s are fixed, pass them as `μs`, otherwise do not pass."
function GPEJacobiPreconditioner(xh::XSpaceHamiltonian{R, T}, g::AbstractMatrix{R}, nc_effective::Integer;
                                 searchreal::Bool=false, μs::AbstractVector{R}=zeros(R, xh.nc), N_isfixed::Bool=false) where {R, T}
    nc_effective == (searchreal ? xh.nc : 2xh.nc) || throw(DimensionMismatch("effective component count does not match searchreal"))
    size(g) == (xh.nc, xh.nc) || throw(DimensionMismatch("g must have size ($(xh.nc), $(xh.nc))"))
    length(μs) == xh.nc || throw(DimensionMismatch("μs has the wrong number of physical components"))

    U_avg = zeros(T, xh.nc, xh.nc)
    for c in axes(xh.U, 1), b in axes(xh.U, 2)
        !isempty(xh.U[c, b]) && (U_avg[c, b] = sum(xh.U[c, b]) / xh.B)
    end

    A_avg = zeros(R, xh.nc, length(xh.∇))
    for c in axes(xh.A, 1), i in axes(xh.A, 2)
        !isempty(xh.A[c, i]) && (A_avg[c, i] = sum(xh.A[c, i]) / xh.B)
    end

    # perform LU for identity matrices just to initialise with the correct type
    Jₚ = [lu!(Matrix{R}(LA.I, nc_effective, nc_effective); check=true) for _ in 1:xh.B]

    wf_length = nc_effective * xh.B # length of the wave function vector of a single component, which is doubled in the complex case
    prec = GPEJacobiPreconditioner{R, T, typeof(xh.ft), eltype(Jₚ)}(
        nc_effective, xh.nc, xh.B, searchreal, N_isfixed, μs, g, Jₚ, U_avg,
        A_avg, xh.∇, xh.∇², xh.ft,
        Vector{R}(undef, wf_length), Vector{Complex{R}}(undef, wf_length),
        Vector{Complex{R}}(undef, wf_length), Vector{Complex{R}}(undef, nc_effective))

    u_length = wf_length + (iszero(μs) ? (N_isfixed ? 1 : nc_effective) : 0) # include unknown chemical potentials when present
    update!(prec, zeros(R, u_length))

    return prec
end

"Update the preconditioner using the nonlinear iteration state `f`."
function update!(prec::GPEJacobiPreconditioner{R, T}, u::AbstractVector) where {R, T}
    (;B, nc, nc_physical, g, ∇, ∇², searchreal) = prec

    μs = !iszero(prec.μs) ? prec.μs :
         prec.N_isfixed ? fill(u[end], nc_physical) :
         searchreal ? @view(u[end-nc+1:end]) : @view(u[end-nc+1:2:end]) # take every second element because every μ is duplicated in the per-component-N case

    u²_avg = zeros(R, nc_physical)
    uᶜuᵈ_avg = zeros(R, 2, 2, nc_physical, nc_physical) # when !searchreal, the first two dimensions enumerate real and imaginary parts. When searchreal, then we only use the (1, 1, :, :) slice
    if searchreal
        for c in 1:nc_physical
            uc = @view u[(c-1)*B+1:c*B]
            u²_avg[c] = sum(abs2, uc) / B
            uᶜuᵈ_avg[1, 1, c, c] = u²_avg[c]
        end
    else
        for c in 1:nc_physical
            uᶜ_real = @view u[(2c-2)*B+1:(2c-1)*B]
            uᶜ_imag = @view u[(2c-1)*B+1:2c*B]
            u²_avg[c] = (sum(abs2, uᶜ_real) + sum(abs2, uᶜ_imag)) / B
            for d in 1:nc_physical
                uᵈ_real = @view u[(2d-2)*B+1:(2d-1)*B]
                uᵈ_imag = @view u[(2d-1)*B+1:2d*B]
                uᶜuᵈ_avg[1, 1, c, d] = sum(uᶜ_real .* uᵈ_real) / B
                uᶜuᵈ_avg[1, 2, c, d] = sum(uᶜ_real .* uᵈ_imag) / B
                uᶜuᵈ_avg[2, 1, c, d] = uᶜuᵈ_avg[1, 2, c, d]
                uᶜuᵈ_avg[2, 2, c, d] = sum(uᶜ_imag .* uᵈ_imag) / B
            end
        end
    end
    gu²_μ = g * u²_avg - μs # a vector whose 𝑖th element is ∑ⱼ 𝑔ᵢⱼ𝑢ⱼ² - 𝜇ᵢ

    # create and LU-decompose the blocks for each momentum mode
    for j in 1:B
        block = zeros(R, nc, nc)
        for c in 1:nc_physical
            if searchreal
                block[c, c] += ∇²[j] + gu²_μ[c] + 2g[c, c] * u²_avg[c]
            else
                cr, ci = 2c - 1, 2c
                block[cr, cr] += ∇²[j] + gu²_μ[c]
                block[ci, ci] += ∇²[j] + gu²_μ[c]
            end
            for d in 1:nc_physical
                h = prec.U_avg[c, d]
                if searchreal
                    block[c, d] += real(h)
                    c != d && (block[c, d] += 2g[c, d] * uᶜuᵈ_avg[1, 1, c, d])
                else
                    cr, ci = 2c - 1, 2c
                    dr, di = 2d - 1, 2d
                    block[cr, dr] +=  real(h) + 2g[c, d] * uᶜuᵈ_avg[1, 1, c, d]
                    block[cr, di] += -imag(h) + 2g[c, d] * uᶜuᵈ_avg[1, 2, c, d]
                    block[ci, dr] +=  imag(h) + 2g[c, d] * uᶜuᵈ_avg[2, 1, c, d]
                    block[ci, di] +=  real(h) + 2g[c, d] * uᶜuᵈ_avg[2, 2, c, d]
                end
            end
        end
        for i in axes(prec.A_avg, 2), c in axes(prec.A_avg, 1)
            A∇ = 2prec.A_avg[c, i] * ∇[i][j]
            if searchreal
                block[c, c] -= A∇
            else
                block[2c-1, 2c-1] -= A∇
                block[2c, 2c] -= A∇
            end
        end
        prec.Jₚ[j] = lu!(block; check=true)
    end

    return prec
end

"Apply inverse of preconditioner `prec` to the nonlinear iteration state `f`."
@views function ldiv!(f′::AbstractVector, prec::GPEJacobiPreconditioner{R}, f::AbstractVector) where R
    f_isreal = eltype(f) <: Real
    buff = f_isreal && prec.ft.basis != :cis ? prec.buff_real : prec.buff_complex
    for c in 1:prec.nc
        window = (c-1)*prec.B+1:c*prec.B
        if prec.ft.basis == :cis && f_isreal
            copyto!(prec.buff_complex2[window], f[window])
            transform!(buff[window], prec.ft, prec.buff_complex2[window]; direction=:forward)
        else
            transform!(buff[window], prec.ft, f[window]; direction=:forward)
        end
    end
    for j in 1:prec.B
        for c in 1:prec.nc
            prec.mode_buffer[c] = buff[(c-1)*prec.B+j]
        end
        ldiv!(prec.Jₚ[j], prec.mode_buffer)
        for c in 1:prec.nc
            buff[(c-1)*prec.B+j] = prec.mode_buffer[c]
        end
    end
    for c in 1:prec.nc
        window = (c-1)*prec.B+1:c*prec.B
        if f_isreal && prec.ft.basis == :cis
            transform!(prec.buff_complex2[window], prec.ft, buff[window]; direction=:backward, normalise=true)
            @. f′[window] = real(prec.buff_complex2[window])
        else
            transform!(f′[window], prec.ft, buff[window]; direction=:backward, normalise=true)
        end
    end
    if iszero(prec.μs)
        nμ = prec.N_isfixed ? 1 : prec.nc
        copyto!(f′, length(f)-nμ+1, f, length(f)-nμ+1, nμ)
    end
    return f′
end

ldiv!(prec::GPEJacobiPreconditioner, f::AbstractVector) = ldiv!(f, prec, f)
