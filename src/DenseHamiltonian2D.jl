"""
A type representing a spatial [𝑟 = (𝑥, 𝑦)], 𝑛-component, possibly quasimomentum-dependent Hamiltonian (𝐻ᵢⱼ)
    𝐻ᵢᵢ(𝑟) = [-i𝛿∇ + 𝑞 - 𝐴(𝑟)]² + 𝑈ᵢᵢ(𝑟)
    𝐻ᵢⱼ(𝑟) = 𝑈ᵢⱼ(𝑟)
as a dense matrix.
"""
# TODO reanme {R, T, S} -> {Tr, Th, Te} for "type real", "type Hamiltonian", "type eigenvalues".
mutable struct DenseHamiltonian2D{R<:Real,T<:Number,S<:Number} <: XSpaceHamiltonian2D{:dense} # in practice `T` shoudld be `R` or `Complex{R}` (and same for `S`) -- always check this. If this is not the case, probably your 𝑈 or 𝐴 do not return R's.
    xlims::Tuple{R, R}
    ylims::Tuple{R, R}
    Lx::R # length along 𝑥
    Ly::R # length along 𝑦
    M::Int # maximum harmonic number (will use -M:M for periodic, 1:M for nonperiodic)
    δ::R # coefficient of the momentum term: -iδ∇
    nc::Int # number of components
    isperiodic::Bool
    ishermitian::Bool # `H` is nonhermitian if decays Γ are present
    𝑈::Matrix{<:Union{Function,Nothing}} # nc-component matrix containing coordinate-space potentials and couplings
    𝑈_iseven::BitMatrix # nc-component matrix indicating if 𝑈ᵢⱼ is an even function 𝑈ᵢⱼ(𝑥, 𝑦) = 𝑈ᵢⱼ(-𝑥, -𝑦)
    𝐴_x::Union{Function,Nothing}
    𝐴_y::Union{Function,Nothing}
    Γ::Vector{R} # decay rates
    H::Matrix{T} # momentum-space Hamiltonian used for diagonalisation
    ε::Vector{S} # eigenvalues, can be complex for nonhermitian `H`, hence additional type `S`
    V::Matrix{T} # eigenvectors matrix
    ε_q::Array{S,3} # ε_q[n, iqx, iqy] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    V_q::Array{T,4} # V_q[:, n, iqx, iqy] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)
end

"""
Construct a `DenseHamiltonian2D` object using the coordinate-space functions stored in `𝑈`, decay rates `Γ`, and gauge field (same for all components) 𝐴_x, 𝐴_y.
`M` is the maximum harmonic number. In the periodic case, the Hamiltonian will be `nc*(2M+1)²`-by-`nc*(2M+1)²` where `nc` is the number of components.
In nonperiodic case, the size will be `nc*M²`-by-`nc*M²`.
`𝑈_iseven[i, j]` matters only if `isperiodic=true` and shows whether `𝑈[i, j]` is an even function (i.e. whether 𝑢(𝑥, 𝑦) = 𝑢(-𝑥, -𝑦)). If it is, then Fourier transform is real, which is used for better accuracy.
If *all* functions are even (and real), then the resulting Fourier-space Hamiltonian is real (provided also there is no 𝐴 and Γ), giving a speed-up and better accuracy (compared to complex diagonalisation).
If `𝑈[i, j] === nothing` or it is complex, then the value of `𝑈_iseven[i, j]` does not matter.
"""
function DenseHamiltonian2D(𝑈::AbstractMatrix{<:Union{Function,Nothing}}, xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                            𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1)),
                            𝐴_x::Union{Function,Nothing}=nothing, 𝐴_y::Union{Function,Nothing}=nothing) where R <: Real
    Lx, Ly = xlims[2]-xlims[1], ylims[2]-ylims[1]
    
    PI = R(π) # π of the working type to prevent widening

    nc = size(𝑈, 1) # number of components

    # `isreal` will show if the resulting `H` will be real
    isreal = all( 𝑢(xlims[1], ylims[1]) isa Real for 𝑢 in 𝑈 if !isnothing(𝑢)) & # check if all functions in 𝑈 are real
             isnothing(𝐴_x) & isnothing(𝐴_y) & iszero(Γ)
    if isperiodic # for periodic potential, also check if functions are even 
        isreal &= all(𝑈_iseven[𝑈 .!== nothing])
    end

    B = isperiodic ? (2M+1)^2 : M^2 # size of each Hamiltonian block

    T = isreal ? R : Complex{R} # type of elements of the Hamiltonian
    H = zeros(T, nc*B, nc*B)

    if isperiodic
        N = 4M + 1 # number of points for FFT. This will yield harmonics from -2M to 2M
        dx, dy = Lx/N, Ly/N
        xs = range(xlims[1], xlims[2]-dx, N)
        ys = range(ylims[1], ylims[2]-dy, N)

        fft_buff = Matrix{Complex{R}}(undef, N, N) # a buffer for all (in-place) FFTs
        F = FFTW.plan_fft!(fft_buff) # the savings of rfft are negligible, and the output is much less convenient to handle in `fft_to_matrix`, so using fft. Also, this way we can do FFT in-place

        # iterate over `𝑈` and populate `H`
        for jH in axes(𝑈, 2)
            for iH in 1:jH # only upper triangle is scanned. The lower triangle is filled only if Γ is present
                wi = (iH-1)*B+1:iH*B
                wj = (jH-1)*B+1:jH*B

                𝑢 = 𝑈[iH, jH]

                # calculate and store FFT of 𝑢
                if !isnothing(𝑢)
                    𝑢_isrealeven = (𝑢(xlims[1], ylims[1]) isa Real) & 𝑈_iseven[iH, jH]
                    fft_buff .= 𝑢.(xs, ys')
                    F * fft_buff # in-place FFT, weird syntax
                    fft_buff ./= N^2
                    H[wi, wj] .= fft_to_matrix_naive!(fft_buff, make_real=𝑢_isrealeven)
                end

                # for diagonal block, add Laplacian, Γ, and 𝐴
                if iH == jH
                    if Γ[iH] != 0
                        H[diagind(H)[wi]] .-= im*Γ[iH]/2
                    end
                    # if there is no 𝐴, then add Laplacian. Otherwise it will be added together with 𝐴 components
                    if isnothing(𝐴_x) && isnothing(𝐴_y)
                        H[wi, wj] += Diagonal([(2PI*δ)^2 * ((jx/Lx)^2 + (jy/Ly)^2) for jx in -M:M for jy in -M:M]) # this is -δ²Δ
                    else
                        if !isnothing(𝐴_x)
                            fft_buff .= 𝐴_x.(xs, ys')
                            F * fft_buff
                            fft_buff ./= N^2
                            A_i = fft_to_matrix_naive!(fft_buff)
                            ∂_i = Diagonal([2PI * δ * jx/Lx for jx in -M:M for jy in -M:M]) # this is -iδ∂ₓ
                            H[wi, wj] .+= (∂_i - A_i)^2
                            # if there is no 𝐴𝑦, then add ∂𝑦². Otherwise it will be added together with 𝐴𝑦 in the next `if` clause
                            isnothing(𝐴_y) && (H[wi, wj] += Diagonal([(2PI * δ * jy/Ly)^2 for jx in -M:M for jy in -M:M]))
                        end
                        if !isnothing(𝐴_y)
                            fft_buff .= 𝐴_y.(xs, ys')
                            F * fft_buff
                            fft_buff ./= N^2
                            A_i = fft_to_matrix_naive!(fft_buff)
                            ∂_i = Diagonal([2PI * δ * jy/Ly for jx in -M:M for jy in -M:M]) # this is -iδ∂y
                            H[wi, wj] .+= (∂_i - A_i)^2
                            # if there is no 𝐴ₓ, then add ∂ₓ². Otherwise it was added together with 𝐴ₓ in the preceding `if` clause
                            isnothing(𝐴_x) && (H[wi, wj] += Diagonal([(2PI * δ * jx/Lx)^2 for jx in -M:M for jy in -M:M]))
                        end
                    end
                elseif !all(iszero, Γ) # fill conjugate block if Γ is present (then we cannot use Hermitian view)
                    H[wj, wi] .= @view(H[wi, wj])'
                end
            end
        end
    else # non-periodic
        N = 2M + 1
        xs = range(xlims[1], xlims[2], N)
        ys = range(ylims[1], ylims[2], N)
        dx, dy = xs[2]-xs[1], ys[2]-ys[1]

        fft_buff = Matrix{R}(undef, N, N)
        fft_buff_im = isreal ? R[;;] : Matrix{R}(undef, N, N) # if `isreal`, then this buffer is not needed; make it 0x0
        F = FFTW.plan_r2r!(fft_buff, FFTW.REDFT00)

        # iterate over `𝑈` and populate `H`
        for jH in axes(𝑈, 2)
            for iH in 1:jH # only upper triangle is scanned. The lower triangle is filled only if Γ is present
                wi = (iH-1)*B+1:iH*B
                wj = (jH-1)*B+1:jH*B

                𝑢 = 𝑈[iH, jH]

                # calculate and store FFT of 𝑢
                if !isnothing(𝑢)
                    if 𝑢(xs[1], ys[1]) isa Real
                        fft_buff .= 𝑢.(xs, ys')
                        (F * fft_buff)
                        fft_buff ./= (N-1)^2
                        H[wi, wj] .= dct_to_matrix(fft_buff)
                    else
                        # here `𝑢` is a complex function, but `FFTW.REDFT00` can only handle real ones. So we transform Re and Im separately.
                        fft_buff, fft_buff_im = reim.(𝑢.(xs, ys'))

                        F * fft_buff
                        fft_buff ./= (N-1)^2
                        H[wi, wj] .= dct_to_matrix(fft_buff)

                        F * fft_buff_im
                        fft_buff_im ./= (N-1)^2
                        H[wi, wj] .+= im .* dct_to_matrix(fft_buff_im)
                    end
                end

                if iH == jH # for a diagonal block, add the laplace term, optionally Γ, and the 𝐴's
                    # TODO: just subtract from diagonal
                    H[wi, wj] += Diagonal([(PI*δ)^2 * ((jx/Lx)^2 + (jy/Ly)^2) for jx in 1:M for jy in 1:M]) # add -δ²Δ
                    if Γ[iH] != 0
                        H[diagind(H)[wi]] .-= im*Γ[iH]/2
                    end

                    if !isnothing(𝐴_x)
                        fft_buff .= 𝐴_x.(xs, ys')
                        F * fft_buff
                        fft_buff ./= (N-1)^2
                        A_i = dct_to_matrix(fft_buff)
                        ∂_i = make_∂_x(M, Lx)
                        H[wi, wj] .+= im*(A_i*∂_i + ∂_i*A_i) + A_i^2 # The perfect square for `(∂_x - A_x)^2` is much less accurate
                    end
                    if !isnothing(𝐴_y)
                        fft_buff .= 𝐴_y.(xs, ys')
                        F * fft_buff
                        fft_buff ./= (N-1)^2
                        A_i = dct_to_matrix(fft_buff)
                        ∂_i = make_∂_y(M, Ly)
                        H[wi, wj] .+= im*(A_i*∂_i + ∂_i*A_i) + A_i^2 # The perfect square for `(∂_y - A_y)^2` is much less accurate
                    end
                elseif !all(iszero, Γ) # fill conjugate block if Γ is present (then we cannot use Hermitian view when diagonalising)
                    H[wj, wi] .= @view(H[wi, wj])'
                end
            end
        end
    end
    
    # determine the type of eigenvalues 
    ishermitian = iszero(Γ) # if all `Γ`s are zeros, then Hamiltonian is Hermitian and the eigenvalues real
    S = ishermitian ? R : Complex{R} # type of eigenvalues
    return DenseHamiltonian2D(xlims, ylims, Lx, Ly, M, δ, nc, isperiodic, ishermitian, 𝑈, BitMatrix(𝑈_iseven), 𝐴_x, 𝐴_y, Γ, H, S[], T[;;], S[;;;], T[;;;;])
end

# """
# More efficient way of calculating
#     H = -Δ + im*(A_x*∂_x + A_y*∂_y + ∂_x*A_x + ∂_y*A_y) + A_x^2 + A_y^2 + U
# Currently intended for non-periodic calculation.
# """
# function sum_parts(A_x::Matrix{<:Real}, A_y::Matrix{<:Real}, ∂_y::Matrix{<:Real}, ∂_x::Matrix{<:Real}, U::Matrix{<:Real}, Δ)
#     H = complex.(U)
#     H += -Δ
#     unit = one(eltype(A_x))
#     null = zero(eltype(A_x))
#     symm!('L', 'L', unit, A_x, ∂_x, null, U) # we start using `U` as a buffer
#     symm!('R', 'L', unit, A_x, ∂_x, unit, U)
#     symm!('L', 'L', unit, A_y, ∂_y, unit, U)
#     symm!('R', 'L', unit, A_y, ∂_y, unit, U)
#     H .+= U .* im
#     symm!('L', 'L', unit, A_x, A_x, null, U)
#     H += U
#     symm!('L', 'L', unit, A_y, A_y, null, U)
#     H += U
#     return H
# end

"""
Construct from the result of 2D FFT `u` a matrix `U` indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
In a preliminary step, `u` is `fftshift`'ed.
`make_real=true` will mutate `u`, taking the real parts of elements, which is useful if the original function is even and hence the transform is known to be real.
We call it "naive" because `U` is allocated and then we sipmly go over each element, assigning an appropriate element of `u`.
In the dense case it is preferred over (since it's faster than) [`fft_to_matrix_sparse!`](@ref)
because even if `u[i, j]=0`, the corresponding elements of `U` still must be accessed to be set to zero.
"""
function fft_to_matrix_naive!(u::Matrix{T}; make_real::Bool=false) where T <: Number
    N = size(u, 1) # number of points used for FFT
    M = (N-1) ÷ 4 # maximum harmonic number (recall that N = 4M + 1 in the constructor)
    B = 2M + 1
    U = Matrix{make_real ? real(T) : T}(undef, B^2, B^2)
    make_real && (u .= real.(u))
    u = FFTW.fftshift(u) # indexing into `u` is more convenient if we shift
    @floop for jx in 1:B
        for jy in 1:B, j′x in 1:B, j′y in 1:B
            j₋x = j′x - jx
            j₋y = j′y - jy
            U[(j′x-1)B+j′y, (jx-1)B+jy] = u[j₋x+B, j₋y+B]
        end
    end
    return U
end

"""
Based on results of 2D DCT `u`, return the matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
"""
function dct_to_matrix(u::Matrix{T}) where T <: Number
    N = size(u, 1) # number of points used for FFT
    M = (N-1) ÷ 2 # maximum harmonic number
    H = Matrix{T}(undef, M^2, M^2)
    @floop for jx in 1:M
        for jy in 1:M, j′x in 1:M, j′y in 1:M
            j₋x = abs(j′x-jx)
            j₋y = abs(j′y-jy)
            H[(j′x-1)M+j′y, (jx-1)M+jy] = (u[j₋x+1, j₋y+1] - u[j₋x+1, j′y+jy+1] - u[j′x+jx+1, j₋y+1] + u[j′x+jx+1, j′y+jy+1]) / 4
        end
    end
    return H
end

"Return a 𝜕ₓ matrix in the sine basis."
function make_∂_x(M, Lx)
    ∂_x = zeros(typeof(Lx), M^2, M^2)
    @floop for jx in 1:M
        for j′x in 1+isodd(jx):2:M, jy in 1:M
            ∂_x[(j′x-1)M+jy, (jx-1)M+jy] = 4j′x*jx/(Lx * (j′x^2 - jx^2))
        end
    end
    return ∂_x
end

"Return a 𝜕y matrix in the sine basis."
function make_∂_y(M, Ly)
    ∂_y = zeros(typeof(Ly), M^2, M^2)
    @floop for jx in 1:M
        for jy in 1:M, j′y in 1+isodd(jy):2:M
            ∂_y[(jx-1)M+j′y, (jx-1)M+jy] = 4j′y*jy/(Ly * (j′y^2 - jy^2))
        end
    end
    return ∂_y
end

"""
Construct the coordinate-space wave function `ψ` of eigenstate `stateno` on a grid having `nx` points in `x` and `ny` points in `y` direction.
Return (`xs`, `ys`, `ψ`). If `qx` and `qy` are passed, then construct `ψ` at the corresponding quasimomenta.
"""
function make_eigenfunction(xh::XSpaceHamiltonian2D, stateno::Integer, nx::Integer, ny::Integer, iqx::Integer=0, iqy::Integer=0)
    (;Lx, Ly, xlims, ylims, M, V, V_q, nc) = xh
    xs = range(0, Lx, nx) # these are the differences `x - xlims[1]`, with `x ∈ xlims`
    ys = range(0, Ly, ny)
    R = typeof(Lx) # real working type
    ψ_type = !xh.isperiodic && eltype(xh.H) isa Real ? R : complex(R)
    ψ = [Matrix{ψ_type}(undef, nx, ny) for _ in 1:nc] # `ψ` are real if elements of H are real and if the problem is nonperiodic (meaning basis is real)
    for c in 1:nc
        if xh.isperiodic
            B = 2M + 1
            if iqx != 0 # if quasimomentum index has been passed
                @floop for (iy, y) in enumerate(ys)
                    for (ix, x) in enumerate(xs)
                        ψ[c][ix, iy] = sum(V_q[(c-1)*B^2+(j-1)B+i, stateno, iqx, iqy]cis(2π*jx*x/Lx + 2π*jy*y/Ly) for (j, jx) in enumerate(-M:M)
                                                                                                                  for (i, jy) in enumerate(-M:M)) / √(Lx*Ly)
                    end
                end
            else # no quasimomentum index
                @floop for (iy, y) in enumerate(ys)
                    for (ix, x) in enumerate(xs)
                        ψ[c][ix, iy] = sum(V[(c-1)*B^2+(j-1)B+i, stateno]cis(2π*jx*x/Lx + 2π*jy*y/Ly) for (j, jx) in enumerate(-M:M)
                                                                                                      for (i, jy) in enumerate(-M:M)) / √(Lx*Ly)
                    end
                end
            end
        else # nonperiodic
            @floop for (iy, y) in enumerate(ys)
                for (ix, x) in enumerate(xs)
                    ψ[c][ix, iy] = sum(V[(c-1)*M^2+(jx-1)M+jy, stateno]sin(π*jx*x/Lx)sin(π*jy*y/Ly) for jx in 1:M for jy in 1:M) * 2 / √(Lx*Ly)
                end
            end
        end
    end
    return xs .+ xlims[1], ys .+ ylims[1], ψ # return "normal" coordinates, in `x ∈ xlims` and `y ∈ ylims`
end

"""
Calculate eigenenergies for all pairs of quasimomenta in `qxs` and `qys`.
Calculate `nev` lowest levels using `ArnoldiMethod`.
If `nev=0` or not passed, then full diagonalisation using `LinearAlgebra` is performed.
Note that `dh.H` is modified in the process.
"""
function diagonalize!(dh::DenseHamiltonian2D{R,T,S}, qxs::AbstractVector{<:Real}, qys::AbstractVector{<:Real}; nev::Integer, verbose::Bool=false) where {R<:Real, T<:Number, S<:Number}
    (;M, xlims, ylims, Lx, Ly, δ, nc, H, 𝑈, 𝑈_iseven, 𝐴_x, 𝐴_y, Γ) = dh

    if !dh.isperiodic
        @warn "Hamiltonian must be periodic. Construct a new one and try again."
        return
    end

    PI = R(π) # π of the working type to prevent widening

    B = (2M + 1)^2 # block size
    nsaves = nev == 0 ? B : nev # number of eigenvalues and eigenvectors to allocate
    dh.ε_q = Array{S,3}(undef, nsaves, length(qxs), length(qys))
    dh.V_q = Array{T,4}(undef, B*nc, nsaves, length(qxs), length(qys))
    
    if isnothing(𝐴_x) && isnothing(𝐴_y)
        H_diag = diagview(dh.H)
        H_diag_copy = diag(dh.H) # a copy for restoring after the calculation
        # from the diagonal of each diagonal block of `H`, extract (𝑈ᵢᵢ)₀ (the 0th harmonic of 𝑈ᵢᵢ) plus decay -iΓ/2
        U_diags = [H_diag[(c-1)B + B÷2+1] for c in 1:nc] # generally, `Hᵢᵢ = -Δᵢᵢ + Uᵢᵢ - iΓ/2`, but Δᵢᵢ = 0 for the central element of the diagonal (see construction of Δ in `DenseHamiltonian1D` constructor)
    else
        N = 4M + 1 # number of points for FFT. This will yield harmonics from -2M to 2M
        dx, dy = Lx/N, Ly/N
        xs = range(xlims[1], xlims[2]-dx, N)
        ys = range(ylims[1], ylims[2]-dy, N)

        fft_buff = Matrix{Complex{R}}(undef, N, N) # a buffer for all (in-place) FFTs
        F = FFTW.plan_fft!(fft_buff) # the savings of rfft are negligible, and the output is much less convenient to handle in `fft_to_matrix`, so using fft. Also, this way we can do FFT in-place

        D_x = [Matrix{T}(undef, B, B) for _ in 1:nc] # for storing `nc` kinetic operators -iδ∂ₓ - 𝐴ₓ
        D_y = [Matrix{T}(undef, B, B) for _ in 1:nc] # for storing `nc` kinetic operators -iδ∂𝑦 - 𝐴𝑦
        U = [Matrix{T}(undef, B, B) for _ in 1:nc] # for storing `nc` terms (𝑈ᵢᵢ)₀ - iΓ/2

        # iterate over `𝑈` and populate `H`
        for c in 1:nc
            𝑢 = 𝑈[c, c]

            # calculate and store FFT of 𝑢
            if isnothing(𝑢)
                U[c] .= 0
            else
                𝑢_isrealeven = (𝑢(xlims[1], ylims[1]) isa Real) & 𝑈_iseven[c, c]
                fft_buff .= 𝑢.(xs, ys')
                F * fft_buff # in-place FFT, weird syntax
                fft_buff ./= N^2
                U[c] .= fft_to_matrix_naive!(fft_buff, make_real=𝑢_isrealeven)
            end

            if Γ[c] != 0
                U[c] .-= LA.I * im*Γ[c]/2
            end
            ∂_x = Diagonal([2PI * δ * jx/Lx for jx in -M:M for jy in -M:M]) # this is -iδ∂ₓ
            ∂_y = Diagonal([2PI * δ * jy/Ly for jx in -M:M for jy in -M:M]) # this is -iδ∂y
            if !isnothing(𝐴_x)
                fft_buff .= 𝐴_x.(xs, ys')
                F * fft_buff
                fft_buff ./= N^2
                A_x = fft_to_matrix_naive!(fft_buff)
                D_x[c] .= ∂_x .- A_x
                # if there is no 𝐴𝑦, then set the 𝑦 kinetic `D_y[c]` term to -iδ∂𝑦. Otherwise `D_y[c]` will be treated in the next if clause
                isnothing(𝐴_y) && (D_y[c] .= ∂_y)
            end
            if !isnothing(𝐴_y)
                fft_buff .= 𝐴_y.(xs, ys')
                F * fft_buff
                fft_buff ./= N^2
                A_y = fft_to_matrix_naive!(fft_buff)
                D_y[c] .= ∂_y .- A_y
                # if there is no 𝐴ₓ, then set the 𝑥 kinetic `D_x[c]` term to -iδ∂ₓ. Otherwise `D_x[c]` was treated in the preceding if clause
                isnothing(𝐴_x) && (D_x[c] .= ∂_x)
            end
        end
    end
    
    # update diagonal blocks and diagonalise
    for (iqy, qy) in enumerate(qys), (iqx, qx) in enumerate(qxs)
        # update diagonal blocks
        if isnothing(𝐴_x) && isnothing(𝐴_y)
            for c in 1:nc
                H_diag[(c-1)B+1:c*B] .= [(2PI*δ*jx/Lx + qx)^2 + (2PI*δ*jy/Ly + qy)^2 + U_diags[c] for jx in -M:M for jy in -M:M]
            end
        else
            for c in 1:nc
                H[(c-1)*B+1:c*B, (c-1)*B+1:c*B] .= (D_x[c] + LA.I*qx)^2 + (D_y[c] + LA.I*qy)^2 + U[c]
            end
        end

        dh.ε_q[:, iqx, iqy], dh.V_q[:, :, iqx, iqy] = diagonalize(dh; nev, verbose)
    end
end

##### Unused but correct and tested functions

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