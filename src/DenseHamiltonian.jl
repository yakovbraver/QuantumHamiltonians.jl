"""
A type representing a spatial [𝑟 = (𝑥, 𝑦)], 𝑛-component, possibly quasimomentum-dependent Hamiltonian (𝐻ᵢⱼ)
    𝐻ᵢᵢ(𝑟) = [-i𝛿∇ + 𝑞 - 𝐴(𝑟)]² + 𝑈ᵢᵢ(𝑟)
    𝐻ᵢⱼ(𝑟) = 𝑉ᵢⱼ(𝑟)
as a dense matrix.
"""
mutable struct DenseHamiltonian{R<:Real,T<:Number,S<:Number} <: XSpaceHamiltonian # in practice `T` shoudld be `R` or `Complex{R}` (and same for `S`) -- always check this. If this is not the case, probably your 𝑈 or 𝐴 do not return R's.
    xlims::Tuple{R, R}
    ylims::Tuple{R, R}
    Lx::R # length along 𝑥
    Ly::R # length along 𝑦
    M::Int # maximum harmonic number (will use -M:M for periodic, 1:M for nonperiodic)
    δ::R # coefficient of the momentum term: -iδ∇
    nc::Int # number of components
    isperiodic::Bool
    ishermitian::Bool # `H` is nonhermitian if decays Γ are present
    𝐻::Matrix{<:Union{Function,Nothing}} # nc-component Hamiltonian matrix containing coordinate-space functions
    𝐴_x::Union{Function,Nothing}
    𝐴_y::Union{Function,Nothing}
    H::Matrix{T} # momentum-space Hamiltonian used for diagonalisation
    ε::Vector{S} # eigenvalues, can be complex for nonhermitian `H`, hence additional type `S`
    V::Matrix{T} # eigenvectors matrix
    ε_q::Array{S,3} # ε_q[n, iqx, iqy] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    V_q::Array{T,4} # V_q[:, n, iqx, iqy] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)
end

"""
Construct a `DenseHamiltonian` object using the coordinate-space functions stored in `𝐻`, decay rates `Γ`, and gauge field (same for all components) 𝐴_x, 𝐴_y.
`M` is the maximum harmonic number. In the periodic case, the Hamiltonian will be `nc*(2M+1)²`-by-`nc*(2M+1)²` where `nc` is the number of components.
In nonperiodic case, the size will be `nc*M²`-by-`nc*M²`.
`𝐻_iseven[i, j]` matters only if `isperiodic=true` and shows whether `𝐻[i, j]` is an even function (i.e. whether ℎ(𝑥, 𝑦) = ℎ(-𝑥, -𝑦)). If it is, then Fourier transform is real, which is used for better accuracy.
If *all* functions are even (and real), then the resulting Fourier-space Hamiltonian is real (provided also there is no 𝐴 and Γ), giving a speed-up and better accuracy (compared to complex diagonalisation).
If `𝐻[i, j] === nothing` or it is complex, then the value of `𝐻_iseven[i, j]` does not matter.
"""
function DenseHamiltonian(xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                          𝐻::AbstractMatrix{<:Union{Function,Nothing}}, 𝐻_iseven::AbstractMatrix{Bool}=falses(size(𝐻)), Γ::Vector{R}=zeros(R, size(𝐻, 1)),
                          𝐴_x::Union{Function,Nothing}=nothing, 𝐴_y::Union{Function,Nothing}=nothing) where R <: Real
    Lx, Ly = xlims[2]-xlims[1], ylims[2]-ylims[1]
    
    PI = R(π) # π of the working type to prevent widening

    nc = size(𝐻, 1) # number of components

    # `isreal` will show if the resulting `H` will be real
    isreal = all( typeof(ℎ(xlims[1], ylims[1])) <: Real for ℎ in 𝐻 if !isnothing(ℎ)) & # check if all functions in 𝐻 are real
             isnothing(𝐴_x) & isnothing(𝐴_y) & all(==(0), Γ)
    if isperiodic # for periodic potential, also check if potential is even 
        isreal &= all(𝐻_iseven)
    end

    H_sz = isperiodic ? (2M+1)^2 : M^2 # size of each Hamiltonian block

    # allocate `H`
    if isreal
        H = zeros(R, nc*H_sz, nc*H_sz)
    else
        H = zeros(Complex{R}, nc*H_sz, nc*H_sz)
    end

    if isperiodic
        N = 4M # number of points for FFT. This will yield harmonics from -2M to 2M
        dx, dy = Lx/N, Ly/N
        xs = range(xlims[1], xlims[2]-dx, N)
        ys = range(ylims[1], ylims[2]-dy, N)

        f = dx/Lx * dy/Ly

        u_real = Matrix{R}(undef, N, N)
        F = FFTW.plan_rfft(u_real)

        # iterate over `𝐻` and populate `H`
        for jH in axes(𝐻, 2)
            for iH in 1:jH # only upper triangle is scanned. The lower triangle is filled only if Γ is present
                wi = (iH-1)*H_sz+1:iH*H_sz
                wj = (jH-1)*H_sz+1:jH*H_sz

                ℎ = 𝐻[iH, jH]
                ℎ_iseven = 𝐻_iseven[iH, jH]

                # calculate and store FFT of ℎ
                if !isnothing(ℎ)
                    if typeof(ℎ(xs[1], ys[1])) <: Real
                        for (iy, y) in enumerate(ys), (ix, x) in enumerate(xs)
                            u_real[ix, iy] = ℎ(x, y)
                        end
                        H[wi, wj] .= dft_to_matrix(F * u_real, make_real=ℎ_iseven) .* f # TODO: `F * u_real` allocates a temporary
                    end
                    #else -- not implemented
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
                        if 𝐴_x !== nothing
                            a_i = [𝐴_x(x, y) for x in xs, y in ys] # using generic naming "_i" to reuse the same variables in the next `if`
                            A_i = F * a_i * f |> dft_to_matrix
                            ∂_i = Diagonal([2PI * δ * jx/Lx for jx in -M:M for jy in -M:M]) # this is -iδ∂ₓ
                            H[wi, wj] += (∂_i - A_i)^2
                            # if there is no 𝐴𝑦, then add ∂ₓ². Otherwise it will be added together with 𝐴𝑦 in the next `if` clause
                            isnothing(𝐴_y) && (H[wi, wj] += Diagonal([(2PI * δ * jy/Ly)^2 for jx in -M:M for jy in -M:M]))
                        end
                        if 𝐴_y !== nothing
                            a_i = [𝐴_y(x, y) for x in xs, y in ys]
                            A_i = F * a_i * f |> dft_to_matrix
                            ∂_i = Diagonal([2PI * δ * jy/Ly for jx in -M:M for jy in -M:M]) # this is -iδ∂y
                            H[wi, wj] += (∂_i - A_i)^2
                            # if there is no 𝐴ₓ, then add ∂𝑦². Otherwise it was added together with 𝐴ₓ in the preceding `if` clause
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

        f = dx/Lx * dy/Ly

        u_real = Matrix{R}(undef, length(xs), length(ys))
        u_imag = Matrix{R}(undef, isreal ? 0 : length(xs), isreal ? 0 : length(ys)) # if `isreal`, then make the matrix 0x0
        F = FFTW.plan_r2r!(u_real, FFTW.REDFT00)

        # iterate over `𝐻` and populate `H`
        for jH in axes(𝐻, 2)
            for iH in 1:jH # only upper triangle is scanned. The lower triangle is filled only if Γ is present
                wi = (iH-1)*H_sz+1:iH*H_sz
                wj = (jH-1)*H_sz+1:jH*H_sz

                ℎ = 𝐻[iH, jH]

                # calculate and store FFT of ℎ
                if !isnothing(ℎ)
                    if typeof(ℎ(xs[1], ys[1])) <: Real
                        for (iy, y) in enumerate(ys), (ix, x) in enumerate(xs)
                            u_real[ix, iy] = ℎ(x, y)
                        end
                        (F * u_real) .*= f
                        H[wi, wj] .= dct_to_matrix(u_real)
                    else
                        # here `ℎ` is a complex function, but `FFTW.REDFT00` can only handle real ones. So we transform Re and Im separately.
                        for (iy, y) in enumerate(ys), (ix, x) in enumerate(xs)
                            u_real[ix, iy], u_imag[ix, iy] = reim(ℎ(x, y))
                        end
                        (F * u_real) .*= f
                        (F * u_imag) .*= f
                        H[wi, wj] .= dct_to_matrix(u_real)
                        H[wi, wj] .+= im .* dct_to_matrix(u_imag)
                    end
                end

                if iH == jH # for a diagonal block, add the laplace term, optionally Γ, and the 𝐴's
                    # TODO: just subtract from diagonal
                    H[wi, wj] += Diagonal([(PI*δ)^2 * ((jx/Lx)^2 + (jy/Ly)^2) for jx in 1:M for jy in 1:M]) # add -δ²Δ
                    # for diagonal block, add Laplacian, Γ, and 𝐴
                    if Γ[iH] != 0
                        H[diagind(H)[wi]] .-= im*Γ[iH]/2
                    end

                    if 𝐴_x !== nothing
                        a_i = [𝐴_x(x, y) for x in xs, y in ys]
                        (F * a_i) .*= f
                        A_i = dct_to_matrix(a_i)
                        ∂_i = make_∂_x(M, Lx)
                        H[wi, wj] += im*(A_i*∂_i + ∂_i*A_i) + A_i^2 # The perfect square for `(∂_x - A_x)^2` is much less accurate
                    end
                    if 𝐴_y !== nothing
                        a_i = [𝐴_y(x, y) for x in xs, y in ys]
                        (F * a_i) .*= f
                        A_i = dct_to_matrix(a_i)
                        ∂_i = make_∂_y(M, Ly)
                        H[wi, wj] += im*(A_i*∂_i + ∂_i*A_i) + A_i^2 # The perfect square for `(∂_y - A_y)^2` is much less accurate
                    end
                elseif !all(iszero, Γ) # fill conjugate block if Γ is present (then we cannot use Hermitian view when diagonalising)
                    H[wj, wi] .= @view(H[wi, wj])'
                end
            end
        end
    end
    
    # determine the type of eigenvalues 
    ishermitian = all(==(0), Γ) # if all `Γ`s are zeros, then Hamiltonian is Hermitian and the eigenvalues real
    S = ishermitian ? R : Complex{R} # type of eigenvalues
    return DenseHamiltonian(xlims, ylims, Lx, Ly, M, δ, nc, isperiodic, ishermitian, 𝐻, 𝐴_x, 𝐴_y, H, S[], eltype(H)[;;], S[;;;], eltype(H)[;;;;])
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
Based on results of a real 2D fft `u`, return the matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
`make_real=true` will take real parts of elements of `u`.
"""
function dft_to_matrix(u; make_real=false)
    B = size(u, 2) ÷ 2 + 1 # the size of each block
    n_B = size(u, 1) # the number of block-rows (= number of block-cols). For actual applications (i.e. when `u` is the output of `rfft`), `n_B == B`
    H = make_real == true ? zeros(real(eltype(u)), B*n_B, B*n_B) : zeros(eltype(u), B*n_B, B*n_B)
    H[diagind(H)] .= real(u[1, 1]) # save the secular component
    u[1, 1] = 0 # remove because it breaks the structure of the loop below if included

    # it is assumed that u[1, 1] == 0 -- otherwise, one would also need to prevent double pushing of the diagonal elements
    @floop for c_u in axes(u, 2)
        for r_u in axes(u, 1) # iterate over columns and rows of `u`
            u[r_u, c_u] == 0 && continue
            val = make_real == true ? real(u[r_u, c_u]) : u[r_u, c_u]
            for r_b in r_u:n_B # a value from `r_u`th row of `u` will be put in block-rows of `H` from `r_u`th to `n_B`th
                c_b = r_b - r_u + 1 # block-column where to place the value
                if c_u < B # for `c_u` < `B`, the value from `c_u`th column of `u` will be put to the `c_u`th lower diagonal of the block
                    for (r, c) in zip(c_u:B, 1:B+1-c_u)
                        push_vals!(H; r_b, c_b, r, c, blocksize=B, val)
                    end
                elseif c_u == B # for `c_u` = `B`, the value from `c_u`th column of `u` will be put to lower left and upper right corners of the block
                    push_vals!(H; r_b, c_b, r=B, c=1, blocksize=B, val)
                    if r_b != c_b
                        push_vals!(H; r_b, c_b, r=1, c=B, blocksize=B, val) # if we're in the diagonal block, then the upper right corner is conjugate to lower left and has already been pushed
                    end
                else # for `c_u` > `B`, the value from `c_u`th column of `u` will be put to the `2B-c_u`th upper diagonal of the block
                    if r_b != c_b # if `r_b == c_b`, then upper diagonal of the block has already been filled by pushing the conjugate element
                        c_u_inv = 2B - c_u
                        for (r, c) in zip(1:B+1-c_u_inv, c_u_inv:B)
                            push_vals!(H; r_b, c_b, r, c, blocksize=B, val)
                        end
                    end
                end
            end
        end
    end
    return H
end

"""
Push value `val` to element (`r`, `c`) of the block (`r_b`, `c_b`) of `H`, with block size being `blocksize`.
The complex-conjugate element is also pushed.
"""
function push_vals!(H; r_b, c_b, r, c, blocksize, val)
    i = (r_b-1)*blocksize + r
    j = (c_b-1)*blocksize + c
    H[i, j] = val
    H[j, i] = val'
end

"""
Based on results of 2D DCT `u`, return the matrix indexed by (𝑗′ₓ𝑗′y, 𝑗ₓ𝑗y).
"""
function dct_to_matrix(u)
    N = size(u, 1) # number of points used for FFT
    M = (N-1) ÷ 2 # maximum harmonic number
    H = Matrix{eltype(u)}(undef, M^2, M^2)
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
function make_eigenfunction(xh::XSpaceHamiltonian, stateno::Integer, nx::Integer, ny::Integer, iqx::Integer=0, iqy::Integer=0)
    (;Lx, Ly, xlims, ylims, M, V, V_q, nc) = xh
    xs = range(0, Lx, nx) # these are the differences `x - xlims[1]`, with `x ∈ xlims`
    ys = range(0, Ly, ny)
    R = typeof(Lx) # real working type
    ψ_type = !xh.isperiodic && eltype(xh.H) <: Real ? R : complex(R)
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
Calculate `nev` lowest eigenvectors and eigenvalues using `ArnoldiMethod`.
Pass `nev=0` for full diagonalisation using `LinearAlgebra`.
"""
function diagonalize!(dh::DenseHamiltonian; nev::Integer)
    if nev == 0
        if dh.ishermitian
            dh.ε, dh.V = eigen(Hermitian(dh.H)) # if `dh.H` is real, the appropriate routine will be selected automatically, no need to use `Symmetric` instead of `Hermitian`
        else
            dh.ε, dh.V = eigen(dh.H)
        end
    else
        if dh.ishermitian
            S, info = partialschur(dense_linear_map(Hermitian(dh.H)); nev, which=:LM); # `which=:SR` with no shift-invert does not converge
            @show info
            dh.V = S.Q
            dh.ε = inv.(real.(S.eigenvalues)) # invert back
        else
            S, info = partialschur(dense_linear_map(dh.H); nev, which=:LM);
            @show info
            dh.ε, dh.V = partialeigen(S)
            dh.ε .= inv.(dh.ε)
        end
    end
end

"Helper function for shift-and-invert: construct a linear map that applies the inverse of `A`."
function dense_linear_map(A)
    F = factorize(A)
    LinearMap{eltype(A)}((y, x) -> ldiv!(y, F, x), size(A, 1), ismutating=true)
end

"""
Calculate eigenenergies for all pairs of quasimomenta in `qxs` and `qys`.
Calculate `nev` lowest bands using `ArnoldiMethod`.
If `nev=0` or not passed, then full diagonalisation using `LinearAlgebra` is performed.
Note that `dh.H` is modified in the process.
"""
function diagonalize!(dh::DenseHamiltonian{R,T}, qxs::AbstractVector{<:Real}, qys::AbstractVector{<:Real}; nev::Integer=0) where {R<:Real, T<:Number}
    (;M, xlims, ylims, Lx, Ly, δ, 𝑈, 𝐴_x, 𝐴_y) = dh

    nsaves = nev == 0 ? (2M+1)^2 : nev # number of eigenvalues and eigenvectors to allocate
    dh.ε_q = Array{R,3}(undef, nsaves, length(qxs), length(qys))
    dh.V_q = Array{T,4}(undef, (2M+1)^2, nsaves, length(qxs), length(qys))
    
    if 𝐴_x === nothing
        H_diag = diagview(dh.H)
        U_diag = H_diag[(end+1) ÷ 2] # generally, `H = -Δ + U`, but this element is purely `U`, since Laplace is zero (see construction of Δ in `DenseHamiltonian` constructor)
    else
        N = 4M # number of points for FFT. This will yield harmonics from -2M to 2M
        dx, dy = Lx/N, Ly/N
        xs = range(xlims[1], xlims[2]-dx, N)
        ys = range(ylims[1], ylims[2]-dy, N)

        f = dx/Lx * dy/Ly
        u = [𝑈(x, y) for x in xs, y in ys]

        F = FFTW.plan_rfft(u)
        U = F * u * f |> dft_to_matrix
    
        a_x = [𝐴_x(x, y) for x in xs, y in ys]
        a_y = [𝐴_y(x, y) for x in xs, y in ys]

        D_x = F * a_x * -f |> dft_to_matrix # this is -𝐴ₓ
        D_y = F * a_y * -f |> dft_to_matrix # this is -𝐴y
        
        D_x += Diagonal(typeof(Lx)[2π * δ * jx/Lx for jx in -M:M for jy in -M:M]) # this adds -iδ∂ₓ and results in -iδ∂ₓ-𝐴ₓ
        D_y += Diagonal(typeof(Lx)[2π * δ * jy/Ly for jx in -M:M for jy in -M:M]) # this adds -iδ∂y and results in -iδ∂y-𝐴y
    end
    
    # iterate quasimomenta
    for (iqy, qy) in enumerate(qys), (iqx, qx) in enumerate(qxs)
        # update diagonal
        if 𝐴_x === nothing
            H_diag = [(2π*δ*jx/Lx + qx)^2 + (2π*δ*jy/Ly + qy)^2 + U_diag for jx in -M:M for jy in -M:M]
        else
            dh.H = (D_x + LA.I*qx)^2 + (D_y + LA.I*qy)^2 + U
        end

        # diagonalise
        if nev == 0
            dh.ε_q[:, iqx, iqy], dh.V_q[:, :, iqx, iqy] = eigen(Hermitian(dh.H))
        else
            S, info = partialschur(dense_linear_map(Hermitian(dh.H)); nev, which=:LM, tol=1e-7); # `which=:SR` does not converge, so we use "shift-invert" (although shift is zero)
            @show info
            dh.V_q[:, :, iqx, iqy] = S.Q
            dh.ε_q[:, iqx, iqy] = inv.(real.(S.eigenvalues)) # invert back
        end
    end
end