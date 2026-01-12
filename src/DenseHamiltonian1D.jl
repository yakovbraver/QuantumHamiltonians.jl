"A type for storing the Wannier functions."
mutable struct Wanniers{R<:Real}
    targetlevels::Vector{Int} # numbers of energy levels to use for constructing wanniers
    E::Vector{R} # mean energies
    pos::Vector{R} # positions (wannier centres)
    V::Matrix{Complex{R}} # position eigenvectors
end

"Default-construct an empty `Wanniers` object."
Wanniers{R}() where R <: Real = Wanniers(Int[], R[], R[], Complex{R}[;;])

"""
A type representing a spatial, 𝑛-component, possibly quasimomentum-dependent Hamiltonian (𝐻ᵢⱼ)
    𝐻ᵢᵢ(𝑥) = (-i𝛿∂ₓ + 𝑞)² + 𝑈ᵢᵢ(𝑥)
    𝐻ᵢⱼ(𝑟) = 𝑈ᵢⱼ(𝑟)
as a dense matrix.
All 𝑈ᵢⱼ(𝑟) are assumed real (contrary to the 2D case).
"""
mutable struct DenseHamiltonian1D{R<:Real,T<:Number,S<:Number} <: XSpaceHamiltonian1D{:dense} # in practice `T` shoudld be `R` if there is no 𝑞 and 𝑈 is even, or `Complex{R}` otherwise -- always check this. If this is not the case, probably your 𝑈 or 𝐴 do not return R's.
    xlims::Tuple{R, R}
    Lx::R # length along 𝑥
    M::Int # maximum harmonic number (will use -M:M for periodic, 1:M for nonperiodic)
    δ::R # coefficient of the momentum term
    nc::Int # number of components
    isperiodic::Bool
    ishermitian::Bool # `H` is nonhermitian if decays Γ are present
    𝑈::Matrix{<:Union{Function,Nothing}} # nc-component matrix containing coordinate-space potentials and couplings
    H::Matrix{T} # momentum-space Hamiltonian used for diagonalisation
    ε::Vector{S} # eigenvalues, can be complex for nonhermitian `H`, hence additional type `S`
    V::Matrix{T} # eigenvectors matrix
    ε_q::Array{S,2} # ε_q[n, iqx] = `n`th band eigenvalue at quasimomentum at index `iqx`
    V_q::Array{T,3} # V_q[:, n, iqx] = `n`th band eigenvector at quasimomentum at index `iqx`
    wanniers::Wanniers{R} # wanniers are implemented for the case `nc = 1` only
end

"""
Construct a `DenseHamiltonian1D` object using the coordinate-space functions stored in `𝑈` and decay rates `Γ`.
`M` is the maximum harmonic number. In the periodic case, the Hamiltonian will be `nc*(2M+1)`-by-`nc*(2M+1)` where `nc` is the number of components.
In nonperiodic case, the size will be `nc*M`-by-`nc*M`.
`𝑈_iseven[i, j]` matters only if `isperiodic=true` and shows whether `𝑈[i, j]` is an even function (i.e. whether 𝑢(𝑥) = 𝑢(-𝑥)). If it is, then Fourier transform is real, which is used for better accuracy.
If *all* functions are even (and real), then the resulting Fourier-space Hamiltonian is real (provided also there is no Γ), giving a speed-up and better accuracy (compared to complex diagonalisation).
If `𝑈[i, j] === nothing`, then the value of `𝑈_iseven[i, j]` does not matter.
"""
function DenseHamiltonian1D(𝑈::AbstractMatrix{<:Union{Function,Nothing}}, xlims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                            𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1))) where R <: Real
    Lx = xlims[2] - xlims[1]

    PI = R(π) # π of the working type to prevent widening

    nc = size(𝑈, 1) # number of components

    # `isreal` will show if the resulting `H` will be real
    isreal = all(==(0), Γ)
    if isperiodic # for periodic potential, also check if functions are even 
        isreal &= all(𝑈_iseven[𝑈 .!== nothing])
    end

    B = isperiodic ? 2M+1 : M # size of each Hamiltonian block

    # allocate `H`
    if isreal
        H = zeros(R, nc*B, nc*B)
    else
        H = zeros(Complex{R}, nc*B, nc*B)
    end

    if isperiodic
        N = 4M + 1 # number of points for FFT. This will yield harmonics from -2M to 2M
        dx = Lx/N
        xs = range(xlims[1], xlims[2]-dx, N)

        fft_buff = Vector{Complex{R}}(undef, N) # a buffer for all (in-place) FFTs
        F = FFTW.plan_fft!(fft_buff) # the savings of rfft are negligible, and the output is much less convenient to handle in `fft_to_matrix`, so using fft. Also, this way we can do FFT in-place

        # iterate over `𝑈` and populate `H`
        for jH in axes(𝑈, 2)
            for iH in 1:jH # only upper triangle is scanned. The lower triangle is filled only if Γ is present
                wi = (iH-1)*B+1:iH*B
                wj = (jH-1)*B+1:jH*B

                𝑢 = 𝑈[iH, jH]

                # calculate and store FFT of 𝑢
                if !isnothing(𝑢)
                    𝑢_isrealeven = 𝑈_iseven[iH, jH]
                    fft_buff .= 𝑢.(xs)
                    F * fft_buff # in-place FFT, weird syntax
                    fft_buff ./= N
                    H[wi, wj] .= fft_to_matrix_1D!(fft_buff, make_real=𝑢_isrealeven)
                end

                # for diagonal block, add Laplacian, Γ, and 𝐴
                if iH == jH
                    H[wi, wj] += Diagonal([(2PI*δ)^2 * (jx/Lx)^2 for jx in -M:M]) # this is -δ²Δ
                    if Γ[iH] != 0
                        H[diagind(H)[wi]] .-= im*Γ[iH]/2
                    end
                elseif !all(iszero, Γ) # fill conjugate block if Γ is present (then we cannot use Hermitian view when diagonalising)
                    H[wj, wi] .= @view(H[wi, wj])'
                end
            end
        end
    else # non-periodic
        N = 2M + 1
        xs = range(xlims[1], xlims[2], N)
        dx = xs[2] - xs[1]

        fft_buff = Vector{R}(undef, N)
        F = FFTW.plan_r2r!(fft_buff, FFTW.REDFT00)

        # iterate over `𝑈` and populate `H`
        for jH in axes(𝑈, 2)
            for iH in 1:jH # only upper triangle is scanned. The lower triangle is filled only if Γ is present
                wi = (iH-1)*B+1:iH*B
                wj = (jH-1)*B+1:jH*B

                𝑢 = 𝑈[iH, jH]

                # calculate and store FFT of 𝑢
                if !isnothing(𝑢)
                    fft_buff .= 𝑢.(xs)
                    (F * fft_buff)
                    fft_buff ./= N-1
                    H[wi, wj] .= dct_to_matrix_1D(fft_buff)
                end

                if iH == jH # for a diagonal block, add the laplace term and optionally Γ
                    H[wi, wj] += Diagonal([(PI*δ)^2 * (jx/Lx)^2 for jx in 1:M]) # add -δ²Δ
                    if Γ[iH] != 0
                        H[diagind(H)[wi]] .-= im*Γ[iH]/2
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
    return DenseHamiltonian1D(xlims, Lx, M, δ, nc, isperiodic, ishermitian, 𝑈, H, S[], eltype(H)[;;], S[;;], eltype(H)[;;;], Wanniers{R}())
end

"""
Construct from the result of 1D FFT `u` the matrix `U` indexed by (𝑗′ₓ, 𝑗ₓ).
`make_real=true` will mutate `u`, taking the real parts of (the first half of) elements, which is useful if the original function is even and hence the transform is known to be real.
"""
function fft_to_matrix_1D!(u::Vector{T}; make_real::Bool=false) where T <: Number
    N = length(u) # number of points used for FFT
    M = (N-1) ÷ 4 # maximum harmonic number (recall that N = 4M + 1 in the constructor)
    B = 2M + 1 # size of the resulting matrix

    U = Matrix{make_real ? real(T) : T}(undef, B, B)
    u_half = @view(u[1:B]) # we assume that `u` is the transform of a real function, so transform obeys uₘ* = u₋ₘ and we only need to work with one half
    make_real && (u_half .= real.(u_half))

    for (i, val) in enumerate(u_half)
        U[diagind(U, 1-i)] .= val  # fill lower triangle (including the diagonal)
        U[diagind(U, i-1)] .= val' # fill upper triangle (including the diagonal)
    end
    return U
end

"""
Based on results of 1D DCT `u`, return the matrix indexed by (𝑗′ₓ, 𝑗ₓ).
"""
function dct_to_matrix_1D(u::Vector{T}) where T <: Number
    N = length(u) # number of points used for FFT
    M = (N-1) ÷ 2 # maximum harmonic number
    U = Matrix{T}(undef, M, M)
    @floop for jx in 1:M
        for j′x in 1:M
            j₋x = abs(j′x-jx)
            U[j′x, jx] = (u[j₋x+1] - u[j′x+jx+1]) / 2
        end
    end
    return U
end

"""
Construct eigenfunctions of state numbers `statenos` on a grid having `nx` points in `x` direction.
If a vector of quasimomentum indices `iqxs` is passed, then construct `ψ` for the state `statenos[1]` at the these quasimomenta.
Return (`xs`, `ψ`) where `ψ[x, components, statenos]` or `ψ[x, components, iqxs]`
"""
function make_eigenfunctions(xh::DenseHamiltonian1D; statenos::AbstractVector{<:Integer}, nx::Integer, iqxs::AbstractVector{<:Integer}=Int[])
    (;Lx, xlims, M, V, V_q, nc) = xh
    xs = range(0, Lx, nx) # these are the differences `x - xlims[1]`, with `x ∈ xlims`
    ns = isempty(iqxs) ? length(statenos) : length(iqxs)
    R = typeof(Lx) # real working type
    ψ_type = !xh.isperiodic && eltype(xh.H) <: Real ? R : complex(R)  # `ψ` are real if elements of H are real and if the problem is nonperiodic (meaning basis is real)
    ψ = Array{ψ_type}(undef, nx, nc, ns)
    if isempty(iqxs) # no quasimomentum index
        for (is, stateno) in enumerate(statenos)
            for c in 1:nc
                if xh.isperiodic
                    B = 2M + 1
                    @floop for (ix, x) in enumerate(xs)
                        ψ[ix, c, is] = sum(V[(c-1)*B+j, stateno]cis(2π*jx*x/Lx) for (j, jx) in enumerate(-M:M)) / √Lx
                    end
                else # nonperiodic
                    @floop for (ix, x) in enumerate(xs)
                        ψ[ix, c, is] = sum(V[(c-1)*M+jx, stateno]sin(π*jx*x/Lx) for jx in 1:M) * √(2/Lx)
                    end
                end
            end
        end
    else # quasimomenta indices passed
        for iqx in iqxs
            for c in 1:nc
                B = 2M + 1
                @floop for (ix, x) in enumerate(xs)
                    ψ[ix, c, iqx] = sum(V_q[(c-1)*B+j, statenos[1], iqx]cis(2π*jx*x/Lx) for (j, jx) in enumerate(-M:M)) / √Lx
                end
            end
        end
    end
    return xs .+ xlims[1], ψ # return "normal" coordinates, in `x ∈ xlims`
end

"""
Calculate eigenenergies for all quasimomenta in `qxs`.
Calculate `nev` lowest bands using `ArnoldiMethod`.
If `nev=0` or not passed, then full diagonalisation using `LinearAlgebra` is performed.
"""
function diagonalize!(dh::DenseHamiltonian1D{R,T,S}, qxs::AbstractVector{<:Real}; nev::Integer, verbose::Bool=false) where {R<:Real, T<:Number, S<:Number}
    (;M, Lx, δ, nc, H) = dh
   
    B = 2M + 1 # block size
    nsaves = nev == 0 ? B*nc : nev # number of eigenvalues and eigenvectors to allocate
    dh.ε_q = Array{S,2}(undef, nsaves, length(qxs))
    dh.V_q = Array{T,3}(undef, B*nc, nsaves, length(qxs))
    
    H_diag = diagview(dh.H)
    H_diag_copy = diag(dh.H) # a copy for restoring after the calculation
    # from the diagonal of each diagonal block of `H`, extract the 0th harmonic of 𝑈ᵢᵢ plus decay -iΓ/2
    U_diags = [H_diag[(c-1)B + B÷2+1] for c in 1:nc] # generally, `H₀₀ = -Δ₀₀ + U₀₀ - iΓ/2`, but Δ₀₀ = 0 for the central element of the diagonal (see construction of Δ in `DenseHamiltonian1D` constructor)
    
    # iterate quasimomenta
    for (iqx, qx) in enumerate(qxs)
        # update diagonal
        for c in 1:nc
            H_diag[(c-1)B+1:c*B] .= [(2π*δ*jx/Lx + qx)^2 + U_diags[c] for jx in -M:M]
        end

        # diagonalise
        if nev == 0
            if dh.ishermitian
                dh.ε_q[:, iqx], dh.V_q[:, :, iqx] = eigen(Hermitian(H))
            else
                dh.ε_q[:, iqx], dh.V_q[:, :, iqx] = eigen(H)
            end
        else
            if dh.ishermitian
                ps, info = partialschur(dense_linear_map(Hermitian(H)); nev, which=:LM)
                verbose && @show info
                dh.V_q[:, :, iqx] = ps.Q
                dh.ε_q[:, iqx] = inv.(real.(ps.eigenvalues)) # invert back
            else
                ps, info = partialschur(dense_linear_map(H); nev, which=:LM)
                verbose && @show info
                ε, dh.V_q[:, :, iqx] = partialeigen(ps)
                ε .= inv.(ε)
                reverse!(ε) # we want final eigenvalues in ascending order (by abs)
                dh.ε_q[:, iqx] = ε
            end
        end
    end
    H_diag .= H_diag_copy # restore initial values
end

"""
Calculate Wannier states using the energy eigenstates `targetlevels`. The vector `targetlevels` will be saved in `dh`.
`dh` is assumed to have been diagonalised, without quasimomentum.
Implemented for the 1-component case only.
"""
function compute_wanniers!(dh::DenseHamiltonian1D{R,T,S}; targetlevels::AbstractVector{<:Integer}) where {R<:Real, T<:Number, S<:Number}
    dh.wanniers.targetlevels = targetlevels # store the target levels
    minlevel = targetlevels[1]
    if dh.isperiodic
        X = @view(dh.V[2:end, targetlevels])' * @view(dh.V[1:end-1, targetlevels])
        pos_complex, dh.wanniers.V = eigen(X)
        pos_real = @. mod2pi(angle(pos_complex))/2π * dh.Lx + dh.xlims[1] # `mod2pi` converts the angle from [-π, π) to [0, 2π)
        sp = sortperm(pos_real)               # sort the eigenvalues
        dh.wanniers.pos = pos_real[sp]
        Base.permutecols!!(dh.wanniers.V, sp) # sort the eigenvectors in the same way
    else 
        n_w = length(targetlevels)
        X = Matrix{R}(undef, n_w, n_w) # position operator, will fill only upper triangle
        nj = size(dh.V, 1)
        for n in 1:n_w
            for n′ in 1:n
                X[n′, n] = (n == n′ ? dh.Lx/2 + dh.xlims[1] : 0) - 8dh.Lx/π^2*sum(dh.V[j, minlevel+n-1] * sum(dh.V[j′, minlevel+n′-1]*j*j′/(j^2-j′^2)^2
                                                                                  for j′ = (iseven(j) ? 1 : 2):2:nj) for j = 1:nj)
            end
        end
        dh.wanniers.pos, dh.wanniers.V = eigen(Hermitian(X))
    end
    dh.wanniers.E = transpose(dh.ε[targetlevels]) * abs2.(dh.wanniers.V) |> vec
end

"""
Construct Wannier functions `w` on a grid having `nx` points in `x` direction. All Wannier functions contained in `dh` are constructed.
In the process, energy eigenfunctions `ψ` are also constructed.
Return (`xs`, `ψ`, `w`).
This assumes that wanniers have been calculated; and this is only implemented for the 1-component case.
"""
function make_wannierfunctions(dh::DenseHamiltonian1D; nx::Integer)
    xs, ψ = make_eigenfunctions(dh; statenos=dh.wanniers.targetlevels, nx)
    w = dropdims(ψ; dims=2) * dh.wanniers.V # drop the dimesion corresponding to the component number
    return xs, ψ, w
end

"""
Given complex coordinate-space wanniers `w`, which are actually purely real or purely imaginary,
construct a real array by extracting either the real or imaginary part, whichever is larger.
"""
function make_wanniers_real(w)
    w_real = real(w)
    w_imag = imag(w)
    w_result = similar(w_real)
    for i in axes(w, 2)
        if sum(abs, @view(w_real[:, i])) > sum(abs, @view(w_imag[:, i]))
            w_result[:, i] .= w_real[:, i] # can be optimised using `copyto!`
        else
            w_result[:, i] .= w_imag[:, i]
        end
    end
    return w_result
end

"Compute tunnelling element ⟨𝑤ᵢ|𝐻|𝑤ⱼ⟩."
function compute_tunneling(dh::DenseHamiltonian1D; i::Integer=1, j::Integer=2)
    wᵢ = dh.V[:, dh.wanniers.targetlevels] * dh.wanniers.V[:, i] # one wannier basis vector |𝑤ᵢ⟩ = ∑ₚ |𝜓ₚ⟩ 𝑉ᵢₚ
    wⱼ = dh.V[:, dh.wanniers.targetlevels] * dh.wanniers.V[:, j]
    return dot(wᵢ, dh.H, wⱼ)
end

"Compute TB Hamiltonian matrix, with elements ⟨𝑤ᵢ|𝐻|𝑤ⱼ⟩."
function compute_tb_hamiltonian(dh::DenseHamiltonian1D)
    dh.wanniers.V' * dh.V[:, dh.wanniers.targetlevels]' * dh.H * dh.V[:, dh.wanniers.targetlevels] * dh.wanniers.V
end

# "Return momentum-space matrix of a function `𝑓`, with problem geometry contained in `dh`. `iseven` is only relevant for periodic case, yielding real result for even `𝑓`."
# function p_space_matrix(dh::DenseHamiltonian1D; 𝑓::Function, iseven::Bool=false)
#     (;M, Lx, xlims, isperiodic) = dh

#     if isperiodic
#         N = 4M # number of points for FFT. This will yield harmonics from -2M to 2M
#         dx = Lx/N
#         xs = range(xlims[1], xlims[2]-dx, N)
#         u = 𝑓.(xs) .* dx/Lx
#         return dft_to_matrix_1D(FFTW.rfft(u), iseven)
#     else # non-periodic
#         N = 2M + 1
#         xs = range(xlims[1], xlims[2], N)
#         dx = xs[2] - xs[1]
#         u = 𝑓.(xs) .* dx/Lx
#         return dct_to_matrix_1D(FFTW.r2r!(u, FFTW.REDFT00))
#     end
# end