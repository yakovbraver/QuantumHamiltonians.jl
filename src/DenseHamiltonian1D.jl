"""
A type representing a spatial, 𝑛-component, possibly quasimomentum-dependent Hamiltonian (𝐻ᵢⱼ)
    𝐻ᵢᵢ(𝑥) = (-i𝛿∂ₓ + 𝑞)² + 𝑈ᵢᵢ(𝑥)
    𝐻ᵢⱼ(𝑟) = 𝑈ᵢⱼ(𝑟)
as a dense matrix.
All 𝑈ᵢⱼ(𝑟) are assumed real (contrary to the 2D case).
"""
mutable struct DenseHamiltonian1D{R<:Real,T<:Number,S<:Number} <: XSpaceHamiltonian{:dense} # in practice `T` shoudld be `R` if there is no 𝑞 and 𝑈 is even, or `Complex{R}` otherwise -- always check this. If this is not the case, probably your 𝑈 or 𝐴 do not return R's.
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
Calculate eigenenergies for all quasimomenta in `qxs`.
Calculate `nev` lowest bands using `ArnoldiMethod`.
If `nev=0` or not passed, then full diagonalisation using `LinearAlgebra` is performed.
"""
function diagonalize!(dh::DenseHamiltonian1D{R,T,S}, qxs::AbstractVector{<:Real}; nev::Integer, verbose::Bool=false) where {R<:Real, T<:Number, S<:Number}
    (;M, Lx, δ, nc) = dh
   
    B = 2M + 1 # block size
    nsaves = nev == 0 ? B*nc : nev # number of eigenvalues and eigenvectors to allocate
    dh.ε_q = Array{S,2}(undef, nsaves, length(qxs))
    dh.V_q = Array{T,3}(undef, B*nc, nsaves, length(qxs))
    
    H_diag = diagview(dh.H)
    H_diag_copy = diag(dh.H) # a copy for restoring after the calculation
    # from the diagonal of each diagonal block of `H`, extract the 0th harmonic of 𝑈ᵢᵢ plus decay -iΓ/2
    U_diags = [H_diag[(c-1)B + B÷2+1] for c in 1:nc] # generally, `H₀₀ = -Δ₀₀ + U₀₀ - iΓ/2`, but Δ₀₀ = 0 for the central element of the diagonal (see construction of Δ in `DenseHamiltonian1D` constructor)
    
    # update diagonal blocks and diagonalise
    for (iqx, qx) in enumerate(qxs)
        # update diagonal
        for c in 1:nc
            H_diag[(c-1)B+1:c*B] .= [(2π*δ*jx/Lx + qx)^2 + U_diags[c] for jx in -M:M]
        end

        dh.ε_q[:, iqx], dh.V_q[:, :, iqx] = diagonalize(dh; nev, verbose)
    end
    H_diag .= H_diag_copy # restore initial values
end