"""
A type representing a spatial [𝑟 = (𝑥, 𝑦)], 𝑛-component, possibly quasimomentum-dependent Hamiltonian (𝐻ᵢⱼ)
    𝐻ᵢᵢ(𝑟) = [-i𝛿∇ + 𝑞 - 𝐴(𝑟)]² + 𝑈ᵢᵢ(𝑟)
    𝐻ᵢⱼ(𝑟) = 𝑈ᵢⱼ(𝑟)
as a sparse matrix.
"""
mutable struct SparseHamiltonian{R<:AbstractFloat,T<:Number,S<:Number,D1,D2} <: XSpaceHamiltonian{:sparse} # in practice `T` shoudld be `R` or `Complex{R}` (and same for `S`) -- always check this. If this is not the case, probably your 𝑈 or 𝐴 do not return R's.
    xlims::Vector{Tuple{R, R}}
    L::Vector{R}
    M::Int # maximum harmonic number (will use -M:M for periodic, 1:M for nonperiodic)
    fft_threshold::R # values lower in magnitude will be filtered out
    δ::R # coefficient of the momentum term: -iδ∇
    nc::Int # number of components
    basis::Symbol
    ishermitian::Bool # `H` is nonhermitian if decays Γ are present
    𝑈::Matrix{<:Union{Function,Nothing}} # nc-component matrix containing coordinate-space potentials and couplings
    𝑈_iseven::BitMatrix # nc-component matrix indicating if 𝑈ᵢⱼ is an even function 𝑈ᵢⱼ(𝑟) = 𝑈ᵢⱼ(-𝑟)
    𝐴::Matrix{<:Union{Function,Nothing}}
    Γ::Vector{R} # decay rates
    H::SparseMatrixCSC{T, Int64} # momentum-space Hamiltonian used for diagonalisation (UMFPACKFactorization only supports Int64-type indices)
    H_blocks::Matrix{SparseMatrixCSC{T, Int64}} # a matrix of blocks of `H`, used for quasimomentum diagonalisation
    ε::Vector{S} # eigenvalues, can be complex for nonhermitian `H`, hence additional type `S`
    V::Matrix{T} # eigenvectors matrix
    ε_q::Array{S,D1} # ε_q[n, iqx, iqy] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    V_q::Array{T,D2} # V_q[:, n, iqx, iqy] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)
    wanniers::Wanniers{R} # wanniers are implemented only for the case of 1-component and 1D
end

"""
Construct a `SparseHamiltonian` object using the coordinate-space functions stored in `𝑈`, decay rates `Γ`, and gauge field (same for all components) 𝐴_x, 𝐴_y.
`M` is the maximum harmonic number. In the periodic case, the Hamiltonian will be `nc*(2M+1)²`-by-`nc*(2M+1)²` where `nc` is the number of components.
In nonperiodic case, the size will be `nc*M²`-by-`nc*M²`.
`𝑈_iseven[i, j]` matters only if `isperiodic=true` and shows whether `𝑈[i, j]` is an even function (i.e. whether 𝑢(𝑥, 𝑦) = 𝑢(-𝑥, -𝑦)). If it is, then Fourier transform is real, which is used for better accuracy.
If *all* functions are even (and real), then the resulting Fourier-space Hamiltonian is real (provided also there is no 𝐴 and Γ), giving a speed-up and better accuracy (compared to complex diagonalisation).
If `𝑈[i, j] === nothing` or it is complex, then the value of `𝑈_iseven[i, j]` does not matter.
"""
function SparseHamiltonian(xlims::AbstractVector{Tuple{R,R}},
                           𝑈::AbstractMatrix{<:Union{Function,Nothing}},
                           𝐴::AbstractVecOrMat{<:Union{Function,Nothing}}=fill(nothing, size(𝑈, 1), length(xlims));
                           basis::Symbol, M::Integer, δ::R=one(R), fft_threshold::R=√eps(R),
                           𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1))) where R <: AbstractFloat
    nc = size(𝑈, 1) # number of components
    D = length(xlims) # number of spatial dimensions
    L = [lims[2] - lims[1] for lims in xlims]

    # `H_isreal` will show if the resulting `H` will be real
    𝐴ᵢ_present = [any(𝐴ᵢ .!== nothing) for 𝐴ᵢ in eachcol(𝐴)] # `i` numbers projections; 𝐴ᵢ_present[i] = true if 𝐴ᵢ is nonzero for at least one component
    U_isreal = all( 𝑢([xlims[i][1] for i in eachindex(xlims)]...) isa Real for 𝑢 in 𝑈 if !isnothing(𝑢) ) # check if all functions in 𝑈 are real
    H_isreal = U_isreal && all(𝐴ᵢ_present .== false) && iszero(Γ) # without checking we assume that all 𝐴's are real. Can be generalised for the exotic cases of complex 𝐴.
    if basis == :cis # for periodic potential, also check if functions are even 
        H_isreal &= all(𝑈_iseven[𝑈 .!== nothing])
    end

    B = basis == :cis ? (2M+1)^D : M^D # size of each Hamiltonian block

    T = H_isreal ? R : Complex{R} # type of elements of the Hamiltonian
    H_blocks = Matrix{SparseMatrixCSC{T, Int64}}(undef, nc, nc) # temporary Hamiltonian as an `nc`-by-`nc` matrix of sparse blocks

    ft = FourierTransformer(xlims, M; basis, target_real=U_isreal) # `target_real` will allocate a buffer for the imaginary part of the sin/cos-transform if some of 𝑈's are complex

    𝑈_diag_allequal = allequal(diagview(𝑈))
    𝐴ᵢ_allequal = [allequal(𝐴ᵢ) && !isnothing(𝐴ᵢ[1]) for 𝐴ᵢ in eachcol(𝐴)] # 𝐴ᵢ_allequal[i] shows if projection 𝐴ᵢ is the same for all components; note that this also checks if they are nothing

    makereal = (basis == :cis && H_isreal) # in this case the transform is actually real, but is stored in a complex array `ft.buff`; this will be passed to `fft_to_matrix` to drop imaginary part of `ft.buff`

    # treat diagonal blocks, adding the diagonal potentials 𝑈ᵢᵢ and 𝑝² (conditionally)
    for jH in 1:nc
        if isnothing(𝑈[jH, jH])
            H_blocks[jH, jH] = spzeros(T, Int64, B, B)
            # @debug "Set H[$jH, $jH] to spzero"
        else
            transform!(ft, 𝑈[jH, jH])
            H_blocks[jH, jH] = fft_to_matrix(ft; makesparse=true, makereal, threshold=fft_threshold)
            # @debug "Wrote 𝑈[$jH, $jH] into H[$jH, $jH]" # H[iH, jH] schematically means the block (`iH`, `jH`)
        end
        # Add 𝑝² if basis is sin/cos. But if there are no 𝐴's at all, add in the cis case too (if 𝐴's are present, then 𝑝ᵢ²'s will be added together with 𝐴ᵢ's)
        if basis != :cis || all(𝐴ᵢ_present .== false)
            H_blocks[jH, jH] .+= make_p²(L, M, δ, basis)
            # @debug "Added 𝑝² to H[$jH, $jH]"
        end
        # If all 𝑈 are equal, then pass the reference of the just-calculated first diagonal block into all other diagonal blocks, and break.
        # This can be triggered on the first iteration only, and only if 𝑈's are not all nothing
        if 𝑈_diag_allequal
            for iH in 2:nc
                H_blocks[iH, iH] = H_blocks[jH, jH] # a reference, not a copy!
                # @debug "Copied H[1, 1] to H[$iH, $iH]"
            end
            break
        end
    end

    # treat diagonal blocks, adding the kinetic terms (𝑝ᵢ - 𝐴ᵢ)²
    if any(𝐴ᵢ_present)
        for i in 1:D # iterate over projections of 𝐴
            if !𝐴ᵢ_present[i] && basis != :cis # if the projection 𝐴ᵢ is zero for all components, then skip 𝐴ᵢ. However, if basis is cis, we cannot skip because also need to add 𝑝ᵢ²
                continue
            end
            pᵢ = make_p_i(L, M, δ, basis, i)
            for c in 1:nc
                if isnothing(𝐴[c, i]) # then there is nothing to do, except adding 𝑝ᵢ² in the cis case
                    if basis == :cis
                        pᵢ .^= 2 # in-place squaring
                        H_blocks[c, c] .+= pᵢ
                        # @debug "Added p_$i^2 to H[$c, $c]"
                    end
                    continue
                end
                transform!(ft, 𝐴[c, i])
                A_buff = fft_to_matrix(ft; makesparse=true, threshold=fft_threshold) # contrary to the dense case, an in-place `fft_to_matrix` is impossible

                if basis == :cis
                    A_buff .= pᵢ .- A_buff
                    A_buff2 = A_buff^2 # after this multiplication, `A_buff2` contains (𝑝ᵢ - 𝐴ᵢ)².
                else
                    A_buff2 = im*(A_buff*pᵢ + pᵢ*A_buff) + A_buff^2 # The perfect square for (𝑝ᵢ - 𝐴ᵢ)² is much less accurate. TODO optimise multiplications
                end
                H_blocks[c, c] += A_buff2 # add to the curent block
                # @debug begin basis == :cis ? "Added (p_$i - 𝐴[$c, $i])^2 to H[$c, $c]" : "Added im(𝐴[$c, $i]*p_$i + p_$i*𝐴[$c, $i] + 𝐴[$c, $i]^2) to H[$c, $c]"; end
                if 𝐴ᵢ_allequal[i] # then add `A_buff2` to all other diagonal blocks and break
                    for iH in 2:nc
                        H_blocks[iH, iH] .+= A_buff2
                        # @debug begin basis == :cis ? "Added (p_$i - 𝐴[$c, $i])^2 to H[$iH, $iH]" : "Added im(𝐴[$c, $i]*p_$i + p_$i*𝐴[$c, $i] + 𝐴[$c, $i]^2) to H[$iH, $iH]"; end
                    end
                    break
                end
            end
        end
    end
    # add -iΓ/2
    for c in 1:nc
        if Γ[c] != 0
            H_blocks[c, c] -= im*Γ[c]/2 * LA.I
        end
    end
    # treat off-diagonal blocks (will not be run for a single component)
    for jH in 2:nc
        for iH in 1:jH-1 # only upper triangle is scanned. The lower triangle is filled only if Γ is present
            if isnothing(𝑈[iH, jH])
                H_blocks[iH, jH] = spzeros(T, Int64, B, B)
                # @debug "Set H[$jH, $jH] to spzero"
            else
                transform!(ft, 𝑈[iH, jH])
                H_blocks[iH, jH] = fft_to_matrix(ft; makesparse=true, makereal, threshold=fft_threshold)
                # @debug "Wrote 𝑈[$iH, $jH] into H[$iH, $jH]"
            end
            H_blocks[jH, iH] = H_blocks[iH, jH]' # set the conjugate block
            # Could potentially be avoided if iszero(Γ) because then we could use a Hermitian view.
            # But factorisation is LU anyway, so a non-hermitian workspace is needed. 
        end
    end
    
    H = nc == 1 ? H_blocks[1] : hvcat(nc, transpose(H_blocks)...) # construct the final Hamiltonian; in the 1-component case just reference the only existing block

    # determine the type of eigenvalues
    ishermitian = iszero(Γ) # if all `Γ`s are zeros, then Hamiltonian is Hermitian and the eigenvalues real
    S = ishermitian ? R : Complex{R} # type of eigenvalues

    # create empty placeholders
    ε = S[] # eigenvalues, can be complex for nonhermitian `H`, hence additional type `S`
    V = T[;;] # eigenvectors matrix
    ε_q = Array{S}(undef, ntuple(Returns(0), D+1)) # ε_q[n, iqx, iqy, ...] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    V_q = Array{T}(undef, ntuple(Returns(0), D+2)) # V_q[:, n, iqx, iqy, ...] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)

    return SparseHamiltonian(xlims, L, M, fft_threshold, δ, nc, basis, ishermitian, 𝑈, BitMatrix(𝑈_iseven), 𝐴, Γ, H, H_blocks, ε, V, ε_q, V_q, Wanniers{R}())
end

"Return the filling density of the Hamiltonian matrix."
function matrix_density(xh::SparseHamiltonian)
    nnz(xh.H) / prod(size(xh.H))
end

"""
Calculate eigenenergies for all quasimomenta in `qs = [qxs, qys, ...]` where `qxs` are 𝑞's along 𝑥, etc.
Calculate `nev` lowest levels using `ArnoldiMethod`.
Pass `nev=0` for full diagonalisation using `LinearAlgebra`.
Note that `xh.H` is modified in the process.
"""
function diagonalize!(xh::SparseHamiltonian{R,T,S,D1,D2}, qs::AbstractVector{<:AbstractVector{<:Real}}; nev::Integer, verbose::Bool=false) where {R<:AbstractFloat,T<:Number,S<:Number,D1,D2}
    if xh.basis != :cis
        @warn "Hamiltonian must be periodic. Construct a new one using the cis basis and try again."
        return
    end
    (;M, xlims, L, fft_threshold, δ, nc, H, H_blocks, 𝑈, 𝑈_iseven, 𝐴, Γ) = xh
    D = length(xlims)

    B = (2M + 1)^D # block size
    nsaves = nev == 0 ? B : nev # number of eigenvalues and eigenvectors to allocate
    xh.ε_q = Array{S}(undef, nsaves, ntuple(i -> length(qs[i]), D)...) # ε_q[n, iqx, iqy, ...] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    xh.V_q = Array{T}(undef, B*nc, nsaves, ntuple(i -> length(qs[i]), D)...) # V_q[:, n, iqx, iqy, ...] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)
    
    if all(isnothing.(𝐴)) # very simple case (with no 𝐴) that we can treat separately
        H_diag = diagview(xh.H)
        # from the diagonal of each diagonal block of `H`, extract (𝑈ᵢᵢ)₀ (the 0th harmonic of 𝑈ᵢᵢ) plus decay -iΓ/2
        U_diags = [H_diag[(c-1)B + B÷2+1] for c in 1:nc] # generally, `Hᵢᵢ = -Δᵢᵢ + Uᵢᵢ - iΓ/2`, but Δᵢᵢ = 0 for the central element of the diagonal
    else # the general case with 𝐴
        ft = FourierTransformer(xlims, M; basis=:cis)

        K = Union{typeof(H), Diagonal{T, Vector{T}}, Nothing}[nothing for _ in CartesianIndices(𝐴)] # Matrix of dimensions like 𝐴 for storing corresponding kinetic operators -iδ∂ᵢ - 𝐴ᵢ
        U = Union{typeof(H), Nothing}[nothing for _ in axes(𝑈, 1)] # for storing terms 𝑈ᵢᵢ

        𝑈_diag_allequal = allequal(diagview(𝑈))
        𝐴ᵢ_allequal = [allequal(𝐴ᵢ) for 𝐴ᵢ in eachcol(𝐴)] # 𝐴ᵢ_allequal[i] shows if projection 𝐴ᵢ is the same for all components; they may all be nothing

        # fill the buffers `U`
        for c in 1:nc
            if !isnothing(𝑈[c, c])
                transform!(ft, 𝑈[c, c])
                U[c] = fft_to_matrix(ft, makesparse=true, threshold=fft_threshold)
                @debug "Filled U[$c]"
            end
            # If all 𝑈 are equal, then we will be using only U[1], no need to fill other elements
            𝑈_diag_allequal && break
        end

        # fill the buffers `K`
        for i in 1:D # iterate over projections of 𝐴
            pᵢ = make_p_i(L, M, δ, :cis, i)
            for c in 1:nc
                if isnothing(𝐴[c, i]) # then add 𝑝ᵢ
                    K[c, i] = pᵢ
                    @debug "Wrote p_$i to K[$c, $i]"
                else
                    transform!(ft, 𝐴[c, i])
                    K[c, i] = fft_to_matrix(ft, makesparse=true, threshold=fft_threshold)
                    K[c, i] .= pᵢ .- K[c, i]
                    @debug "Wrote p_$i - 𝐴[$c, $i] to K[$c, $i]"
                end
                # If projection 𝐴ᵢ is the same for all components, then we will be using K[1, i] for all components, no need to fill other rows in the i'th column
                𝐴ᵢ_allequal[i] && break
            end
        end
    end

    # update diagonal blocks and diagonalise
    QS = Vector{R}(undef, length(qs)) # at each iteration will contain the values of quasimomenta, e.g. in 2D it will contain [qx, qy], where we defined qx ≡ qs[1], qy ≡ qs[2]
    for IQ in Iterators.product(eachindex.(qs)...) # example in 2D: IQ = (iqx, iqy), where iqx is an index of qx and iqy is an index of qy
        for i in eachindex(QS)
            QS[i] = qs[i][IQ[i]]
        end

        @debug "🚜 QS = $QS 🚜"

        # update diagonal blocks
        if all(isnothing.(𝐴)) # very simple case (with no 𝐴) that we can treat separately
            p² = make_p²(L, M, δ, :cis, QS) |> parent # `parent` returns the diagonal as a vector TODO make in-place
            for c in 1:nc
                H_diag[(c-1)B+1:c*B] .= p² .+ U_diags[c]
            end
        else # the general case with 𝐴
            for c in 1:nc
                for i in 1:D
                    which_K = 𝐴ᵢ_allequal[i] ? 1 : c
                    buff = K[which_K, i] + LA.I*QS[i]
                    buff = buff^2
                    if i == 1
                        H_blocks[c, c] = buff # set reference
                        @debug "Copied (K[$which_K, $i] + QS[$i])^2 into H[$c, $c]"
                    else
                        H_blocks[c, c] += buff
                        @debug "Added (K[$which_K, $i] + QS[$i])^2 to H[$c, $c]"
                    end
                end

                if 𝑈_diag_allequal
                    H_blocks[c, c] += U[1]
                    @debug "Added U[1] to H[$c, $c]"
                elseif !isnothing(U[c])
                    H_blocks[c, c] += U[c]
                    @debug "Added U[$c] to H[$c, $c]"
                end
                if Γ[c] != 0
                    H_blocks[c, c] -= LA.I * im*Γ[c]/2
                    @debug "Added -im*Γ[$c]/2 to H[$c, $c]"
                end
            end
            xh.H = nc == 1 ? H_blocks[1] : hvcat(nc, transpose(H_blocks)...) # construct the final Hamiltonian; in the 1-component case just reference the only existing block
        end

        xh.ε_q[:, IQ...], xh.V_q[:, :, IQ...] = diagonalize(xh; nev, verbose)
    end
end