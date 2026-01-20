"""
A type representing a spatial [𝑟 = (𝑥, 𝑦)], 𝑛-component, possibly quasimomentum-dependent Hamiltonian (𝐻ᵢⱼ)
    𝐻ᵢᵢ(𝑟) = [-i𝛿∇ + 𝑞 - 𝐴ᵢ(𝑟)]² + 𝑈ᵢᵢ(𝑟)
    𝐻ᵢⱼ(𝑟) = 𝑈ᵢⱼ(𝑟)
as a dense matrix. Here 1 ≤ 𝑖, 𝑗 ≤ 𝑛.
"""
mutable struct DenseHamiltonian{R<:AbstractFloat,T<:Number,S<:Number,D1,D2} <: XSpaceHamiltonian{:dense} # in practice `T` shoudld be `R` or `Complex{R}` (and same for `S`) -- always check this. If this is not the case, probably your 𝑈 or 𝐴 do not return R's.
    xlims::Vector{Tuple{R, R}}
    L::Vector{R}
    M::Int # maximum harmonic number (will use -M:M for periodic, 1:M for nonperiodic)
    δ::R # coefficient of the momentum term: -iδ∇ (same for all components)
    nc::Int # number of components
    basis::Symbol
    ishermitian::Bool # `H` is nonhermitian if decays Γ are present
    𝑈::Matrix{<:Union{Function,Nothing}} # nc-component matrix containing coordinate-space potentials and couplings
    𝑈_iseven::BitMatrix # nc-component matrix indicating if 𝑈ᵢⱼ is an even function 𝑈ᵢⱼ(𝑟) = 𝑈ᵢⱼ(-𝑟)
    𝐴::Matrix{<:Union{Function,Nothing}}
    Γ::Vector{R} # decay rates
    H::Matrix{T} # momentum-space Hamiltonian used for diagonalisation
    ε::Vector{S} # eigenvalues, can be complex for nonhermitian `H`, hence additional type `S`
    V::Matrix{T} # eigenvectors matrix
    ε_q::Array{S,D1} # ε_q[n, iqx, iqy] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    V_q::Array{T,D2} # V_q[:, n, iqx, iqy] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)
    wanniers::Wanniers{R} # wanniers are implemented only for the case of 1-component and 1D
end

"""
Construct a `DenseHamiltonian` object using the coordinate-space functions stored in `𝑈`, decay rates `Γ`, and gauge fields stored in `𝐴`. `𝐴[c, i]` is the ith projection `𝐴ᵢ` of cth component.
`M` is the maximum harmonic number. In the cis case, the Hamiltonian will be `nc*(2M+1)²`-by-`nc*(2M+1)²` where `nc` is the number of components.
In sin/cos case, the size will be `nc*M²`-by-`nc*M²`.
`𝑈_iseven[i, j]` matters only if `basis=:cis` and shows whether `𝑈[i, j]` is an even function (i.e. whether 𝑢(𝑟) = 𝑢(-𝑟)). If it is, then Fourier transform is real, which is used for better accuracy.
If *all* functions are even (and real), then the resulting Fourier-space Hamiltonian is real (provided also there is no 𝐴 and Γ), giving a speed-up and better accuracy (compared to complex diagonalisation).
If `𝑈[i, j] === nothing` or it is complex, then the value of `𝑈_iseven[i, j]` does not matter.
Currently it is assumed that if 𝐴's are present, then Hamiltonian is necessarily complex, but this is not true in general (it is real in the cis basis if A is real-even).
"""
function DenseHamiltonian(xlims::AbstractVector{Tuple{R,R}},
                          𝑈::AbstractMatrix{<:Union{Function,Nothing}},
                          𝐴::AbstractVecOrMat{<:Union{Function,Nothing}}=fill(nothing, size(𝑈, 1), length(xlims));
                          basis::Symbol, M::Integer, δ::R=one(R),
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
    H = zeros(T, nc*B, nc*B)

    ft = FourierTransformer(xlims, M; basis, target_real=U_isreal) # `target_real` will allocate a buffer for the imaginary part of the sin/cos-transform if some of 𝑈's are complex

    𝑈_diag_allequal = allequal(diagview(𝑈))
    𝐴ᵢ_allequal = [allequal(𝐴ᵢ) && !isnothing(𝐴ᵢ[1]) for 𝐴ᵢ in eachcol(𝐴)] # 𝐴ᵢ_allequal[i] shows if projection 𝐴ᵢ is the same for all components; note that this also checks if they are nothing

    makereal = (basis == :cis && H_isreal) # in this case the transform is actually real, but is stored in a complex array `ft.buff`; this will be passed to `fft_to_matrix` to drop imaginary part of `ft.buff`

    # treat diagonal blocks, adding the diagonal potentials 𝑈ᵢᵢ and 𝑝² (conditionally)
    for jH in 1:nc
        h = @view H[(jH-1)*B+1:jH*B, (jH-1)*B+1:jH*B] # a view of the `jH`th diagonal block
        h_set = false # shows if `h` has been set to something (i.e. etiher/both next two if's have been entered)
        if !isnothing(𝑈[jH, jH])
            transform!(ft, 𝑈[jH, jH])
            fft_to_matrix!(h, ft; makereal)
            h_set = true
            # @debug "Wrote 𝑈[$jH, $jH] into H[$jH, $jH]" # H[iH, jH] schematically means the block (`iH`, `jH`)
        end
        # Add 𝑝² if basis is sin/cos. But if there are no 𝐴's at all, add in the cis case too (if 𝐴's are present, then 𝑝ᵢ²'s will be added together with 𝐴ᵢ's)
        if basis != :cis || all(𝐴ᵢ_present .== false)
            h .+= make_p²(L, M, δ, basis)
            h_set = true
            # @debug "Added 𝑝² to H[$jH, $jH]"
        end
        # If all 𝑈 are equal, then copy the just-calculated first diagonal block into all other diagonal blocks, and break.
        # This can be triggered on the first iteration only, and only if 𝑈's are not all nothing
        if 𝑈_diag_allequal && h_set
            for iH in 2:nc
                copyto!(H, CartesianIndices(((iH-1)*B+1:iH*B, (iH-1)*B+1:iH*B)), h, CartesianIndices(h))
                # @debug "Copied H[1, 1] to H[$iH, $iH]"
            end
            break
        end
    end

    # treat diagonal blocks, adding the kinetic terms (𝑝ᵢ - 𝐴ᵢ)²
    if any(𝐴ᵢ_present)
        A_buff = Matrix{T}(undef, B, B)
        A_buff2 = similar(A_buff)
        for i in 1:D # iterate over projections of 𝐴
            if !𝐴ᵢ_present[i] && basis != :cis # if the projection 𝐴ᵢ is zero for all components, then skip 𝐴ᵢ. However, if basis is cis, we cannot skip because also need to add 𝑝ᵢ²
                continue
            end
            pᵢ = make_p_i(L, M, δ, basis, i)
            for c in 1:nc
                if isnothing(𝐴[c, i]) # then there is nothing to do, except adding 𝑝ᵢ² in the cis case
                    if basis == :cis
                        pᵢ .^= 2 # in-place squaring
                        H[(c-1)*B+1:c*B, (c-1)*B+1:c*B] .+= pᵢ
                        # @debug "Added p_$i^2 to H[$c, $c]"
                    end
                    continue
                end
                transform!(ft, 𝐴[c, i])
                fft_to_matrix!(A_buff, ft)

                if basis == :cis
                    A_buff .= pᵢ .- A_buff
                    mul!(A_buff2, A_buff, A_buff) # after this multiplication, `A_buff2` contains (𝑝ᵢ - 𝐴ᵢ)²
                else
                    A_buff2 .= im*(A_buff*pᵢ + pᵢ*A_buff) + A_buff^2 # The perfect square for (𝑝ᵢ - 𝐴ᵢ)² is much less accurate. TODO optimise multiplications
                end
                H[(c-1)*B+1:c*B, (c-1)*B+1:c*B] .+= A_buff2 # add to the curent block
                # @debug begin basis == :cis ? "Added (p_$i - 𝐴[$c, $i])^2 to H[$c, $c]" : "Added im(𝐴[$c, $i]*p_$i + p_$i*𝐴[$c, $i] + 𝐴[$c, $i]^2) to H[$c, $c]"; end
                if 𝐴ᵢ_allequal[i] # then add `A_buff2` to all other diagonal blocks and break
                    for iH in 2:nc
                        H[(iH-1)*B+1:iH*B, (iH-1)*B+1:iH*B] .+= A_buff2
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
            H[diagind(H)[(c-1)*B+1:c*B]] .-= im*Γ[c]/2
        end
    end
    # treat off-diagonal blocks (will not be run for a single component)
    for jH in 2:nc
        for iH in 1:jH-1 # only upper triangle is scanned. The lower triangle is filled only if Γ is present
            isnothing(𝑈[iH, jH]) && continue
            transform!(ft, 𝑈[iH, jH])
            wi = (iH-1)*B+1:iH*B
            wj = (jH-1)*B+1:jH*B
            h = @view H[wi, wj] # a view of the required block
            fft_to_matrix!(h, ft; makereal)
            # @debug "Wrote 𝑈[$iH, $jH] into H[$iH, $jH]"

            if !iszero(Γ) # fill conjugate block if Γ is present (then we cannot use Hermitian view)
                H[wj, wi] .= h' # TODO possible to use `copyto!` ?
                # @debug "Copied H[$iH, $jH]' into H[$jH, $iH]"
            end
        end
    end

    # determine the type of eigenvalues 
    ishermitian = iszero(Γ) # if all `Γ`s are zeros, then Hamiltonian is Hermitian and the eigenvalues real
    S = ishermitian ? R : Complex{R} # type of eigenvalues
    
    # create empty placeholders
    ε = S[] # eigenvalues, can be complex for nonhermitian `H`, hence additional type `S`
    V = T[;;] # eigenvectors matrix
    ε_q = Array{S}(undef, ntuple(Returns(0), D+1)) # ε_q[n, iqx, iqy, ...] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    V_q = Array{T}(undef, ntuple(Returns(0), D+2)) # V_q[:, n, iqx, iqy, ...] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)

    return DenseHamiltonian(xlims, L, M, δ, nc, basis, ishermitian, 𝑈, BitMatrix(𝑈_iseven), 𝐴, Γ, H, ε, V, ε_q, V_q, Wanniers{R}())
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
Calculate eigenenergies for all quasimomenta in `qs = [qxs, qys, ...]` where `qxs` are 𝑞's along 𝑥, etc.
Calculate `nev` lowest levels using `ArnoldiMethod`.
Pass `nev=0` for full diagonalisation using `LinearAlgebra`.
Note that `dh.H` is modified in the process.
"""
function diagonalize!(dh::DenseHamiltonian{R,T,S,D1,D2}, qs::AbstractVector{<:AbstractVector{<:Real}}; nev::Integer, verbose::Bool=false) where {R<:AbstractFloat,T<:Number,S<:Number,D1,D2}
    if dh.basis != :cis
        @warn "Hamiltonian must be periodic. Construct a new one using the cis basis and try again."
        return
    end
    (;M, xlims, L, δ, nc, H, 𝑈, 𝑈_iseven, 𝐴, Γ) = dh
    D = length(xlims)

    B = (2M + 1)^D # block size
    nsaves = nev == 0 ? B : nev # number of eigenvalues and eigenvectors to allocate
    dh.ε_q = Array{S}(undef, nsaves, ntuple(i -> length(qs[i]), D)...) # ε_q[n, iqx, iqy, ...] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    dh.V_q = Array{T}(undef, B*nc, nsaves, ntuple(i -> length(qs[i]), D)...) # V_q[:, n, iqx, iqy, ...] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)
    
    if all(isnothing.(𝐴)) # very simple case (with no 𝐴) that we can treat separately
        H_diag = diagview(dh.H)
        # from the diagonal of each diagonal block of `H`, extract (𝑈ᵢᵢ)₀ (the 0th harmonic of 𝑈ᵢᵢ) plus decay -iΓ/2
        U_diags = [H_diag[(c-1)B + B÷2+1] for c in 1:nc] # generally, `Hᵢᵢ = -Δᵢᵢ + Uᵢᵢ - iΓ/2`, but Δᵢᵢ = 0 for the central element of the diagonal
    else # the general case with 𝐴
        # two buffers that we will need in the q-loop for matrix multiplication
        buff1 = Matrix{T}(undef, B, B)
        buff2 = Matrix{T}(undef, B, B)

        ft = FourierTransformer(xlims, M; basis=:cis)

        K = Union{Matrix{T}, Diagonal{T, Vector{T}}, Nothing}[nothing for _ in CartesianIndices(𝐴)] # Matrix of dimensions like 𝐴 for storing corresponding kinetic operators -iδ∂ᵢ - 𝐴ᵢ
        U = Union{Matrix{T}, Nothing}[nothing for _ in axes(𝑈, 1)] # for storing terms 𝑈ᵢᵢ

        𝑈_diag_allequal = allequal(diagview(𝑈))
        𝐴ᵢ_allequal = [allequal(𝐴ᵢ) for 𝐴ᵢ in eachcol(𝐴)] # 𝐴ᵢ_allequal[i] shows if projection 𝐴ᵢ is the same for all components; they may all be nothing

        # fill the buffers `U`
        for c in 1:nc
            if !isnothing(𝑈[c, c])
                transform!(ft, 𝑈[c, c])
                U[c] = fft_to_matrix(ft)
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
                    K[c, i] = fft_to_matrix(ft)
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
                H_block = @view H[(c-1)*B+1:c*B, (c-1)*B+1:c*B]
                for i in 1:D
                    which_K = 𝐴ᵢ_allequal[i] ? 1 : c
                    copy!(buff1, K[which_K, i])
                    buff1 += LA.I*QS[i]
                    mul!(buff2, buff1, buff1)
                    if i == 1
                        copyto!(H, CartesianIndices(((c-1)*B+1:c*B, (c-1)*B+1:c*B)), buff2, CartesianIndices(buff2))
                        @debug "Copied (K[$which_K, $i] + QS[$i])^2 into H[$c, $c]"
                    else
                        H_block .+= buff2
                        @debug "Added (K[$which_K, $i] + QS[$i])^2 to H[$c, $c]"
                    end
                end

                if 𝑈_diag_allequal
                    H_block .+= U[1]
                    @debug "Added U[1] to H[$c, $c]"
                elseif !isnothing(U[c])
                    H_block .+= U[c]
                    @debug "Added U[$c] to H[$c, $c]"
                end
                if Γ[c] != 0
                    H_block -= LA.I * im*Γ[c]/2
                    @debug "Added -im*Γ[$c]/2 to H[$c, $c]"
                end
            end
        end

        dh.ε_q[:, IQ...], dh.V_q[:, :, IQ...] = diagonalize(dh; nev, verbose)
    end
end