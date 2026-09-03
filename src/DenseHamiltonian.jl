"""
A type representing a spatial, 𝐷-dimensional, 𝑛-component, possibly quasimomentum-dependent Hamiltonian (𝐻ᵢⱼ)
    𝐻ᵢᵢ(r) = [-i𝛿∇ + q - Aᵢ(r)]² + 𝑈ᵢᵢ(r) - iΓᵢ/2
    𝐻ᵢⱼ(r) = 𝑈ᵢⱼ(r)
as a dense matrix. Here  1 ≤ 𝑖, 𝑗 ≤ 𝑛,  r = (𝑥₁, …, 𝑥_𝐷),  Aᵢ = (𝐴ᵢ₁, …, 𝐴ᵢ_𝐷),  q = (𝑞₁, …, 𝑞_𝐷).
"""
mutable struct DenseHamiltonian{R, T, S, D1, D2, FourierTransformer} <: PSpaceHamiltonian{:dense, R, T, S, D1, D2}
    xlims::Vector{Tuple{R, R}}
    L::Vector{R}
    B::Int # "block size" -- number of points in the contiguous array corresponding to each component. The size of `H` is `B*nc`-by-`B*nc`
    δ::R # coefficient of the momentum term: -iδ∇ (same for all components)
    nc::Int # number of components
    basis::Symbol
    ishermitian::Bool # `H` is nonhermitian if decays Γ are present
    𝑈::Matrix{<:Union{Function, Nothing}} # nc-component matrix containing coordinate-space potentials and couplings. Return type must be R or T
    𝑈_iseven::BitMatrix # nc-component matrix indicating if 𝑈ᵢⱼ is an even function 𝑈ᵢⱼ(r) = 𝑈ᵢⱼ(-r)
    𝐴::Matrix{<:Union{Function, Nothing}} # `𝐴[c, i]` is `i`th projection of the `c`th component of the vector potential
    Γ::Vector{R} # decay rates
    H::Matrix{T} # momentum-space Hamiltonian used for diagonalisation
    ε::Vector{S} # eigenvalues, can be complex for nonhermitian `H`, hence additional type `S`
    V::Matrix{T} # eigenvectors matrix
    ε_q::Array{S,D1} # ε_q[n, iqx, iqy] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    V_q::Array{T,D2} # V_q[:, n, iqx, iqy] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)
    wanniers::Wanniers{R} # wanniers are implemented only for the case of 1-component and 1D
    # transformer and buffers for applying the Hamiltonian to x-space vectors via `mul!`
    ft::FourierTransformer
    uₚ_buff_real::Vector{R}
    uₚ_buff_real2::Vector{R}
    uₚ_buff_complex::Vector{Complex{R}}
    uₚ_buff_complex2::Vector{Complex{R}}
end

"""
Construct a `DenseHamiltonian` object using the coordinate-space functions stored in `𝑈`, decay rates `Γ`, and gauge fields stored in `𝐴`. `𝐴[c, i]` is the `i`th projection `𝐴ᵢ` of cth component.
`M` is the maximum harmonic number. In the cis case, the Hamiltonian will be `nc*(2M+1)²`-by-`nc*(2M+1)²` where `nc` is the number of components. In sin/cos case, the size will be `nc*M²`-by-`nc*M²`.
`𝑈_iseven[i, j]` matters only if `basis=:cis` and shows whether `𝑈[i, j]` is an even function (i.e. whether 𝑢(𝑟) = 𝑢(-𝑟)). If it is, then Fourier transform is real, which is used for better accuracy.
If *all* functions are even (and real), then the resulting Fourier-space Hamiltonian is real (provided also there is no 𝐴 and Γ), giving a speed-up and better accuracy (compared to complex diagonalisation).
If `𝑈[i, j] ≡ nothing` or it is complex, then the value of `𝑈_iseven[i, j]` does not matter.
Currently it is assumed that if 𝐴's are present, then Hamiltonian is necessarily complex, but this is not true in general (it is real in the cis basis if A is real-even, exactly as for 𝑈).
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

    # size of each Hamiltonian block
    B = basis == :cis ? (2M+1)^D :
        basis == :sin ?      M^D : (M+1)^D

    T = H_isreal ? R : Complex{R} # type of elements of the Hamiltonian
    H = zeros(T, nc*B, nc*B)

    ft = FourierTransformerP(xlims, M; basis)

    𝑈_diag_allequal = allequal(diagview(𝑈))
    𝐴ᵢ_allequal = [allequal(𝐴ᵢ) && !isnothing(𝐴ᵢ[1]) for 𝐴ᵢ in eachcol(𝐴)] # 𝐴ᵢ_allequal[i] shows if projection 𝐴ᵢ is the same for all components; note that this also checks if they are nothing

    makereal = (basis == :cis && H_isreal) # in this case the transform is actually real, but is stored in a complex array `ft.buff`; this will be passed to `fft_to_operator` to drop imaginary part of `ft.buff`

    # treat diagonal blocks, adding the diagonal potentials 𝑈ᵢᵢ and 𝑝² (conditionally)
    for jH in 1:nc
        h = @view H[(jH-1)*B+1:jH*B, (jH-1)*B+1:jH*B] # a view of the `jH`th diagonal block
        h_set = false # shows if `h` has been set to something (i.e. etiher/both next two if's have been entered)
        if !isnothing(𝑈[jH, jH])
            transform!(ft, 𝑈[jH, jH])
            fft_to_operator!(h, ft; makereal)
            h_set = true
            # @debug "Wrote 𝑈[$jH, $jH] into H[$jH, $jH]" # H[iH, jH] schematically means the block (`iH`, `jH`)
        end
        # Add 𝑝² if basis is sin/cos. But if there are no 𝐴's at all, add in the cis case too (if 𝐴's are present, then 𝑝ᵢ²'s will be added together with 𝐴ᵢ's)
        if basis != :cis || all(𝐴ᵢ_present .== false)
            h .+= make_p²_matrix(L, M, δ, basis)
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
            pᵢ = make_pⁱ_matrix(L, M, δ, basis, i)
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
                fft_to_operator!(A_buff, ft)

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
            fft_to_operator!(h, ft; makereal)
            # @debug "Wrote 𝑈[$iH, $jH] into H[$iH, $jH]"

            # copy adjoint of H[wi, wj] into H[wj, wi]. Needed for nonhermitian diagonalisation (cannot use Hermitian view) and also for GPE
            copy_adjoint!(H, wj, wi, H, wi, wj)
            # @debug "Copied H[$iH, $jH]' into H[$jH, $iH]"
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

    # buffers for applying the Hamiltonian to x-space vectors
    ft = FourierTransformerP(xlims, M; basis, target_rank=1) # a rank-1 transformer used in `mul!`; it will be stored in the Hamiltonian
    uₚ_buff_real = Vector{R}(undef, nc*B)
    uₚ_buff_real2 = similar(uₚ_buff_real)
    uₚ_buff_complex = similar(uₚ_buff_real, Complex{R})
    uₚ_buff_complex2 = similar(uₚ_buff_complex)

    return DenseHamiltonian(xlims, L, B, δ, nc, basis, ishermitian, 𝑈, BitMatrix(𝑈_iseven), 𝐴, Γ, H, ε, V, ε_q, V_q, Wanniers{R}(),
                            ft, uₚ_buff_real, uₚ_buff_real2, uₚ_buff_complex, uₚ_buff_complex2)
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

"Helper function for q-diagonalisation that updates the diagonal blocks of `xh.H`."
function update_diag!(xh::DenseHamiltonian, U, K, QS, 𝑈_diag_allequal, 𝐴ᵢ_allequal, D, buff1, buff2)
    (;nc, B, Γ, H) = xh
    for c in 1:nc
        H_block = @view H[(c-1)*B+1:c*B, (c-1)*B+1:c*B]
        for i in 1:D
            which_K = 𝐴ᵢ_allequal[i] ? 1 : c
            copy!(buff1, K[which_K, i])
            buff1 += LA.I*QS[i]
            mul!(buff2, buff1, buff1)
            if i == 1
                copyto!(H, CartesianIndices(((c-1)*B+1:c*B, (c-1)*B+1:c*B)), buff2, CartesianIndices(buff2))
                # @debug "Copied (K[$which_K, $i] + QS[$i])^2 into H[$c, $c]"
            else
                H_block .+= buff2
                # @debug "Added (K[$which_K, $i] + QS[$i])^2 to H[$c, $c]"
            end
        end

        if 𝑈_diag_allequal
            H_block .+= U[1]
            # @debug "Added U[1] to H[$c, $c]"
        elseif !isnothing(U[c])
            H_block .+= U[c]
            # @debug "Added U[$c] to H[$c, $c]"
        end
        if Γ[c] != 0
            H_block -= LA.I * im*Γ[c]/2
            # @debug "Added -im*Γ[$c]/2 to H[$c, $c]"
        end
    end
    return
end
