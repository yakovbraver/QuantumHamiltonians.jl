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
    δ::R # coefficient of the momentum term: -iδ∇
    nc::Int # number of components
    isperiodic::Bool
    ishermitian::Bool # `H` is nonhermitian if decays Γ are present
    𝑈::Matrix{<:Union{Function,Nothing}} # nc-component matrix containing coordinate-space potentials and couplings
    𝑈_iseven::BitMatrix # nc-component matrix indicating if 𝑈ᵢⱼ is an even function 𝑈ᵢⱼ(𝑥, 𝑦) = 𝑈ᵢⱼ(-𝑥, -𝑦)
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
Construct a `DenseHamiltonian` object using the coordinate-space functions stored in `𝑈`, decay rates `Γ`, and gauge field (same for all components) 𝐴_x, 𝐴_y.
`M` is the maximum harmonic number. In the periodic case, the Hamiltonian will be `nc*(2M+1)²`-by-`nc*(2M+1)²` where `nc` is the number of components.
In nonperiodic case, the size will be `nc*M²`-by-`nc*M²`.
`𝑈_iseven[i, j]` matters only if `isperiodic=true` and shows whether `𝑈[i, j]` is an even function (i.e. whether 𝑢(𝑥, 𝑦) = 𝑢(-𝑥, -𝑦)). If it is, then Fourier transform is real, which is used for better accuracy.
If *all* functions are even (and real), then the resulting Fourier-space Hamiltonian is real (provided also there is no 𝐴 and Γ), giving a speed-up and better accuracy (compared to complex diagonalisation).
If `𝑈[i, j] === nothing` or it is complex, then the value of `𝑈_iseven[i, j]` does not matter.
𝐴[c, i] is ith projection 𝐴ᵢ of cth component
"""
function DenseHamiltonian(xlims::AbstractVector{Tuple{R,R}},
                                   𝑈::AbstractMatrix{<:Union{Function,Nothing}},
                                   𝐴::AbstractVecOrMat{<:Union{Function,Nothing}}=fill(nothing, size(𝑈, 1), length(xlims));
                                   isperiodic::Bool, M::Integer, δ::R=one(R),
                                   𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1))) where R <: AbstractFloat
    nc = size(𝑈, 1) # number of components
    D = length(xlims) # number of spatial dimensions
    L = [lims[2] - lims[1] for lims in xlims]

    # `H_isreal` will show if the resulting `H` will be real
    𝐴ᵢ_present = [any(𝐴ᵢ .!== nothing) for 𝐴ᵢ in eachcol(𝐴)] # `i` numbers projections; 𝐴ᵢ_present = true if 𝐴ᵢ is nonzero for at least one component
    U_isreal = all( 𝑢([xlims[i][1] for i in eachindex(xlims)]...) isa Real for 𝑢 in 𝑈 if !isnothing(𝑢) ) # check if all functions in 𝑈 are real
    H_isreal = U_isreal && all(𝐴ᵢ_present .== false) && iszero(Γ) # without checking we assume that all 𝐴's are real. Can be generalised for the exotic cases of complex 𝐴.
    if isperiodic # for periodic potential, also check if functions are even 
        H_isreal &= all(𝑈_iseven[𝑈 .!== nothing])
    end

    B = isperiodic ? (2M+1)^D : M^D # size of each Hamiltonian block

    T = H_isreal ? R : Complex{R} # type of elements of the Hamiltonian
    H = zeros(T, nc*B, nc*B)

    basis=(isperiodic ? :cis : :sin)
    ft = FourierTransformer(xlims, M; basis, target_real=U_isreal) # `target_real` will allocate a buffer for the imaginary part of the sin/cos-transform if some of 𝑈's are complex

    𝑈_diag_allequal = allequal(diagview(𝑈))
    𝐴ᵢ_allequal = [allequal(𝐴ᵢ) && !isnothing(𝐴ᵢ[1]) for 𝐴ᵢ in eachcol(𝐴)] # note that this also checks if they are nothing

    # treat diagonal blocks, adding the diagonal potentials 𝑈ᵢᵢ and 𝑝² (conditionally)
    for jH in 1:nc
        h = @view H[(jH-1)*B+1:jH*B, (jH-1)*B+1:jH*B] # a view of the `jH`th diagonal block
        h_set = false # shows if `h` has been set to something (i.e. etiher/both next two if's have been entered)
        if !isnothing(𝑈[jH, jH])
            transform!(ft, 𝑈[jH, jH])
            if basis == :cis && H_isreal # in this case the transform is actually real, but is stored in a complex array `ft.buff`,
                ft.buff .= real.(ft.buff) # make it real since otherwise `fft_to_matrix!` will not be able to write into `h` (which is real)
            end
            fft_to_matrix!(h, ft)
            h_set = true
            # @debug "Wrote 𝑈[$jH, $jH] into H[$jH, $jH]" # H[iH, jH] schematically means the block (`iH`, `jH`)
        end
        # Add 𝑝² if basis is sin/cos. But if there are no 𝐴's, add in the cis case too (if 𝐴's are present, then 𝑝ᵢ²'s will be added together with 𝐴ᵢ's)
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
    if any(𝐴 .!== nothing)
        A_buff = Matrix{T}(undef, B, B)
        A_buff2 = similar(A_buff)
        for i in 1:D # iterate over projections of 𝐴
            if !𝐴ᵢ_present[i] && basis != :cis # if the projection 𝐴ᵢ is zero for all components, then skip 𝐴ᵢ. However, if basis is cis, we cannot skip because also need to addf 𝑝ᵢ²
                continue
            end
            pᵢ = make_p_i(L, M, δ, basis, i)
            for c in 1:nc
                if isnothing(𝐴[c, i]) # then there is nothing to do, except adding 𝑝ᵢ² in the cis case
                    if basis == :cis
                        H[(c-1)*B+1:c*B, (c-1)*B+1:c*B] .+= pᵢ^2 # TODO write an in-place squaring function for this Diagonal matrix
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
                if 𝐴ᵢ_allequal[i] # then also copy into all other diagonal blocks and break
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
    for iH in 1:nc
        if Γ[iH] != 0
            H[diagind(H)[(iH-1)*B+1:iH*B]] .-= im*Γ[iH]/2
        end
    end
    # treat off-diagonal blocks (will not be run for a single component)
    for jH in 1:nc
        for iH in 1:jH-1 # only upper triangle is scanned. The lower triangle is filled only if Γ is present
            isnothing(𝑈[iH, jH]) && continue
            transform!(ft, 𝑈[iH, jH])
            if basis == :cis && H_isreal # in this case the transform is actually real, but is stored in a complex array,
                ft.buff .= real.(ft.buff) # so make it real since otherwise `fft_to_matrix!` will not be able to write into `h` (which is a real array)
            end
            wi = (iH-1)*B+1:iH*B
            wj = (jH-1)*B+1:jH*B
            h = @view H[wi, wj] # a view of the required block
            fft_to_matrix!(h, ft)
            # @debug "Wrote 𝑈[$iH, $jH] into H[$iH, $jH]" # H[jH, jH] schematically means the block (`iH`, `jH`)

            if !iszero(Γ) # fill conjugate block if Γ is present (then we cannot use Hermitian view)
                H[wj, wi] .= h' # TODO possible to use `copyto!` ?
                # @debug "Copied H[$iH, $jH]' into H[$jH, $iH]" # H[jH, jH] schematically means the block (`iH`, `jH`)
            end
        end
    end

    # determine the type of eigenvalues 
    ishermitian = iszero(Γ) # if all `Γ`s are zeros, then Hamiltonian is Hermitian and the eigenvalues real
    S = ishermitian ? R : Complex{R} # type of eigenvalues
    
    # create empty placeholders
    ε = S[] # eigenvalues, can be complex for nonhermitian `H`, hence additional type `S`
    V = T[;;] # eigenvectors matrix
    ε_q = Array{S}(undef, ntuple(Returns(0), D+1)) # ε_q[n, iqx, iqy] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    V_q = Array{T}(undef, ntuple(Returns(0), D+2)) # V_q[:, n, iqx, iqy] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)

    return DenseHamiltonian(xlims, L, M, δ, nc, isperiodic, ishermitian, 𝑈, BitMatrix(𝑈_iseven), 𝐴, Γ, H, ε, V, ε_q, V_q, Wanniers{R}())
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
Construct 2D coordinate-space wave function `ψ` of eigenstate `stateno` on a grid having `nx` points in `x` and `ny` points in `y` direction.
Return (`xs`, `ys`, `ψ`). If `qx` and `qy` are passed, then construct `ψ` at the corresponding quasimomenta.
"""
function make_eigenfunction(xh::XSpaceHamiltonian, stateno::Integer, nx::Integer, ny::Integer, iqx::Integer=0, iqy::Integer=0)
    (;L, xlims, M, V, V_q, nc) = xh
    Lx, Ly = L
    xs = range(0, Lx, nx) # these are the differences `x - xlims[1][1]`, with `x ∈ xlims[1]`
    ys = range(0, Ly, ny) # these are the differences `y - xlims[2][1]`, with `y ∈ xlims[2]`
    R = eltype(L) # real working type
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
    return xs .+ xlims[1][1], ys .+ xlims[2][1], ψ # return "normal" coordinates, in `x ∈ xlims` and `y ∈ ylims`
end

"""
Construct 1D eigenfunctions of state numbers `statenos` on a grid having `nx` points in `x` direction.
If a vector of quasimomentum indices `iqxs` is passed, then construct `ψ` for the state `statenos[1]` at the these quasimomenta.
Return (`xs`, `ψ`) where `ψ[x, components, statenos]` or `ψ[x, components, iqxs]`
"""
function make_eigenfunctions(xh::DenseHamiltonian; statenos::AbstractVector{<:Integer}, nx::Integer, iqxs::AbstractVector{<:Integer}=Int[])
    (;L, xlims, M, V, V_q, nc) = xh
    Lx = L[1]
    xs = range(0, Lx, nx) # these are the differences `x - xlims[1]`, with `x ∈ xlims`
    ns = isempty(iqxs) ? length(statenos) : length(iqxs)
    R = eltype(L) # real working type
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
    return xs .+ xlims[1][1], ψ # return "normal" coordinates, in `x ∈ xlims`
end

# """
# Calculate eigenenergies for all pairs of quasimomenta in `qxs` and `qys`.
# Calculate `nev` lowest levels using `ArnoldiMethod`.
# If `nev=0` or not passed, then full diagonalisation using `LinearAlgebra` is performed.
# Note that `dh.H` is modified in the process.
# """
# function diagonalize!(dh::DenseHamiltonian2D{R,T,S}, qxs::AbstractVector{<:Real}, qys::AbstractVector{<:Real}; nev::Integer, verbose::Bool=false) where {R<:Real, T<:Number, S<:Number}
#     (;M, xlims, ylims, Lx, Ly, δ, nc, H, 𝑈, 𝑈_iseven, 𝐴_x, 𝐴_y, Γ) = dh

#     if !dh.isperiodic
#         @warn "Hamiltonian must be periodic. Construct a new one and try again."
#         return
#     end

#     PI = R(π) # π of the working type to prevent widening

#     B = (2M + 1)^2 # block size
#     nsaves = nev == 0 ? B : nev # number of eigenvalues and eigenvectors to allocate
#     dh.ε_q = Array{S,3}(undef, nsaves, length(qxs), length(qys))
#     dh.V_q = Array{T,4}(undef, B*nc, nsaves, length(qxs), length(qys))
    
#     if all(isnothing.(𝐴_x)) && all(isnothing.(𝐴_y))
#         H_diag = diagview(dh.H)
#         # from the diagonal of each diagonal block of `H`, extract (𝑈ᵢᵢ)₀ (the 0th harmonic of 𝑈ᵢᵢ) plus decay -iΓ/2
#         U_diags = [H_diag[(c-1)B + B÷2+1] for c in 1:nc] # generally, `Hᵢᵢ = -Δᵢᵢ + Uᵢᵢ - iΓ/2`, but Δᵢᵢ = 0 for the central element of the diagonal (see construction of Δ in `DenseHamiltonian1D` constructor)
#     else
#         N = 4M + 1 # number of points for FFT. This will yield harmonics from -2M to 2M
#         dx, dy = Lx/N, Ly/N
#         xs = range(xlims[1], xlims[2]-dx, N)
#         ys = range(ylims[1], ylims[2]-dy, N)

#         fft_buff = Matrix{Complex{R}}(undef, N, N) # a buffer for all (in-place) FFTs
#         F = FFTW.plan_fft!(fft_buff) # the savings of rfft are negligible, and the output is much less convenient to handle in `fft_to_matrix`, so using fft. Also, this way we can do FFT in-place

#         𝑈_diag_allequal = allequal(diagview(𝑈)) & !isnothing(𝑈[1, 1])
#         𝐴_allequal = allequal(𝐴_x) & allequal(𝐴_y) & !isnothing(𝐴_x[1]) & !isnothing(𝐴_y[1])

#         nD = 𝐴_allequal ? 1 : nc # number of kinetic operators -iδ∂ₓ - 𝐴ₓ to allocate
#         nU = 𝑈_diag_allequal ? 1 : nc # number of terms (𝑈ᵢᵢ)₀ - iΓ/2 to allocate
#         D_x = Union{Matrix{T}, Nothing}[isnothing(𝐴_x[c]) ? nothing : Matrix{T}(undef, B, B) for c in 1:nD] # for storing kinetic operators -iδ∂ₓ - 𝐴ₓ
#         D_y = Union{Matrix{T}, Nothing}[isnothing(𝐴_y[c]) ? nothing : Matrix{T}(undef, B, B) for c in 1:nD] # for storing kinetic operators -iδ∂𝑦 - 𝐴𝑦
#         U = Union{Matrix{T}, Nothing}[isnothing(𝑈[c, c]) ? nothing : Matrix{T}(undef, B, B) for c in 1:nU] # for storing terms (𝑈ᵢᵢ)₀ (note that -iΓ/2 will not be stored here)

#         ∂_x = Diagonal([2PI * δ * jx/Lx for jx in -M:M for jy in -M:M]) # this is -iδ∂ₓ
#         ∂_y = Diagonal([2PI * δ * jy/Ly for jx in -M:M for jy in -M:M]) # this is -iδ∂y
#         # populate `D_x`, `D_y`, and `U`
#         for c in 1:nc
#             𝑢 = 𝑈[c, c]
#             𝑎_𝑥, 𝑎_𝑦 = 𝐴_x[1], 𝐴_y[1]
#             if !isnothing(𝑢) nU > 1 || (nU == 1 && c == 1) # if [we need more than one 𝑈ᵢᵢ (meaning all 𝑈ᵢᵢ's are different)] or [we need only one 𝑈ᵢᵢ (meaning all 𝑈ᵢᵢ's are equal) and we are on the first iteration]
#                 𝑢_isrealeven = (𝑢(xlims[1], ylims[1]) isa Real) & 𝑈_iseven[c, c]
#                 fft_buff .= 𝑢.(xs, ys')
#                 F * fft_buff # in-place FFT, weird syntax
#                 fft_buff ./= N^2
#                 U[c] .= fft_to_matrix_naive!(fft_buff, make_real=𝑢_isrealeven)
#             end
#             if !isnothing(𝑎_𝑥) && (nD > 1 || (nD == 1 && c == 1))
#                 fft_buff .= 𝑎_𝑥.(xs, ys')
#                 F * fft_buff
#                 fft_buff ./= N^2
#                 A_x = fft_to_matrix_naive!(fft_buff)
#                 D_x[c] .= ∂_x .- A_x
#                 # if there is no 𝐴𝑦, then set the 𝑦 kinetic `D_y[c]` term to -iδ∂𝑦. Otherwise `D_y[c]` will be treated in the next if clause
#                 isnothing(𝑎_𝑦) && (D_y[c] .= ∂_y)
#             end
#             if !isnothing(𝑎_𝑦)
#                 fft_buff .= 𝑎_𝑦.(xs, ys')
#                 F * fft_buff
#                 fft_buff ./= N^2
#                 A_y = fft_to_matrix_naive!(fft_buff)
#                 D_y[c] .= ∂_y .- A_y
#                 # if there is no 𝐴ₓ, then set the 𝑥 kinetic `D_x[c]` term to -iδ∂ₓ. Otherwise `D_x[c]` was treated in the preceding if clause
#                 isnothing(𝑎_𝑥) && (D_x[c] .= ∂_x)
#             end
#         end
#     end
    
#     buff_D = Matrix{T}(undef, B, B)
#     # update diagonal blocks and diagonalise
#     for (iqy, qy) in enumerate(qys), (iqx, qx) in enumerate(qxs)
#         # update diagonal blocks
#         if all(isnothing.(𝐴_x)) && all(isnothing.(𝐴_y))
#             buff = [(2PI*δ*jx/Lx + qx)^2 + (2PI*δ*jy/Ly + qy)^2 for jx in -M:M for jy in -M:M] # could make use of ∂_x above but that one is a matrix, while here it's a vector
#             for c in 1:nc
#                 H_diag[(c-1)B+1:c*B] .= buff .+ U_diags[c]
#             end
#         else
#             # first treat -iδ∂ₓ-𝐴ₓ, using the first diagonal block of `H` as a buffer
#             for c in 1:nc
#                 H_block = @view H[(c-1)*B+1:c*B, (c-1)*B+1:c*B]
#                 if !isnothing(D_x[c])
#                     if (nD > 1 || (nD == 1 && c == 1))  # if [more than one D_x exist (meaning all 𝐴ₓ's are different)] or [only one D_x exists (meaning all 𝐴ₓ's are equal) and we are on the first iteration]
#                         H_block .= (D_x[c] + LA.I*qx)^2 # then calulate
#                     else
#                         H_block .= @view H[1:B, 1:B] # otherwise just copy the first block
#                     end
#                 else # if there is no 𝐴ₓ in this block
#                     H_block .= (∂_x + LA.I*qx)^2
#                 end
#             end
#             # then treat -iδ∂𝑦-𝐴𝑦 and also 𝑈 with Γ, using the buffer `buff_D`
#             for c in 1:nc
#                 H_block = @view H[(c-1)*B+1:c*B, (c-1)*B+1:c*B]
#                 if !isnothing(D_y[c])
#                     if (nD > 1 || (nD == 1 && c == 1))  # if [more than one D_y exist (meaning all 𝐴ₓ's are different)] or [only one D_y exists (meaning all 𝐴ₓ's are equal) and we are on the first iteration]
#                         buff_D .= (D_y[c] + LA.I*qy)^2 # then calulate
#                         H_block .+= buff_D 
#                     else
#                         H_block .+= buff_D # otherwise just add the buffer
#                     end
#                 else # if there is no 𝐴𝑦 in this block
#                     H_block .+= (∂_y + LA.I*qy)^2
#                 end
#                 Γ[c] != 0 && (H_block .-= LA.I * im*Γ[c]/2)
#                 !isnothing(U[c]) && (H_block .+= U[nU == 1 ? 1 : c])
#             end
#         end

#         dh.ε_q[:, iqx, iqy], dh.V_q[:, :, iqx, iqy] = diagonalize(dh; nev, verbose)
#     end
# end