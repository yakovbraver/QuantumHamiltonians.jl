abstract type PSpaceHamiltonian{Storage, R<:AbstractFloat, T<:Union{R, Complex{R}}, S<:Union{R, Complex{R}}, D1, D2} end
# `R` - base real type, `T` - Hamiltonian, eigenvectors elements, `S` - eigenvalues
# The types are restricted *here*, therefore no need to specify restrictions in functions (except for constructors). E.g. an object with complex R cannot be constructed.

matrix_density(xh::PSpaceHamiltonian) = error("Matrix density calculation is available for sparse Hamiltonians only.")

"General dense constructor. If the problem is 1D, 𝐴 may be passed as a vector, whose elements are treated as corresponding to the different components."
function PSpaceHamiltonian{:dense}(xlims::AbstractVector{Tuple{R,R}},
                                   𝑈::AbstractMatrix{<:Union{Function,Nothing}},
                                   𝐴::AbstractVecOrMat{<:Union{Function,Nothing}}=fill(nothing, size(𝑈, 1), length(xlims));
                                   basis::Symbol, M::Integer, δ::R=one(R),
                                   𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1))) where R <: AbstractFloat
    return DenseHamiltonian(xlims, 𝑈, 𝐴; basis, M, δ, 𝑈_iseven, Γ)
end

"""
1-component (but many-D) dense constructor accepting a 𝑈 as a function, 𝐴 as a vector (with elements treated as corresponding to the different dimensions),
𝑈_iseven as a bool, and Γ as a real.
"""
function PSpaceHamiltonian{:dense}(xlims::AbstractVector{Tuple{R,R}},
                                   𝑈::Union{Function,Nothing},
                                   𝐴::AbstractVector{<:Union{Function,Nothing}}=fill(nothing, length(xlims));
                                   basis::Symbol, M::Integer, δ::R=one(R),
                                   𝑈_iseven::Bool=false, Γ::R=zero(R)) where R <: AbstractFloat
    return DenseHamiltonian(xlims, [𝑈;;], reshape(𝐴, (1, length(xlims))); basis, M, δ, 𝑈_iseven=[𝑈_iseven;;], Γ=[Γ])
end

"General sparse constructor. If the problem is 1D, 𝐴 may be passed as a vector, whose elements are treated as corresponding to the different components."
function PSpaceHamiltonian{:sparse}(xlims::AbstractVector{Tuple{R,R}},
                                    𝑈::AbstractMatrix{<:Union{Function,Nothing}},
                                    𝐴::AbstractVecOrMat{<:Union{Function,Nothing}}=fill(nothing, size(𝑈, 1), length(xlims));
                                    basis::Symbol, M::Integer, δ::R=one(R), fft_threshold::R=√eps(R),
                                    𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1))) where R <: AbstractFloat
    if R !== Float64
        println("Sparse diagonalisation is only supported for R = Float64, got R = $R. Constructed object will use Float64.")
        threshold = fft_threshold == √eps(R) ? √eps(Float64) : Float64(fft_threshold) # if `fft_threshold` is the default eps value, then switch to appropriate type
    else
        threshold = fft_threshold
    end
    lims = map(t -> Float64.(t), xlims)
    return SparseHamiltonian(lims, 𝑈, 𝐴; basis, M, δ=Float64(δ), 𝑈_iseven, Γ=Float64.(Γ), fft_threshold=threshold)
end

"""
1-component (but many-D) sparse constructor accepting a 𝑈 as a function, 𝐴 as a vector (with elements treated as corresponding to the different dimensions),
𝑈_iseven as a bool, and Γ as a real.
"""
function PSpaceHamiltonian{:sparse}(xlims::AbstractVector{Tuple{R,R}},
                                   𝑈::Union{Function,Nothing},
                                   𝐴::AbstractVector{<:Union{Function,Nothing}}=fill(nothing, length(xlims));
                                   basis::Symbol, M::Integer, δ::R=one(R), fft_threshold::R=√eps(R),
                                   𝑈_iseven::Bool=false, Γ::R=zero(R)) where R <: AbstractFloat
    if R !== Float64
        println("Sparse diagonalisation is only supported for R = Float64, got R = $R. Constructed object will use Float64.")
        threshold = fft_threshold == √eps(R) ? √eps(Float64) : Float64(fft_threshold) # if `fft_threshold` is the default eps value, then switch to appropriate type
    else
        threshold = fft_threshold
    end
    lims = map(t -> Float64.(t), xlims)
    return SparseHamiltonian(lims, [𝑈;;], reshape(𝐴, (1, length(xlims))); basis, M, δ=Float64(δ), 𝑈_iseven=[𝑈_iseven;;], Γ=Float64[Γ], fft_threshold=threshold)
end

"""
Construct 1D coordinate-space eigenfunctions of state numbers `statenos` on a grid having `nx` points in `x` direction.
If a vector of quasimomentum indices `iqxs` is passed, then construct `ψ` for the state `statenos[1]` at the these quasimomenta.
Return (`xs`, `ψ`) where `ψ[x, components, statenos]` or `ψ[x, components, iqxs]`
"""
function make_eigenfunctions(xh::PSpaceHamiltonian{Storage,R}; statenos::AbstractVector{<:Integer}, nx::Integer, iqxs::AbstractVector{<:Integer}=Int[]) where {Storage,R}
    (;L, xlims, M, basis, V, V_q, nc) = xh
    Lx = L[1]
    xs = range(0, Lx, nx) # these are the differences `x - xlims[1]`, with `x ∈ xlims`
    ns = isempty(iqxs) ? length(statenos) : length(iqxs)
    ψ_type = basis != :cis && eltype(xh.H) <: Real ? R : complex(R)  # `ψ` are real if elements of H are real and if the basis is real (sin/cos)
    ψ = Array{ψ_type}(undef, nx, nc, ns)
    if isempty(iqxs) # no quasimomentum index
        for (is, stateno) in enumerate(statenos)
            for c in 1:nc
                if basis == :cis
                    B = 2M + 1
                    @floop for (ix, x) in enumerate(xs)
                        ψ[ix, c, is] = sum(V[(c-1)*B+j, stateno]cis(2π*jx*x/Lx) for (j, jx) in enumerate(-M:M)) / √Lx
                    end
                elseif basis == :sin
                    @floop for (ix, x) in enumerate(xs)
                        ψ[ix, c, is] = sum(V[(c-1)*M+jx, stateno]sin(π*jx*x/Lx) for jx in 1:M) * √(2/Lx)
                    end
                else # basis == :cos
                    b = M + 1 # not B to prevent Core.Box
                    @floop for (ix, x) in enumerate(xs)
                        ψ[ix, c, is] = sum(V[(c-1)*b+jx+1, stateno]cos(π*jx*x/Lx) for jx in 1:M) * √(2/Lx)
                    end
                    ψ[:, c, is] .+= V[(c-1)*b+1, stateno] / √Lx # treat zeroth harmonic separately
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

"""
Construct a D-dimensional x-space wave function using its p-space representation `ψₚ` (1D vector).
Pass integer `pad` to pad `ψₚ` with zeros, interpolating the x-space function as if reconstructed using `2^pad*xh.M` harmonics (instead of `xh.M`).
Return a tuple (`xs`, `ψ`) where `ψ[component][x, y, …]` while `xs[:, 1]` contains sampled 𝑥, `xs[:, 2]` contains sampled 𝑦, etc..
"""
function make_wavefunction(xh::PSpaceHamiltonian{Storage, R}, ψₚ::AbstractVector{T}; pad::Integer=0) where {Storage, R, T<:Number}
    (;xlims, basis, nc) = xh
    ψₚ_isreal = T <: Real

    B = length(ψₚ) ÷ nc # original block-size
    # number of x points in each dimension
    nx = basis == :cis ? 2xh.M+1 :
         basis == :sin ?    xh.M : xh.M+1

    M = 2^pad * xh.M
    ft = FourierTransformerP(xlims, M; basis, target_real=ψₚ_isreal, target_rank=1)
    nx_padded = size(ft.xs, 1) # number of x points in each dimension for the padded array
    
    ψₓ_type = basis != :cis && ψₚ_isreal ? R : complex(R) # `ψ` are real if elements of `ψₚ` are real and if the basis is real (sin/cos). If basis is real but `ψₚ` are complex, this will yield complex function as expected.
    D = length(xlims) # number of spatial dimensions
    ψₓ = [Array{ψₓ_type}(undef, ntuple(Returns(nx_padded), D)) for _ in 1:nc]
    
    if pad > 0
        ψₚᶜ_indices = ntuple(Returns(nx), D) # indices that `ψₚᶜ` will have when reshaped into a D-dimensional tensor
        ψₚᶜ_padded = zeros(T, size(ψₓ[1])) # a buffer for a padded version of `c`th component, same shape as each components `ψₓ`
        # determine the indices of `ψₚᶜ_padded` where `ψₚᶜ` will be copied to
        if basis == :cis
            offset = (nx_padded - nx) ÷ 2
            ψₚᶜ_padded_indices = CartesianIndices(ntuple(i -> offset+1:nx+offset, D)) # will copy to the middle of the array
        else # sin/cos
            ψₚᶜ_padded_indices = CartesianIndices(ψₚᶜ_indices) # will copy to the low-frequency corner
        end
    end
    
    for c in 1:nc
        ψₚᶜ = @view ψₚ[(c-1)B+1:c*B] # take part of `ψₚ` corresponding to the `c`th component
        if pad > 0
            # reshape `ψₚᶜ` into a D-dimensional tensor and copy into the padded array
            copyto!(ψₚᶜ_padded, ψₚᶜ_padded_indices, reshape(ψₚᶜ, ψₚᶜ_indices), CartesianIndices(ψₚᶜ_indices))
            transform!(ft, ψₚᶜ_padded; direction=:backward) # here the input vector is D-dimensional
        else
            transform!(ft, ψₚᶜ; direction=:backward) # here the input vector is 1D; `transform!` will reshape it.
        end
        fft_to_state!(ψₓ[c], ft; direction=:backward) # this essentially copies from `ft.buff` into `ψₓ[c]`, but there are a few extra steps to take care of
    end
    return ft.xs, ψₓ
end

"""
Construct x-space wave function of eigenstate `stateno`.
If a vector of quasimomenta indices `iqs = [iqx, iqy, …]` is provided, then construct the wave function at those indices for band number `stateno`.
Pass integer `pad` to interpolate the x-space function as if reconstructed using `2^pad*xh.M` harmonics (instead of `xh.M`).
Return a tuple (`xs`, `ψ`) where `ψ[component][x, y, …]` while `xs[:, 1]` contains sampled 𝑥, `xs[:, 2]` contains sampled 𝑦, etc..
"""
function make_eigenfunction(xh::PSpaceHamiltonian{Storage, R}, stateno::Integer, iqs::Union{Nothing, AbstractVector{<:Integer}}=nothing; pad::Integer=0) where {Storage, R}
    if !isnothing(iqs) # if quasimomentum index has been passed
        make_wavefunction(xh, xh.V_q[:, stateno, iqs...]; pad)
    else
        make_wavefunction(xh, xh.V[:, stateno]; pad)
    end
end

"""
Calculate eigenenergies for all quasimomenta in `qs = [qxs, qys, ...]` where `qxs` are 𝑞's along 𝑥, etc.
Calculate `nev` lowest levels using `ArnoldiMethod`.
Pass `nev=0` for full diagonalisation using `LinearAlgebra`.

Note that `xh.H` is modified in the process. In the case when 𝐴 is absent, only the diagonal of `xh.H` is modified, and it is restored in the end (this is cheap, ~3 ms for M=50).
When 𝐴 is present, the entire diagonal blocks of `xh.H` are modified, and they are not restored in the end. However, they are constructed from scratch when the function is called rather than using the contents of `xh.H`.
Thus, in both cases this function can be called repeatedly (e.g. for different `qs`) without reconstructing `xh`.
"""
function diagonalize!(xh::PSpaceHamiltonian{Storage, R, T, S}, qs::AbstractVector{<:AbstractVector{<:Real}}; nev::Integer, verbose::Bool=false) where {Storage, R, T, S}
    xh.basis != :cis && error("Hamiltonian must be periodic. Construct a new one using the cis basis and try again.")
    (;M, xlims, L, δ, nc, H, 𝑈, 𝑈_iseven, 𝐴, Γ) = xh
    D = length(xlims)
    if Storage == :dense
        makesparse = false
        threshold = zero(R) # the value does not matter since it is not used when `makesparse=false`
    else
        makesparse = true
        threshold = xh.fft_threshold
    end

    B = (2M + 1)^D # block size
    nsaves = nev == 0 ? B : nev # number of eigenvalues and eigenvectors to allocate
    xh.ε_q = Array{S}(undef, nsaves, ntuple(i -> length(qs[i]), D)...) # ε_q[n, iqx, iqy, ...] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    xh.V_q = Array{T}(undef, B*nc, nsaves, ntuple(i -> length(qs[i]), D)...) # V_q[:, n, iqx, iqy, ...] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)
    
    if all(isnothing.(𝐴)) # very simple case (with no 𝐴) that we can treat separately
        H_diag = diagview(xh.H)
        H_diag_copy = copy(H_diag) # a copy for restoring after diagonalisation
        # from the diagonal of each diagonal block of `H`, extract (𝑈ᵢᵢ)₀ (the 0th harmonic of 𝑈ᵢᵢ) plus decay -iΓ/2
        U_diags = T[H_diag[(c-1)B + B÷2+1] for c in 1:nc] # generally, `Hᵢᵢ = -Δᵢᵢ + Uᵢᵢ - iΓ/2`, but Δᵢᵢ = 0 for the central element of the diagonal
    else # the general case with 𝐴
        if Storage == :dense
            # two buffers that are need in the q-loop in th dense case for matrix multiplication
            buff1 = Matrix{T}(undef, B, B)
            buff2 = Matrix{T}(undef, B, B)
        else
            buff1 = nothing
            buff2 = nothing
        end

        ft = FourierTransformerP(xlims, M; basis=:cis)

        K = Union{typeof(H), Diagonal{T, Vector{T}}, Nothing}[nothing for _ in CartesianIndices(𝐴)] # Matrix of dimensions like 𝐴 for storing corresponding kinetic operators -iδ∂ᵢ - 𝐴ᵢ
        U = Union{typeof(H), Nothing}[nothing for _ in axes(𝑈, 1)] # for storing terms 𝑈ᵢᵢ

        𝑈_diag_allequal = allequal(diagview(𝑈))
        𝐴ᵢ_allequal = Bool[allequal(𝐴ᵢ) for 𝐴ᵢ in eachcol(𝐴)] # 𝐴ᵢ_allequal[i] shows if projection 𝐴ᵢ is the same for all components; they may all be nothing

        # fill the buffers `U`
        for c in 1:nc
            if !isnothing(𝑈[c, c])
                transform!(ft, 𝑈[c, c])
                U[c] = fft_to_operator(ft; makesparse, threshold)
                # @debug "Filled U[$c]"
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
                    # @debug "Wrote p_$i to K[$c, $i]"
                else
                    transform!(ft, 𝐴[c, i])
                    K[c, i] = fft_to_operator(ft; makesparse, threshold)
                    K[c, i] .= pᵢ .- K[c, i]
                    # @debug "Wrote p_$i - 𝐴[$c, $i] to K[$c, $i]"
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

        # @debug "🚜 QS = $QS 🚜"

        # update diagonal blocks
        if all(isnothing.(𝐴)) # very simple case (with no 𝐴) that we can treat separately
            p² = make_p²(L, M, δ, :cis, QS) |> parent # `parent` returns the diagonal as a vector TODO make in-place
            for c in 1:nc
                H_diag[(c-1)B+1:c*B] .= p² .+ U_diags[c]
            end
        else # the general case with 𝐴
            update_diag!(xh, U, K, QS, 𝑈_diag_allequal, 𝐴ᵢ_allequal, D, buff1, buff2)
        end

        xh.ε_q[:, IQ...], xh.V_q[:, :, IQ...] = diagonalize(xh; nev, verbose)
    end
    all(isnothing.(𝐴)) && copy!(H_diag, H_diag_copy) # restore the original diagonal
    return
end

################ Dense ################

"""
Calculate `nev` lowest eigenvectors and eigenvalues using `ArnoldiMethod`.
Pass `nev=0` for full diagonalisation using `LinearAlgebra`.
The result is written into `xh.ε` and `xh.V`.
"""
function diagonalize!(xh::PSpaceHamiltonian; nev::Integer, verbose::Bool=false)
    xh.ε, xh.V = diagonalize(xh; nev, verbose)
    return
end

"""
Calculate `nev` lowest eigenvectors and eigenvalues using `ArnoldiMethod`.
Pass `nev=0` for full diagonalisation using `LinearAlgebra`.
Return a tuple (eigenvalues, eigenvectors).
"""
function diagonalize(xh::PSpaceHamiltonian{:dense}; nev::Integer, verbose::Bool=false)
    if nev == 0
        if xh.ishermitian
            return eigen(Hermitian(xh.H)) # if `xh.H` is real, the appropriate routine will be selected automatically, no need to use `Symmetric` instead of `Hermitian`
        else
            return eigen(xh.H)
        end
    else
        if xh.ishermitian
            ps, info = partialschur(dense_linear_map(Hermitian(xh.H)); nev, which=:LM)
            verbose && @show info
            ps.eigenvalues .= inv.(real.(ps.eigenvalues)) # invert back
            return ps.eigenvalues, ps.Q # here `ps.eigenvalues` is Complex, but once returned it will be copied into real `xh.ε` with no error because imaginary part is zeroed out
        else
            ps, info = partialschur(dense_linear_map(xh.H); nev, which=:LM)
            verbose && @show info
            ε, V = partialeigen(ps)
            ε .= inv.(ε)
            reverse!(ε) # we want final eigenvalues in ascending order (by abs)
            reverse!(V; dims=2) # reverse the eigenvectors accordingly. Takes ~10⁵ less time than diagonalisation itself, so is negligible
            return ε, V
        end
    end
end

"Helper function for shift-and-invert: construct a linear map that applies the inverse of `A`."
function dense_linear_map(A)
    F = factorize(A) # Bunch-Kaufman for Hermitian `A`, LU otherwise
    return LM.LinearMap{eltype(A)}((y, x) -> ldiv!(y, F, x), size(A, 1), ismutating=true)
end

################ Sparse ################

"""
Calculate `nev` lowest eigenvectors and eigenvalues.
Return a tuple (eigenvalues, eigenvectors).
"""
function diagonalize(xh::PSpaceHamiltonian{:sparse}; nev::Integer, verbose::Bool=false)
    prob = LS.LinearProblem(xh.H, similar(xh.H, size(xh.H, 1)))
    linsolve = LS.init(prob, LS.UMFPACKFactorization())
    linmap = LinSolveLinMap{eltype(xh.H), typeof(linsolve)}(linsolve, size(xh.H))
    ps, info = partialschur(linmap; nev, which=:LM);
    verbose && @show info
    ε, V = partialeigen(ps)
    reverse!(ε) # we want final eigenvalues in ascending order (by abs)
    reverse!(V; dims=2) # reverse the eigenvectors accordingly
    if xh.ishermitian # if xh.H is Hermitian but complex, the solver returns complex eigenvalues
        ε .= real.(inv.(ε)) # so we make them real manually
    else
        ε .= inv.(ε)
    end
    return ε, V
end

"A linear map holding a `LinearSolve` object, used for applying the inverse map."
struct LinSolveLinMap{T,L} <: LM.LinearMap{T}
    linsolve::L
    size::Dims{2}
end

Base.size(lm::LinSolveLinMap) = lm.size

function LM._unsafe_mul!(y, lm::LinSolveLinMap, x::AbstractVector)
    copy!(lm.linsolve.b, x)
    copy!(y, LS.solve!(lm.linsolve).u) # `solve!` allocates up to 50 KiB :(
end

################ GPE-related methods ################

"""
For a state `v`, return `E, μ, η`, where `E` is mean energy per particle, `μ` is a vector of chemical potentials of each component,
and `η` is a vector of relative particle numbers of each component.
`v_is_pspace=true` means that `v` is given in p-space, and x-space otherwise.
By default, `makereal=true` so that the returned `E` and `μ` are made real (by dropping imaginary part). Set `makereal=false` if you consider a decaying state, whereby imaginary part is important.
"""
function get_Eμη(xh::PSpaceHamiltonian{Storage, R}, v::AbstractVector{<:Number}, g::AbstractMatrix{<:Number}=zeros(typeof(xh.δ), xh.nc, xh.nc);
                 v_is_pspace=true, makereal=true) where {Storage, R}
    (;xlims, M, nc, basis) = xh
    B = size(xh.H, 1) ÷ nc  
     
    if v_is_pspace # if `v` is in p-space, then make `vₚ` point to `v`
        vₚ = v
    else # if `v` is in x-space, then perform FT to transition to p-space
        v_isreal = eltype(v) <: Real
        ft = FourierTransformerP(xlims, M; basis, target_real=v_isreal, target_rank=1)
        v_type = !v_isreal ? Complex{R} : eltype(ft.buff) # if v in x-space is complex, then result will be complex; otherwise the same as determined in `ft`
        vₚ = Vector{v_type}(undef, length(v))
        @views for c in 1:nc
            window = (c-1)B+1:c*B
            transform!(ft, v[window]; direction=:forward)
            fft_to_state!(vₚ[window], ft; direction=:forward)
        end
    end

    η = [@views sum(abs2, vₚ[(c-1)B+1:c*B]) for c in 1:nc]
    η_total = sum(η)
    e = [@views dot(vₚ[(c-1)B+1:c*B], xh.H[(c-1)B+1:c*B, :], vₚ[1:nc*B]) for c in 1:nc] # using `vₚ[1:nc*B]` instead of just `vₚ` because it might contain chemical potentials as the last `nc` elements
    E = sum(e) / η_total
    μ = e ./ η
    
    if !iszero(g)
        # we need `ft` object to get the coordinates
        v_isreal = eltype(v) <: Real
        ft = FourierTransformerP(xlims, M; basis, target_real=v_isreal, target_rank=1)
        dV = prod(ft.xs[2, i] - ft.xs[1, i] for i in axes(ft.xs, 2)) # volume element
        if v_is_pspace # if `v` is in p-space, then perform FT to x-space
            # create an array of arrays holding squared x-space densities |𝜓(𝑥)|² for each component
            ψ² = map(1:nc) do c
                @views transform!(ft, v[(c-1)B+1:c*B]; direction=:backward)
                ψ = fft_to_state(ft; direction=:backward)
                ψ .= abs2.(ψ)
                return ψ
            end
        else # if `v` is in x-space, then calculate abs2 directly, but we need a vector of vectors instead of contiguous
            ψ² = map(1:nc) do c
                @views abs2.(v[(c-1)B+1:c*B])
            end
        end
        # for each `i`th component: calculate the sum ∑ⱼ 𝑔ᵢⱼ|𝜓ⱼ|², then multiply by |𝜓ᵢ|², then integrate
        ψ²_sum = similar(ψ²[1])
        for i in 1:nc
            ψ²_sum .= 0
            for j in 1:nc
                g[i, j] == 0 && continue
                @. ψ²_sum += g[i, j] * ψ²[j]
            end
            ψ²_sum .*= ψ²[i]
            U = sum(ψ²_sum) * dV # for sin, endpoints are not included but are zero, so this is equivalent to the trapezoid rule. For cis, rectangle rule is more appropriate because there is no boundary
            basis == :cos && (U -= (ψ²_sum[1] + ψ²_sum[end])/2 * dV)
            μ[i] += U / η[i]
            E += U / 2η_total
        end
    end
    if makereal
        return real(E), real(μ), η
    else
        return E, μ, η
    end
end