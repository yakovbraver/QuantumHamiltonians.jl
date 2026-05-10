"""
A lazy linear map describing the action of the BdG operator on an x-space vector
    𝑐 = (𝑎₁(x), …, 𝑎ₙ(x), 𝑏₁(x), …, 𝑏ₙ(x))
"""
struct BdGMap{T, H, P, R, FT} <: LM.LinearMap{T}
    Hₚ::H # Hamiltonian matrix in p-space; can be dense or sparse
    H⁺ₚ::H # p-space matrix corresponding to the conjugated x-space Hamiltonian; can be dense or sparse
    G::Matrix{Vector{P}} # An analogue of the BdG matrix
    nc::Int # number of components 
    B::Int # block size
    ψₚ_buff1_real::Vector{R} # real buffer for storing ψₚ, needed in the sin/cos cases when acting on real vectors
    ψₚ_buff2_real::Vector{R} # real buffer for storing Hₚ*ψₚ, needed in the sin/cos cases when acting on real vectors
    ψₚ_buff1_complex::Vector{Complex{R}} # complex buffer for storing ψₚ, needed when acting on complex vectors
    ψₚ_buff2_complex::Vector{Complex{R}} # complex buffer for storing Hₚ*ψₚ, needed when acting on complex vectors
    ft::FT # Fourier transformer supporting both forward and backward transforms
    size::Dims{2} # size that the map would have were it a concrete matrix. For us it's double the size of `Hₚ`
end

Base.size(lm::BdGMap) = lm.size

"""
Construct a `BdGMap` object for analysing stability of x-space discretised state `f`, containing all components in a contiguous vector.
Pass `floquet=true` if map is to be used for a periodic problem. This affects the resulting type of the map.
Currrently, only Hamiltonians that are real in x-space are supported.
"""
function BdGMap(xh::PSpaceHamiltonian{Storage, R}, f::AbstractVector{S}, g::AbstractMatrix{R}, μ::AbstractVector{R}; floquet=false) where {Storage, R, S}
    # eltype of the `G` matrix: real if Hamiltonian is real in x-space and `f` is real; complex otherwise
    P = all(isnothing.(xh.𝐴)) && S <: Real ? R : Complex{R} # TODO determination of whether Hamiltonian is real in incorrect because off-diagonal blocks can be complex
    # eltype of the `G` matrix: real if Hamiltonian is real in x-space and `f` is real; complex otherwise
    T = floquet ? Complex{R} : P # in the Floquet case, map is always complex in the coordinate space (because of (-i∇ + q)²); otherwise, the type is `P`

    # Fill the `G` matrix. The Hamiltonian part is treated separately.
    if xh.nc == 1
        #  (𝜇 + 2𝑔|𝑓|²)𝑎 + 𝑔𝑓²𝑏  = i𝜆𝑎
        # -(𝜇 + 2𝑔|𝑓|²)𝑏 - 𝑔𝑓⁺²𝑎 = i𝜆𝑏
        G = [similar(f, P) for _ in 1:2, _ in 1:2]
        @. G[1, 1] = 2g[1] * abs2(f) - μ[1]
        @. G[1, 2] = g[1] * f^2
        @. G[2, 1] = -conj(G[1, 2])
        @. G[2, 2] = -G[1, 1]
    elseif xh.nc == 2
        #  (2𝑔₁₁|𝑓₁|² + 𝑔₁₂|𝑓₂|² - 𝜇₁)𝑎₁ + 𝑔₁₂𝑓₁𝑓₂⁺𝑎₂ + 𝑔₁₁𝑓₁²𝑏₁ + 𝑔₁₂𝑓₁𝑓₂𝑏₂    = i𝜆𝑎₁
        #  𝑔₂₁𝑓₁⁺𝑓₂𝑎₁ + (2𝑔₂₂|𝑓₂|² + 𝑔₂₁|𝑓₁|² - 𝜇₂)𝑎₂ + 𝑔₂₁𝑓₁𝑓₂𝑏₁ + 𝑔₂₂𝑓₂²𝑏₂    = i𝜆𝑎₂
        # -𝑔₁₁𝑓₁⁺²𝑎₁ - 𝑔₁₂𝑓₁⁺𝑓₂⁺𝑎₂ - (2𝑔₁₁|𝑓₁|² + 𝑔₁₂|𝑓₂|² - 𝜇₁)𝑏₁ - 𝑔₁₂𝑓₁⁺𝑓₂𝑏₂ = i𝜆𝑏₁
        # -𝑔₂₁𝑓₁⁺𝑓₂⁺𝑎₁ - 𝑔₂₂𝑓₂⁺²𝑎₂ - 𝑔₂₁𝑓₁𝑓₂⁺𝑏₁ - (2𝑔₂₂|𝑓₂|² + 𝑔₂₁|𝑓₁|² - 𝜇₂)𝑏₂ = i𝜆𝑏₂
        @views f₁, f₂ = f[1:end÷2], f[end÷2+1:end]
        G = [similar(f₁, P) for _ in 1:4, _ in 1:4]
        @. G[1, 1] = 2g[1,1]abs2(f₁) + g[1,2]abs2(f₂) - μ[1]
        @. G[1, 2] = g[1,2] * f₁ * conj(f₂)
        @. G[1, 3] = g[1,1] * f₁^2
        @. G[1, 4] = g[1,2] * f₁ * f₂
        @. G[2, 1] = conj(G[1, 2]) # assumes g[1,2] = g[2,1]
        @. G[2, 2] = 2g[2,2]abs2(f₂) + g[2,1]abs2(f₁) - μ[2]
        @. G[2, 3] = G[1, 4]       # assumes g[1,2] = g[2,1]
        @. G[2, 4] = g[2,2] * f₂^2
        @. G[3, 1] = -conj(G[1, 3])
        @. G[3, 2] = -conj(G[1, 4])
        @. G[3, 3] = -G[1, 1]
        @. G[3, 4] = -conj(G[1, 2])
        @. G[4, 1] = -conj(G[1, 4]) # assumes g[1,2] = g[2,1]
        @. G[4, 2] = -conj(G[2, 4]) # assumes g[1,2] = g[2,1]
        @. G[4, 3] = -G[1, 2]
        @. G[4, 4] = -G[2, 2]
    else
        error("BdGMap not implemented for $(xh.nc) components")
    end

    # prepare the plan that can transform either real or complex vectors (hence `target_real=false`), because the map might need to act on complex ones during diagonalisation
    ft = FourierTransformerP(xh.xlims, xh.M; xh.basis, target_real=false, target_rank=1)
    
    ψₚ_buff1_real = similar(f, R)
    ψₚ_buff2_real = similar(f, R)
    ψₚ_buff1_complex = similar(f, Complex{R})
    ψₚ_buff2_complex = similar(f, Complex{R})

    Hₚ = copy(xh.H) # could be a reference, but we modify it when scanning quasimomenta
    H⁺ₚ = copy(xh.H) # copy the same Hamiltonian for now: TODO implement conjugation
    B = size(xh.H, 1) ÷ xh.nc # block size
    return BdGMap{T, typeof(Hₚ), P, R, typeof(ft)}(
        Hₚ, H⁺ₚ, G, xh.nc, B, ψₚ_buff1_real, ψₚ_buff2_real, ψₚ_buff1_complex, ψₚ_buff2_complex, ft, size(xh.H) .* 2
    )
end

"""
Apply the Hamiltonian and the BdG matrix to `ψ_in`, storing the result in `ψ_out`.
The format of `ψ_in` is (𝑎₁, …, 𝑎ₙ, 𝑏₁, …, 𝑏ₙ) where 𝑛 is the number of components.
"""
function LM._unsafe_mul!(ψ_out, bdg_map::BdGMap{T}, ψ_in::AbstractVector) where T
    (;Hₚ, H⁺ₚ, G, nc, B, ft) = bdg_map
    block(i) = (i-1)B+1:i*B
    
    ψ_in_isreal = eltype(ψ_in) <: Real
    if ft.basis == :cis || !ψ_in_isreal || T <: Complex # then dealing with complex functions, so will be using the complex buffers `ψₚ_buff1`, `ψₚ_buff2`
        ψₚ_buff1, ψₚ_buff2 = bdg_map.ψₚ_buff1_complex, bdg_map.ψₚ_buff2_complex
    else
        ψₚ_buff1, ψₚ_buff2 = bdg_map.ψₚ_buff1_real, bdg_map.ψₚ_buff2_real
    end

    # transform `ψ_in` to p-space, multiply by `H` and transform back, writing into `ψ_out`
    @views for i in 1:2 # i = 1 iterates 𝑎₁, …, 𝑎ₙ; i = 2 iterates 𝑏₁, …, 𝑏ₙ
        # transform 𝑎₁, …, 𝑎ₙ to p-space, writing into appropriate parts of `ψₚ_buff1`
        for j in 1:nc
            transform!(ft, ψ_in[block((i-1)nc+j)]; direction=:forward)
            fft_to_state!(ψₚ_buff1[block(j)], ft; direction=:forward)
        end
        # apply Hamiltonian
        if i == 1
            mul!(ψₚ_buff2, Hₚ, ψₚ_buff1)
        else
            mul!(ψₚ_buff2, H⁺ₚ, ψₚ_buff1)
            @. ψₚ_buff2 = -ψₚ_buff2 # because should have multiplied by -H⁺ₚ
        end
        # transform 𝑎₁, …, 𝑎ₙ back to x-space, writing into appropriate parts of `ψ_out`
        for j in 1:nc
            transform!(ft, ψₚ_buff2[block(j)]; direction=:backward)
            fft_to_state!(ψ_out[block((i-1)nc+j)], ft; direction=:backward, makereal=(ψ_in_isreal && T <: Real)) # if initial ψ is real and map also, then make the result real
        end
    end
    # add G * ψ_in to `ψ_out`
    @views for j in axes(G, 2), i in axes(G, 1)
        @. ψ_out[block(i)] += G[i, j] * ψ_in[block(j)]
    end
    return
end

"""
Update the Hamiltonians `bdg_map.H` and `bdg_map.H` using quasimomenta `q` and the reference Hamiltonian `xh.H`.
Currently, assumes that Hamiltonian is just the Laplacian. The body is to be replaced with something as in q-diagonalisation function.
"""
function update_H!(bdg_map::BdGMap{T}, xh::PSpaceHamiltonian, q::AbstractVector{<:Real}) where {T}
    Hₚ_diag  = diagview(bdg_map.Hₚ)
    H⁺ₚ_diag = diagview(bdg_map.H⁺ₚ)
    p²  = make_p²_matrix(xh.L, xh.M, xh.δ, :cis, q) |> parent # `parent` returns the diagonal as a vector
    B = size(xh.H, 2)÷xh.nc # block size
    for c in 1:xh.nc
        Hₚ_diag[(c-1)B+1:c*B]  .= p²
        H⁺ₚ_diag[(c-1)B+1:c*B] .= p²
    end
    return
end

"""
Compute BdG stability spectrum and eigenfunctions for an x-space state `ψ`.
By default, calculate all eigenvalues using a dense map.
Alternatively, pass `nev` number of smallest-magnitude eigenvalues to calculate.
Additionally, `storage=:sparse` or `storage=:lazy` will use sparse or matrix-free representations respectively.
`ψ` can be a vector or a N×1 matrix (where N is the number of x points).
The interaction strength `g` can be passed as a scalar number in the 1-component case.
The chemical potential `μ` can be passed as a vector, or a number if it is the same for all components (including the 1-component case).
"""
function bdg_spectrum(xh::PSpaceHamiltonian{Storage, R}, ψ::AbstractVecOrMat{<:Union{R, Complex{R}}}, g::Union{R, AbstractMatrix{R}}, μ::Union{R, AbstractVector{R}};
                      storage=:dense, nev::Integer=0, verbose::Bool=false) where {Storage, R}

    μ_input = μ isa R ? fill(μ, xh.nc) : μ # if only one μ is passed, then construct a vector of same values; also, 1-component case needs a vector
    g_input = g isa R ? [g;;] : g # constructor needs a vector, so make one in the 1-component case
    bdg_map = BdGMap(xh, ψ, g_input, μ_input)
    
    diagonalize(bdg_map, ψ; storage, nev, verbose)
end

"Diagonalise `bdg_map`."
function diagonalize(bdg_map::BdGMap{T}, ψ::AbstractVecOrMat{<:Union{R, Complex{R}}}; storage=:dense, nev::Integer=0, verbose::Bool=false) where {T, R}
    if storage == :lazy
        if nev > 0
            # Here we do shift-invert. "Inversion" is actually solution of a linear system. But `LS.LinearProblem` does not work with LinearMaps, so we wrap `bdg_map` in a SciMLOperator
            bdg_op = SciMLOperators.FunctionOperator(BdGMap!, similar(ψ, 2length(ψ)); p=bdg_map, isconstant=true)
            prob = LS.LinearProblem(bdg_op, similar(ψ, 2length(ψ)))
            linsolve = LS.init(prob, LS.KrylovJL_BICGSTAB())
            linmap = LinSolveLinMap{T, typeof(linsolve)}(linsolve, size(bdg_map))
            ps, info = partialschur(linmap; nev, which=:LM)
            verbose && @show info
            vals, vecs = partialeigen(ps)
            return inv.(vals), vecs
        else
            ps, info = partialschur(bdg_map; nev=size(bdg_map, 1), which=:LM);
            verbose && @show info
            return partialeigen(ps)
        end
    elseif storage == :sparse
        bdg_map_sparse = sparse(bdg_map)
        if nev > 0
            prob = LS.LinearProblem(bdg_map_sparse, similar(bdg_map_sparse, size(bdg_map_sparse, 1)))
            linsolve = LS.init(prob, LS.UMFPACKFactorization())
            linmap = LinSolveLinMap{T, typeof(linsolve)}(linsolve, size(bdg_map_sparse))
            ps, info = partialschur(linmap; nev, which=:LM)
            verbose && @show info
            vals, vecs = partialeigen(ps)
            return inv.(vals), vecs
        else
            ps, info = partialschur(bdg_map_sparse; nev=size(bdg_map, 1), which=:LM)
            verbose && @show info
            return partialeigen(ps)
        end
    else # dense
        bdg_map_dense = Matrix(bdg_map)
        if nev > 0
            prob = LS.LinearProblem(bdg_map_dense, similar(bdg_map_dense, size(bdg_map_dense, 1)))
            linsolve = LS.init(prob, LS.LUFactorization())
            linmap = LinSolveLinMap{T, typeof(linsolve)}(linsolve, size(bdg_map_dense))
            ps, info = partialschur(linmap; nev, which=:LM)
            verbose && @show info
            vals, vecs = partialeigen(ps)
            return @views inv.(vals[1:nev]), vecs[:, 1:nev]
        else
            F = eigen(bdg_map_dense)
            return F.values, F.vectors
        end
    end
end

"Helepr function for wrapping `BdGMap` in a SciMLOperator. `params` will be a `BdGMap` object."
function BdGMap!(w, v, u, params, t)
    mul!(w, params, v)
end

"""
Compute BdG stability spectrum and eigenfunctions for an x-space state `ψ` for quasimomenta `qs = [qxs, qys, …]`.
By default, calculate all eigenvalues using a dense map.
Alternatively, pass `nev` number of smallest-magnitude eigenvalues to calculate.
Additionally, `storage=:sparse` or `storage=:lazy` will use sparse or matrix-free representations respectively.
`ψ` can be a vector or a N×1 matrix (where N is the number of x points).
The interaction strength `g` can be passed as a scalar number in the 1-component case.
The chemical potential `μ` can be passed as a vector, or a number if it is the same for all components (including the 1-component case).
Return a tuple `vals, vecs` in the format `vals[n, iqx, iqy, ...]`, `vecs[:, n, iqx, iqy, ...]`.
"""
function bdg_spectrum(xh::PSpaceHamiltonian{Storage, R}, ψ::AbstractVecOrMat{<:Union{R, Complex{R}}}, g::Union{R, AbstractMatrix{R}}, μ::Union{R, AbstractVector{<:R}}, qs::AbstractVector{<:AbstractVector{<:Real}};
                      storage=:dense, nev::Integer=0, verbose::Bool=false) where {Storage, R}
    μ_input = μ isa R ? fill(μ, xh.nc) : μ # if only one μ is passed, then construct a vector of same values; also, 1-component case needs a vector
    g_input = g isa R ? [g;;] : g # constructor needs a vector, so make one in the 1-component case
    bdg_map = BdGMap(xh, ψ, g_input, μ_input; floquet=true)

    (;M, xlims, L, δ, nc, H, 𝑈, 𝑈_iseven, 𝐴, Γ) = xh
    D = length(xlims)
    B = (2M + 1)^D # block size
    nsaves = nev == 0 ? 2B*nc : nev # number of eigenvalues and eigenvectors to store: if `nev` is zero (or not passed), then store all
    vals = Array{Complex{R}}(undef, nsaves, ntuple(i -> length(qs[i]), D)...) # vals[n, iqx, iqy, ...] = `n`th eigenvalue at momentum at indices (`iqx`, `iqy`)
    vecs = Array{Complex{R}}(undef, 2B*nc, nsaves, ntuple(i -> length(qs[i]), D)...) # vecs[:, n, iqx, iqy, ...] = `n`th eigenvector at momentum at indices (`iqx`, `iqy`)
    
    QS = Vector{R}(undef, length(qs)) # at each iteration will contain the values of quasimomenta, e.g. in 2D it will contain [qx, qy], where we defined qx ≡ qs[1], qy ≡ qs[2]
    for IQ in Iterators.product(eachindex.(qs)...) # example in 2D: IQ = (iqx, iqy), where iqx is an index of qx and iqy is an index of qy
        for i in eachindex(QS)
            QS[i] = qs[i][IQ[i]]
        end
        update_H!(bdg_map, xh, QS)
        vals[:, IQ...], vecs[:, :, IQ...] = diagonalize(bdg_map, ψ; storage, nev, verbose)
    end
    
    return vals, vecs
end

################ p-space approach (the BdG matrix is constructed explicitly in p-space) ################

"""
Compute BdG stability spectrum and eigenfunctions for an x-space state `ψ` (1-component case).
If `nev > 0`, calculate only `nev` eigenvalues of of type `whichvals` (`:LI` = largest imaginary by default).
`xh` must contain half the number of harmonics of `ψ` (because having N points in `ψ` we can only construct a p-space operator of size N/2).
`ψ` can be a vector or a N×1 matrix (where N is the number of x points).
"""
function bdg_spectrum_pspace(xh::PSpaceHamiltonian{Storage, R}, ψ::AbstractVecOrMat{<:Union{R, Complex{R}}}, g::AbstractFloat, μ::AbstractFloat; ψ_iseven=false, nev::Integer=0, whichvals::Symbol=:LI, verbose::Bool=false) where {Storage, R}
    (;xlims, M, basis) = xh
    # transform `ψ2` to p-space
    ψ_isreal = eltype(ψ) <: Real
    ψ2 = g .* ψ.^2
    ft = FourierTransformerP(xlims, M; basis, target_real=ψ_isreal, target_rank=2) # the constructed matrix will correspond to `M`
    transform!(ft, ψ2)
    v2 = fft_to_operator(ft; makereal=(ψ_iseven && ψ_isreal))
    if ψ_isreal
        vconj2 = v2
        vabs2 = 2 .* v2
    else
        # transform `conj(ψ2)` to p-space
        transform!(ft, conj(ψ2))
        vconj2 = fft_to_operator(ft)
        # transform `ψabs2` to p-space
        ψabs2 = abs2.(ψ)
        ft = FourierTransformerP(xlims, M; basis, target_real=true, target_rank=2)
        transform!(ft, ψabs2)
        vabs2 = fft_to_operator(ft)
    end
    # construct the matrix
    A11 = xh.H - μ*LA.I + vabs2
    A = [A11     v2
         -vconj2 -A11]
    if nev == 0
        vals, vecs = eigen(A)
    else
        ps, info = partialschur(A; nev, which=whichvals, restarts=200) # note: no shift-invert
        verbose && @show info
        vals, vecs = partialeigen(ps)
    end
    return vals, vecs
end

"""
Compute BdG stability spectrum and eigenfunctions for an x-space state `ψ` (2-component case).
If `nev > 0`, calculate only `nev` eigenvalues of smallest magnitude.
`xh` must contain half the number of harmonics of `ψ`.
`ψ` must be a N×nc matrix, where `N` is the number of x points and `nc` is the number of components.
The chemical potential `μ` can be passed as a vector, or a number if it is the same for all components.
`q` is the quasimomentum vector [qx, qy, …], zero by default.
Note that the off-diagonal blocks of `xh.H` are not taken into account at all (because one needs to figure out conjugation).
"""
function bdg_spectrum_pspace(xh::PSpaceHamiltonian{Storage, R}, ψ::AbstractMatrix{<:Union{R, Complex{R}}}, g::AbstractMatrix{<:AbstractFloat}, μ::Union{R, AbstractVector{<:R}}, q=zeros(R, length(xh.xlims));
                             nev::Integer=0, verbose::Bool=false) where {Storage, R}
    (;xlims, M, basis, H, nc) = xh
    μs = μ isa R ? fill(μ, nc) : μ # if only one μ is passed, then construct a vector of same values
    ψ_isreal = eltype(ψ) <: Real
    ft = FourierTransformerP(xlims, M; basis, target_real=ψ_isreal, target_rank=2) # the constructed matrix will correspond to `2M` -- internally it will use twice because target_rank=2
    transform!(ft, ψ[:, 1].^2)
    v₁² = fft_to_operator(ft)
    transform!(ft, ψ[:, 2].^2)
    v₂² = fft_to_operator(ft)
    transform!(ft, ψ[:, 1].*ψ[:, 2])
    v₁v₂ = fft_to_operator(ft)
    if ψ_isreal
        v₁⁺² = v₁² # "+" means conjugate
        V₁²  = v₁² # uppercase means modulus
        v₂⁺² = v₂²  
        V₂²  = v₂²
        v₁⁺v₂⁺ = v₁v₂
        v₁v₂⁺ = v₁v₂
        v₁⁺v₂ = v₁v₂
    else
        transform!(ft, conj(ψ[:, 1]).^2)
        v₁⁺² = fft_to_operator(ft)
        transform!(ft, conj(ψ[:, 2]).^2)
        v₂⁺² = fft_to_operator(ft)
        
        ft_real = FourierTransformerP(xlims, M; basis, target_real=true, target_rank=2)
        transform!(ft_real, abs2.(ψ[:, 1]))
        V₁² = fft_to_operator(ft_real)
        transform!(ft_real, abs2.(ψ[:, 2]))
        V₂² = fft_to_operator(ft_real)

        transform!(ft, conj.(ψ[:, 1]) .* conj.(ψ[:, 2]))
        v₁⁺v₂⁺ = fft_to_operator(ft)
        transform!(ft, ψ[:, 1] .* conj.(ψ[:, 2]))
        v₁v₂⁺ = fft_to_operator(ft)
        transform!(ft, conj.(ψ[:, 1]) .* ψ[:, 2])
        v₁⁺v₂ = fft_to_operator(ft)
    end
    B = size(v₁², 1) # our usual blocksize -- number of points corresponding to each (of the two) components -- of xh_half
    A = Matrix{eltype(v₁²)}(undef, 4B, 4B)
    block(a, b) = CartesianIndices(((a-1)B+1:a*B, (b-1)B+1:b*B))

    # Blocks (1, 1) and (3, 3) require special treatment if quasimomenta are passed
    A₁₁ = @view A[block(1, 1)]
    A₃₃ = @view A[block(3, 3)]
    if basis == :cis && !iszero(q) # then need to include quasimomentum on the diagonal of `A`
        # from the diagonal of each diagonal block of `H`, extract (𝑈ᵢᵢ)₀ (the 0th harmonic of 𝑈ᵢᵢ) plus decay -iΓ/2
        U_diags = [H[(c-1)B + B÷2+1, (c-1)B + B÷2+1] for c in 1:nc] # generally, `Hᵢᵢ = -Δᵢᵢ + Uᵢᵢ - iΓ/2`, but Δᵢᵢ = 0 for the central element of the diagonal

        # block (1, 1)
        copyto!(A, block(1, 1), H, block(1, 1))
        p² = make_p²_matrix(xh.L, xh.M, xh.δ, basis, q) |> parent
        A₁₁[diagind(A₁₁)] .= p² .+ U_diags[1] .- μs[1]
        @. A₁₁ += 2g[1,1]V₁² + g[1,2]V₂²

        # block (3, 3)
        copyto!(A, block(3, 3), H, block(2, 2))
        A₃₃[diagind(A₃₃)] .= p² .+ U_diags[2] .- μs[2]
        @. A₃₃ += 2g[2,2]V₂² + g[2,1]V₁²
    else
        # block (1, 1)
        @. A₁₁ = H[block(1, 1)] + 2g[1,1]V₁² + g[1,2]V₂²
        A₁₁[diagind(A₁₁)] .-= μs[1]

        # block (3, 3)
        @. A₃₃ = H[block(2, 2)] + 2g[2,2]V₂² + g[2,1]V₁²
        A₃₃[diagind(A₃₃)] .-= μs[2]
    end

    @. A[block(1, 2)] = g[1,1]v₁²
    @. A[block(1, 3)] = g[1,2]v₁v₂⁺
    @. A[block(1, 4)] = g[1,2]v₁v₂
    @. A[block(2, 1)] = -g[1,1]v₁⁺²
    @. A[block(2, 2)] = -A₁₁ # assumes `xh.H` is real in x-space
    @. A[block(2, 3)] = -g[1,2]v₁⁺v₂⁺
    @. A[block(2, 4)] = -g[1,2]v₁⁺v₂
    @. A[block(3, 1)] = g[2,1]v₁⁺v₂
    @. A[block(3, 2)] = g[2,1]v₁v₂
    @. A[block(3, 4)] = g[2,2]v₂²
    @. A[block(4, 1)] = -g[2,1]v₁⁺v₂⁺
    @. A[block(4, 2)] = -g[2,1]v₁v₂⁺
    @. A[block(4, 3)] = -g[2,2]v₂⁺²
    @. A[block(4, 4)] = -A₃₃ # assumes `xh.H` is real in x-space

    if nev == 0
        vals, vecs = eigen(A)
    else
        prob = LS.LinearProblem(A, similar(A, size(A, 1)))
        linsolve = LS.init(prob, LS.LUFactorization())
        linmap = LinSolveLinMap{eltype(A), typeof(linsolve)}(linsolve, size(A))
        ps, info = partialschur(linmap; nev, which=:LM)
        verbose && @show info
        vals, vecs = partialeigen(ps)
        return inv.(vals), vecs
    end
    return vals, vecs
end