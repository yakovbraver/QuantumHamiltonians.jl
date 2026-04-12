"""
A lazy linear map describing the action of the BdG operator on an x-space vector
    𝑐 = (𝑎₁(x), …, 𝑎ₙ(x), 𝑏₁(x), …, 𝑏ₙ(x))
"""
struct BdGMap{T,H,P,R,FT_FORWARD,FT_BACKWARD} <: LM.LinearMap{T}
    Hₚ::H
    H⁺ₚ::H
    G::Matrix{Vector{P}} # An analogue of the BdG matrix
    nc::Int # number of components 
    B::Int # block size
    ψₚ_buff1_real::Vector{R} # real buffer for storing ψₚ, needed in the sin/cos cases when acting on real vectors
    ψₚ_buff2_real::Vector{R} # real buffer for storing Hₚ*ψₚ, needed in the sin/cos cases when acting on real vectors
    ψₚ_buff1_complex::Vector{Complex{R}} # complex buffer for storing ψₚ, needed when acting on complex vectors
    ψₚ_buff2_complex::Vector{Complex{R}} # complex buffer for storing Hₚ*ψₚ, needed when acting on complex vectors
    ft_forward::FT_FORWARD
    ft_backward::FT_BACKWARD
    size::Dims{2} # size that the map would have were it a concrete matrix. For us it's double the size of `Hₚ`
end

Base.size(lm::BdGMap) = lm.size

"""
Construct a `BdGMap` object for analysing stability of x-space discretised state `f`, containing all components in a contiguous vector.
Pass `floquet=true` if map is to be used for a periodic problem. This affects the resulting type of the map.
"""
function BdGMap(xh::XSpaceHamiltonian{Storage, R}, f::AbstractVector{S}, g::AbstractMatrix{R}, μ::AbstractVector{R}; floquet=false) where {Storage, R, S}
    # eltype of the `G` matrix: real if Hamiltonian is real in x-space and `f` is real; complex otherwise
    P = all(isnothing.(xh.𝐴)) && S <: Real ? R : Complex{R} # TODO determination of whether Hamiltonian is real in incorrect because off-diagonal blocks can be complex
    # eltype of the `G` matrix: real if Hamiltonian is real in x-space and `f` is real; complex otherwise
    T = floquet ? Complex{R} : P # in the Floquet case, map is always complex in the coordinate space (because of (-i∇ + q)²); otherwise, the type is `P`

    if xh.nc == 1
        #  (𝐻  - 𝜇 + 2𝑔|𝑓|²)𝑎 + 𝑔𝑓²𝑏  = i𝜆𝑎
        # -(𝐻⁺ - 𝜇 + 2𝑔|𝑓|²)𝑏 - 𝑔𝑓⁺²𝑎 = i𝜆𝑏
        G = [similar(f, P) for _ in 1:2, _ in 1:2]
        @. G[1, 1] = 2g[1] * abs2(f) - μ[1]
        @. G[1, 2] = g[1] * f^2
        @. G[2, 1] = -conj(G[1, 2])
        @. G[2, 2] = -G[1, 1]
    elseif xh.nc == 2
        # (𝐻𝑐)₁ + (2𝑔₁₁|𝑓₁|² + 𝑔₁₂|𝑓₂|² - 𝜇)𝑎₁ + 𝑔₁₂𝑓₁𝑓₂⁺𝑎₂ + 𝑔₁₁𝑓₁²𝑏₁ + 𝑔₁₂𝑓₁𝑓₂𝑏₂ = i𝜆𝑎₁ 
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
        println("BdGMap not implemented for $(xh.nc) components")
    end

    # prepare the plans that can transform either real or complex vectors (hence `target_real=false`), because the map might need to act on complex ones during diagonalisation
    ft_forward  = FourierTransformer(xh.xlims, xh.M; xh.basis, target_real=false, target_rank=1, isforward=true)
    ft_backward = FourierTransformer(xh.xlims, xh.M; xh.basis, target_real=false, target_rank=1, isforward=false)
    
    ψₚ_buff1_real = similar(f, R)
    ψₚ_buff2_real = similar(f, R)
    ψₚ_buff1_complex = similar(f, Complex{R})
    ψₚ_buff2_complex = similar(f, Complex{R})

    Hₚ = copy(xh.H) # could be a reference, but we modify it when scanning quasimomenta
    H⁺ₚ = copy(xh.H) # reference the same Hamiltonian for now: TODO implement conjugation
    B = size(xh.H, 1) ÷ xh.nc # block size
    return BdGMap{T, typeof(Hₚ), P, R, typeof(ft_forward), typeof(ft_backward)}(
        Hₚ, H⁺ₚ, G, xh.nc, B, ψₚ_buff1_real, ψₚ_buff2_real, ψₚ_buff1_complex, ψₚ_buff2_complex, ft_forward, ft_backward, size(xh.H) .* 2
    )
end

"""
Apply the Hamiltonian and the BdG matrix to `ψ_in`, storing the result in `ψ_out`.
The format of `ψ_in` is (𝑎₁, …, 𝑎ₙ, 𝑏₁, …, 𝑏ₙ) where 𝑛 is the number of components.
"""
function LM._unsafe_mul!(ψ_out, bdg_map::BdGMap{T}, ψ_in::AbstractVector) where T
    (;Hₚ, H⁺ₚ, G, nc, B, ft_forward, ft_backward) = bdg_map
    block(i) = (i-1)B+1:i*B
    # c_in = [@view ψ_in[(i-1)q+1:i*q] for i in 1:4] # holds (𝑎₁, 𝑎₂, 𝑏₁, 𝑏₂)
    # c_out = [@view ψ_out[(i-1)q+1:i*q] for i in 1:4]
    
    ψ_in_isreal = eltype(ψ_in) <: Real
    if ft_forward.basis == :cis || !ψ_in_isreal || T <: Complex # then dealing with complex functions, so will be using the complex buffers `ψₚ_buff1`, `ψₚ_buff2`
        ψₚ_buff1, ψₚ_buff2 = bdg_map.ψₚ_buff1_complex, bdg_map.ψₚ_buff2_complex
    else
        ψₚ_buff1, ψₚ_buff2 = bdg_map.ψₚ_buff1_real, bdg_map.ψₚ_buff2_real
    end

    # transform `ψ_in` to p-space, multiply by `H` and transform back, writing into `ψ_out`
    @views for i in 1:2 # i = 1 iterates 𝑎₁, …, 𝑎ₙ; i = 2 iterates 𝑏₁, …, 𝑏ₙ
        # transform 𝑎₁, …, 𝑎ₙ to p-space, writing into appropriate parts of `ψₚ_buff1`
        for j in 1:nc
            transform!(ft_forward, ψ_in[block((i-1)nc+j)])
            fft_to_vector!(ψₚ_buff1[block(j)], ft_forward)
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
            transform!(ft_backward, ψₚ_buff2[block(j)])
            fft_to_vector!(ψ_out[block((i-1)nc+j)], ft_backward; makereal=(ψ_in_isreal && T <: Real)) # if initial ψ is real and map also, then make the result real
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
function update_H!(bdg_map::BdGMap{T}, xh::XSpaceHamiltonian, q::AbstractVector{<:Real}) where {T}
    Hₚ_diag  = diagview(bdg_map.Hₚ)
    H⁺ₚ_diag = diagview(bdg_map.H⁺ₚ)
    p²  = make_p²(xh.L, xh.M, xh.δ, :cis, q) |> parent # `parent` returns the diagonal as a vector
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
function bdg_spectrum(xh::XSpaceHamiltonian{Storage, R}, ψ::AbstractVecOrMat{<:Union{R, Complex{R}}}, g::Union{R, AbstractMatrix{R}}, μ::Union{R, AbstractVector{R}};
                      storage=:dense, nev::Integer=0, verbose::Bool=false) where {Storage, R}

    μ_input = μ isa R ? fill(μ, xh.nc) : μ # if only one μ is passed, then construct a vector of same values; also, 1-component case needs a vector
    g_input = g isa R ? [g;;] : g # constructor needs a vector, so make one in the 1-component case
    bdg_map = BdGMap(xh, ψ, g_input, μ_input)
    
    diagonalize(bdg_map, ψ; storage, nev, verbose)
end

"Diagonalise `bdg_map`."
function diagonalize(bdg_map::BdGMap, ψ::AbstractVecOrMat{<:Union{R, Complex{R}}}; storage=:dense, nev::Integer=0, verbose::Bool=false) where R
    if storage == :lazy
        if nev > 0
            # Here we do shift-invert. "Inversion" is actually solution of a linear system. But `LS.LinearProblem` does not work with LinearMaps, so we wrap `bdg_map` in a SciMLOperator
            bdg_op = SciMLOperators.FunctionOperator(BdGMap!, similar(ψ, 2length(ψ)); p=bdg_map, isconstant=true)
            prob = LS.LinearProblem(bdg_op, similar(ψ, 2length(ψ)))
            linsolve = LS.init(prob, LS.KrylovJL_BICGSTAB())
            linmap = LinSolveLinMap{eltype(bdg_map.Hₚ), typeof(linsolve)}(linsolve, size(bdg_map))
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
            linmap = LinSolveLinMap{eltype(bdg_map.Hₚ), typeof(linsolve)}(linsolve, size(bdg_map_sparse))
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
            linmap = LinSolveLinMap{eltype(bdg_map.Hₚ), typeof(linsolve)}(linsolve, size(bdg_map_dense))
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
"""
function bdg_spectrum(xh::XSpaceHamiltonian{Storage, R}, ψ::AbstractVecOrMat{<:Union{R, Complex{R}}}, g::Union{R, AbstractMatrix{R}}, μ::Union{R, AbstractVector{<:R}}, qs::AbstractVector{<:AbstractVector{<:Real}};
                      storage=:dense, nev::Integer=0, verbose::Bool=false) where {Storage, R}
    μ_input = μ isa R ? fill(μ, xh.nc) : μ # if only one μ is passed, then construct a vector of same values; also, 1-component case needs a vector
    g_input = g isa R ? [g;;] : g # constructor needs a vector, so make one in the 1-component case
    bdg_map = BdGMap(xh, ψ, g_input, μ_input; floquet=true)

    (;M, xlims, L, δ, nc, H, 𝑈, 𝑈_iseven, 𝐴, Γ) = xh
    D = length(xlims)
    B = (2M + 1)^D # block size
    nsaves = nev == 0 ? 2B*nc : nev # number of eigenvalues and eigenvectors to store: if `nev` is zero (or not passed), then store all
    vals = Array{Complex{R}}(undef, nsaves, ntuple(i -> length(qs[i]), D)...) # vals[n, iqx, iqy, ...] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    vecs = Array{Complex{R}}(undef, 2B*nc, nsaves, ntuple(i -> length(qs[i]), D)...) # vecs[:, n, iqx, iqy, ...] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)
    
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