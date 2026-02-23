"""
A lazy linear map describing the action of the BdG operator on an x-space vector (𝑎(x), 𝑏(x)) in the 1-component GPE case according to
     (𝐻 - 𝜇 + 2𝑔|𝑓|²)𝑎 + 𝑔𝑓²𝑏  = iλ𝑎
    -(𝐻 - 𝜇 + 2𝑔|𝑓|²)𝑏 - 𝑔𝑓⁺²𝑎 = iλ𝑏
where 𝑓(𝑥) is the state whose stability is investigated.
"""
struct BdGMap1comp{T,R,H,FT_FORWARD,FT_BACKWARD} <: LM.LinearMap{T}
    Hₚ::H
    G::Matrix{Vector{T}} # An analogue of the BdG matrix
    ψₚ_buff1::Vector{Complex{R}} # buffer for storing ψₚ
    ψₚ_buff2::Vector{Complex{R}} # buffer for storing Hₚ*ψₚ
    ft_forward::FT_FORWARD
    ft_backward::FT_BACKWARD
    size::Dims{2} # size that the map would have were it a concrete matrix. For us it's double the size of `Hₚ`
end

Base.size(lm::BdGMap1comp) = lm.size

"Construct a `BdGMap1comp` object for analysing stability of x-space discretised state `f`."
function BdGMap1comp(μ::R, g::R, xh::XSpaceHamiltonian{Storage, R}, f::AbstractVector{S}) where {Storage, R, S}
    # eltype of map: real if Hamiltonian is real in x-space and f is real; complex otherwise
    T = all(isnothing.(xh.𝐴)) && S <: Real ? R : Complex{R} # TODO determination of whether Hamiltonian is real in incorrect because off-diagonal blocks can be complex
    
    G = [similar(f, T) for _ in 1:2, _ in 1:2]
    @. G[1, 1] = 2g * abs2(f) - μ
    @. G[1, 2] = g * f^2
    @. G[2, 1] = -conj(G[1, 2])
    @. G[2, 2] = -G[1, 1]

    # prepare the plans that can transform either real or complex vectors (hence `target_real=false`), because the map might need to act on complex ones during diagonalisation
    ft_forward  = FourierTransformer(xh.xlims, xh.M; xh.basis, target_real=false, target_rank=1, isforward=true)
    ft_backward = FourierTransformer(xh.xlims, xh.M; xh.basis, target_real=false, target_rank=1, isforward=false)
    
    # cis case: always need 2 complex buffers.
    # sin/cos case: need 2 complex when acting on complex function. When acting on real function, we can use `ft_forward.buff` and `ft_backward.buff`
    ψₚ_buff1 = similar(f, Complex{R})
    ψₚ_buff2 = similar(f, Complex{R})

    return BdGMap1comp(xh.H, G, ψₚ_buff1, ψₚ_buff2, ft_forward, ft_backward, size(xh.H) .* 2)
end

function LM._unsafe_mul!(ψ_out, bdg_map::BdGMap1comp{T}, ψ_in::AbstractVector) where T
    (;Hₚ, G, ψₚ_buff1, ψₚ_buff2, ft_forward, ft_backward) = bdg_map
    a_in = @view ψ_in[1:end÷2]
    b_in = @view ψ_in[end÷2+1:end]
    a_out = @view ψ_out[1:end÷2]
    b_out = @view ψ_out[end÷2+1:end]
    
    ψ_in_isreal = eltype(ψ_in) <: Real

    ### calculate `a_out`
    # transform `ψ` to p-space, multiply by `H` and transform back
    transform!(ft_forward, a_in)
    if ft_forward.basis == :cis || !ψ_in_isreal # then dealing with complex functions, so will be using the complex buffers `ψₚ_buff1`, `ψₚ_buff2`
        fft_to_vector!(ψₚ_buff1, ft_forward)
        mul!(ψₚ_buff2, Hₚ, ψₚ_buff1)
        transform!(ft_backward, ψₚ_buff2)
    else # basis is sin/cos and ψ_in_isreal -- then we can use the buffers in `ft_forward` and `ft_backward`
        fft_to_vector!(ft_backward.buff, ft_forward)
        mul!(ft_forward.buff, Hₚ, ft_backward.buff)
        transform!(ft_backward, ft_forward.buff)
    end
    fft_to_vector!(a_out, ft_backward; makereal=(ψ_in_isreal && T <: Real)) # if initial ψ is real and map also, then make the result real
    @. a_out += G[1, 1] * a_in + G[1, 2] * b_in

    ### calculate `b_out`
    # transform `ψ` to p-space, multiply by `H` and transform back
    transform!(ft_forward, b_in)
    if ft_forward.basis == :cis || !ψ_in_isreal # then dealing with complex functions, so will be using the complex buffers `ψₚ_buff1`, `ψₚ_buff2`
        fft_to_vector!(ψₚ_buff1, ft_forward)
        mul!(ψₚ_buff2, Hₚ, ψₚ_buff1)
        @. ψₚ_buff2 = -ψₚ_buff2 # because we should have multiplied by -Hₚ
        transform!(ft_backward, ψₚ_buff2)
    else # basis is sin/cos and ψ_in_isreal -- then we can use the buffers in `ft_forward` and `ft_backward`
        fft_to_vector!(ft_backward.buff, ft_forward)
        mul!(ft_forward.buff, Hₚ, ft_backward.buff)
        @. ft_forward.buff = -ft_forward.buff # because we should have multiplied by -Hₚ
        transform!(ft_backward, ft_forward.buff)
    end
    fft_to_vector!(b_out, ft_backward; makereal=(ψ_in_isreal && T <: Real))
    @. b_out += G[2, 2] * b_in + G[2, 1] * a_in
    return
end

"""
A lazy linear map describing the action of the BdG operator on an x-space vector
    𝑐 = (𝑎₁(x), …, 𝑎ₙ(x), 𝑏₁(x), …, 𝑏ₙ(x))
"""
struct BdGMap{T,R,H,FT_FORWARD,FT_BACKWARD} <: LM.LinearMap{T}
    Hₚ::H
    H⁺ₚ::H
    G::Matrix{Vector{T}} # An analogue of the BdG matrix
    nc::Int # number of components 
    B::Int # block size
    ψₚ_buff1_real::Vector{R} # real buffer for storing ψₚ, needed in the sin/cos cases when acting on real vectors
    ψₚ_buff2_real::Vector{R} # real buffer for storing Hₚ*ψₚ, needed in the sin/cos cases when acting on real vectors
    ψₚ_buff1_complex::Vector{Complex{R}} # complex buffer for storing ψₚ, needed in when acting on complex vectors
    ψₚ_buff2_complex::Vector{Complex{R}} # complex buffer for storing Hₚ*ψₚ, needed in when acting on complex vectors
    ft_forward::FT_FORWARD
    ft_backward::FT_BACKWARD
    size::Dims{2} # size that the map would have were it a concrete matrix. For us it's double the size of `Hₚ`
end

Base.size(lm::BdGMap) = lm.size

"Construct a `BdGMap` object for analysing stability of x-space discretised state `f`."
function BdGMap(xh::XSpaceHamiltonian{Storage, R}, f::AbstractVector{S}, g::AbstractMatrix{R}, μ::AbstractVector{R}) where {Storage, R, S}
    # eltype of map: real if Hamiltonian is real in x-space and f is real; complex otherwise
    T = all(isnothing.(xh.𝐴)) && S <: Real ? R : Complex{R} # TODO determination of whether Hamiltonian is real in incorrect because off-diagonal blocks can be complex

    # (𝐻𝑐)₁ + (2𝑔₁₁|𝑓₁|² + 𝑔₁₂|𝑓₂|² - 𝜇)𝑎₁ + 𝑔₁₂𝑓₁𝑓₂⁺𝑎₂ + 𝑔₁₁𝑓₁²𝑏₁ + 𝑔₁₂𝑓₁𝑓₂𝑏₂ = iλ𝑎₁ 
    @views f₁, f₂ = f[1:end÷2], f[end÷2+1:end]
    G = [similar(f₁, T) for _ in 1:4, _ in 1:4]
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

    # prepare the plans that can transform either real or complex vectors (hence `target_real=false`), because the map might need to act on complex ones during diagonalisation
    ft_forward  = FourierTransformer(xh.xlims, xh.M; xh.basis, target_real=false, target_rank=1, isforward=true)
    ft_backward = FourierTransformer(xh.xlims, xh.M; xh.basis, target_real=false, target_rank=1, isforward=false)
    
    ψₚ_buff1_real = similar(f, R)
    ψₚ_buff2_real = similar(f, R)
    ψₚ_buff1_complex = similar(f, Complex{R})
    ψₚ_buff2_complex = similar(f, Complex{R})

    H⁺ₚ = xh.H # reference the same Hamiltonian for now: TODO implement conjugation
    B = size(xh.H, 1) ÷ xh.nc # block size
    return BdGMap(xh.H, H⁺ₚ, G, xh.nc, B, ψₚ_buff1_real, ψₚ_buff2_real, ψₚ_buff1_complex, ψₚ_buff2_complex, ft_forward, ft_backward, size(xh.H) .* 2)
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
Compute BdG stability spectrum and eigenfunctions for an x-space state `ψ`.
By default, calculate all eigenvalues using a dense map.
Alternatively, pass `nev` number of smallest-magnitude eigenvalues to calculate.
Additionally, `storage=:sparse` or `storage=:lazy` will use sparse or matrix-free representations respectively.
`ψ` can be a vector or a N×1 matrix (where N is the number of x points).
The chemical potential `μ` can be passed as a vector, or a number if it is the same for all components.
"""
function bdg_spectrum(xh::XSpaceHamiltonian{Storage, R}, ψ::AbstractVecOrMat{<:Union{R, Complex{R}}}, g::Union{R, AbstractMatrix{R}}, μ::Union{R, AbstractVector{<:R}};
                      storage=:dense, nev::Integer=0, verbose::Bool=false) where {Storage, R}
    if xh.nc == 1
        bdg_map = BdGMap1comp(μ, g, xh, ψ)
    else
        μ_input = μ isa R ? fill(μ, xh.nc) : μ # if only one μ is passed, then construct a vector of same values
        bdg_map = BdGMap(xh, ψ, g, μ_input)
    end
    if storage == :lazy
        if nev > 0
            # Here we do shift-invert. "Inversion" is actually solution of a linear system. But `LS.LinearProblem` does not work with LinearMaps, so we wrap `bdg_map` in a SciMLOperator
            bdg_op = SciMLOperators.FunctionOperator(BdGMap!, similar(ψ, 2length(ψ)); p=bdg_map, isconstant=true)
            prob = LS.LinearProblem(bdg_op, similar(ψ, 2length(ψ)))
            linsolve = LS.init(prob, LS.KrylovJL_BICGSTAB())
            linmap = LinSolveLinMap{eltype(xh.H), typeof(linsolve)}(linsolve, size(bdg_map))
            ps, info = partialschur(linmap; nev, which=:LM)
            verbose && @show info
            vals, vecs = partialeigen(ps)
            return inv.(vals), vecs
        else
            ps, info = partialschur(bdg_map; nev, which=:LM);
            verbose && @show info
            return partialeigen(ps)
        end
    elseif storage == :sparse
        bdg_map_sparse = sparse(bdg_map)
        if nev > 0
            prob = LS.LinearProblem(bdg_map_sparse, similar(bdg_map_sparse, size(bdg_map_sparse, 1)))
            linsolve = LS.init(prob, LS.UMFPACKFactorization())
            linmap = LinSolveLinMap{eltype(xh.H), typeof(linsolve)}(linsolve, size(bdg_map_sparse))
            ps, info = partialschur(linmap; nev, which=:LM)
            verbose && @show info
            vals, vecs = partialeigen(ps)
            return inv.(vals), vecs
        else
            ps, info = partialschur(linmap; nev, which=:LM)
            verbose && @show info
            return partialeigen(ps)
        end
    else # dense
        bdg_map_dense = Matrix(bdg_map)
        if nev > 0
            prob = LS.LinearProblem(bdg_map_dense, similar(bdg_map_dense, size(bdg_map_dense, 1)))
            linsolve = LS.init(prob, LS.LUFactorization())
            linmap = LinSolveLinMap{eltype(xh.H), typeof(linsolve)}(linsolve, size(bdg_map_dense))
            ps, info = partialschur(linmap; nev, which=:LM)
            verbose && @show info
            vals, vecs = partialeigen(ps)
            return inv.(vals), vecs
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
