"""
A lazy linear map describing the action of the BdG operator on an x-space vector (𝑎(x), 𝑏(x)) in the 1-component GPE case according to
     (𝐻 - 𝜇 + 2𝑔|𝑓₀|²)𝑎 + 𝑔𝑓₀²𝑏  = iλ𝑎
    -(𝐻 - 𝜇 + 2𝑔|𝑓₀|²)𝑏 - 𝑔𝑓₀⁺²𝑎 = iλ𝑏
"""
struct BdGMap1comp{T,R,H,FT_FORWARD,FT_BACKWARD} <: LinearMaps.LinearMap{T}
    Hₚ::H
    G::Matrix{Vector{T}} # An analogue of the BdG matrix
    ψₚ_buff1::Vector{<:Union{R, Complex{R}}} # buffer for storing ψₚ
    ψₚ_buff2::Vector{<:Union{R, Complex{R}}} # buffer for storing Hₚ*ψₚ
    ft_forward::FT_FORWARD
    ft_backward::FT_BACKWARD
    size::Dims{2} # size that the map would have were it a concrete matrix. For us it's double the size of `Hₚ`
end

Base.size(lm::BdGMap1comp) = lm.size

function BdGMap1comp(μ::R, g::R, xh::XSpaceHamiltonian, f₀::AbstractVector{S}) where {R <: Real, S}
    # eltype of map: real if Hamiltonian is real in x-space and f₀ is real; complex otherwise
    T = all(isnothing.(xh.𝐴)) && S <: Real ? R : Complex{R}
    
    G = [similar(f₀, T) for _ in 1:2, _ in 1:2]
    @. G[1, 1] = 2g * abs2(f₀) - μ
    @. G[1, 2] = g * f₀^2
    @. G[2, 1] = -conj(G[1, 2])
    @. G[2, 2] = -G[1, 1]

    # prepare the plans that can transform either real or complex vectors (hence `target_real=false`), because the map might need to act on complex ones during diagonalisation
    ft_forward  = FourierTransformer(xh.xlims, xh.M; xh.basis, target_real=false, target_rank=1, isforward=true)
    ft_backward = FourierTransformer(xh.xlims, xh.M; xh.basis, target_real=false, target_rank=1, isforward=false)
    
    # cis case: always need 2 complex buffers.
    # sin/cos case: always need 2 real buffers when acting on real function but 2 complex when acting on complex
    # but maybe we could also use the buffers of ft objects... For now we assume cis case
    ψₚ_buff1 = similar(f₀, Complex{R})
    ψₚ_buff2 = similar(f₀, Complex{R})

    return BdGMap1comp(xh.H, G, ψₚ_buff1, ψₚ_buff2, ft_forward, ft_backward, size(xh.H) .* 2)
end

function LinearMaps._unsafe_mul!(ψ_out, bdg_map::BdGMap1comp{T}, ψ_in::AbstractVector) where T
    (;Hₚ, G, ψₚ_buff1, ψₚ_buff2, ft_forward, ft_backward) = bdg_map
    a_in = @view ψ_in[1:end÷2]
    b_in = @view ψ_in[end÷2+1:end]
    a_out = @view ψ_out[1:end÷2]
    b_out = @view ψ_out[end÷2+1:end]
    
    ### calculate `a_out`
    # transform `ψ` to p-space, multiply by `H` and transform back
    transform!(ft_forward, a_in)
    fft_to_vector!(ψₚ_buff1, ft_forward)
    mul!(ψₚ_buff2, Hₚ, ψₚ_buff1)
    transform!(ft_backward, ψₚ_buff2)
    # if initial ψ is real and map also, then make the result real
    @views fft_to_vector!(a_out, ft_backward; makereal=(eltype(ψ_in) <: Real && T <: Real))
    @. a_out += G[1, 1] * a_in + G[1, 2] * b_in

    ### calculate `b_out`
    # transform `ψ` to p-space, multiply by `H` and transform back
    transform!(ft_forward, b_in)
    fft_to_vector!(ψₚ_buff1, ft_forward)
    mul!(ψₚ_buff2, Hₚ, ψₚ_buff1)
    @. ψₚ_buff2 = -ψₚ_buff2 # because we should have multiplied by -Hₚ
    transform!(ft_backward, ψₚ_buff2)
    # if initial ψ is real and map also, then make the result real
    @views fft_to_vector!(b_out, ft_backward; makereal=(eltype(ψ_in) <: Real && T <: Real))
    @. b_out += G[2, 2] * b_in + G[2, 1] * a_in
    return
end

"""
Compute BdG stability spectrum and eigenfunctions for an x-space state `ψ` (1-component case).
Calculate `nev` eigenvalues of of type `whichvals` (`:LI` = largest imaginary by default).
`ψ` can be a vector or a N×1 matrix (where N is the number of x points).
"""
function bdg_spectrum_xspace(xh::XSpaceHamiltonian{Storage, R}, ψ::AbstractVecOrMat{<:Union{R, Complex{R}}}, g::AbstractFloat, μ::AbstractFloat; nev::Integer, whichvals::Symbol=:LI, verbose::Bool=false) where {Storage, R}
    bdg_map = BdGMap1comp(μ, g, xh, ψ)
    ps, info = partialschur(bdg_map; nev, which=whichvals);
    verbose && @show info
    return partialeigen(ps)
end