abstract type XSpaceHamiltonian{S} end

function XSpaceHamiltonian{:dense}(𝑈::AbstractMatrix{<:Union{Function,Nothing}}, xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                                   𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1)),
                                   𝐴_x::Union{Function,Nothing}=nothing, 𝐴_y::Union{Function,Nothing}=nothing) where R <: Real
    return DenseHamiltonian2D(𝑈,xlims, ylims; isperiodic, M, δ, 𝑈_iseven, Γ, 𝐴_x, 𝐴_y)
end

function XSpaceHamiltonian{:sparse}(𝑈::AbstractMatrix{<:Union{Function,Nothing}}, xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                                    𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1)),
                                    𝐴_x::Union{Function,Nothing}=nothing, 𝐴_y::Union{Function,Nothing}=nothing, fft_threshold::R=√eps(R)) where R <: Real
    return SparseHamiltonian2D(𝑈, xlims, ylims; isperiodic, M, δ, 𝑈_iseven, Γ, 𝐴_x, 𝐴_y, fft_threshold)
end