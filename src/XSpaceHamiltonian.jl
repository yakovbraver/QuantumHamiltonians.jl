abstract type XSpaceHamiltonian{S} end

function XSpaceHamiltonian{:dense}(xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                                   𝐻::AbstractMatrix{<:Union{Function,Nothing}}, 𝐻_iseven::AbstractMatrix{Bool}=falses(size(𝐻)), Γ::Vector{R}=zeros(R, size(𝐻, 1)),
                                   𝐴_x::Union{Function,Nothing}=nothing, 𝐴_y::Union{Function,Nothing}=nothing) where R <: Real
    return DenseHamiltonian2D(xlims, ylims; isperiodic, M, δ, 𝐻, 𝐻_iseven, Γ, 𝐴_x, 𝐴_y)
end

function XSpaceHamiltonian{:sparse}(xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                                    𝐻::AbstractMatrix{<:Union{Function,Nothing}}, 𝐻_iseven::AbstractMatrix{Bool}=falses(size(𝐻)), Γ::Vector{R}=zeros(R, size(𝐻, 1)),
                                    𝐴_x::Union{Function,Nothing}=nothing, 𝐴_y::Union{Function,Nothing}=nothing, fft_threshold::R=√eps(R)) where R <: Real
    return SparseHamiltonian2D(xlims, ylims; isperiodic, M, δ, 𝐻, 𝐻_iseven, Γ, 𝐴_x, 𝐴_y, fft_threshold)
end