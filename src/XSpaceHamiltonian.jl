abstract type XSpaceHamiltonian{S} end

"Constructor accepting a 𝑈 as a matrix of functions and 𝑈_iseven as a matrix of bools."
function XSpaceHamiltonian{:dense}(𝑈::AbstractMatrix{<:Union{Function,Nothing}}, xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                                   𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1)),
                                   𝐴_x::Union{Function,Nothing}=nothing, 𝐴_y::Union{Function,Nothing}=nothing) where R <: Real
    return DenseHamiltonian2D(𝑈, xlims, ylims; isperiodic, M, δ, 𝑈_iseven, Γ, 𝐴_x, 𝐴_y)
end

"Single-component constructor accepting a 𝑈 as a function, 𝑈_iseven as a bool, and Γ as a real."
function XSpaceHamiltonian{:dense}(𝑈::Function, xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                                   𝑈_iseven::Bool=false, Γ=zero(R),
                                   𝐴_x::Union{Function,Nothing}=nothing, 𝐴_y::Union{Function,Nothing}=nothing) where R <: Real
    return DenseHamiltonian2D([𝑈;;], xlims, ylims; isperiodic, M, δ, 𝑈_iseven=[𝑈_iseven;;], Γ=[Γ], 𝐴_x, 𝐴_y) # pack `𝑈` and `𝑈_iseven` into matrices, `Γ` into a vector
end

"Constructor accepting a 𝑈 as a matrix of functions and 𝑈_iseven as a matrix of bools."
function XSpaceHamiltonian{:sparse}(𝑈::AbstractMatrix{<:Union{Function,Nothing}}, xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                                    𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1)),
                                    𝐴_x::Union{Function,Nothing}=nothing, 𝐴_y::Union{Function,Nothing}=nothing, fft_threshold::R=√eps(R)) where R <: Real
    return SparseHamiltonian2D(𝑈, xlims, ylims; isperiodic, M, δ, 𝑈_iseven, Γ, 𝐴_x, 𝐴_y, fft_threshold)
end

"Single-component constructor accepting a 𝑈 as a function, 𝑈_iseven as a bool, and Γ as a real."
function XSpaceHamiltonian{:sparse}(𝑈::Function, xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                                    𝑈_iseven::Bool=false, Γ=zero(R),
                                    𝐴_x::Union{Function,Nothing}=nothing, 𝐴_y::Union{Function,Nothing}=nothing, fft_threshold::R=√eps(R)) where R <: Real
    return SparseHamiltonian2D([𝑈;;], xlims, ylims; isperiodic, M, δ, 𝑈_iseven=[𝑈_iseven;;], Γ=[Γ], 𝐴_x, 𝐴_y, fft_threshold) # pack `𝑈` and `𝑈_iseven` into matrices, `Γ` into a vector
end