abstract type XSpaceHamiltonian{S} end

abstract type XSpaceHamiltonian1D{S} <: XSpaceHamiltonian{S} end
abstract type XSpaceHamiltonian2D{S} <: XSpaceHamiltonian{S} end

### 1D Constructors

"Constructor accepting a 𝑈 as a matrix of functions and 𝑈_iseven as a matrix of bools."
function XSpaceHamiltonian{:dense}(𝑈::AbstractMatrix{<:Union{Function,Nothing}}, xlims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                                   𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1))) where R <: Real
    return DenseHamiltonian1D(𝑈, xlims; isperiodic, M, δ, 𝑈_iseven, Γ)
end

"Constructor accepting a 𝑈 as a matrix of functions and 𝑈_iseven as a matrix of bools."
function XSpaceHamiltonian{:dense}(𝑈::Function, xlims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                                   𝑈_iseven::Bool=false, Γ::R=zero(R)) where R <: Real
    return DenseHamiltonian1D([𝑈;;], xlims; isperiodic, M, δ, 𝑈_iseven=[𝑈_iseven;;], Γ=[Γ])
end

### 2D Constructors

"Constructor accepting a 𝑈 as a matrix of functions and 𝑈_iseven as a matrix of bools."
function XSpaceHamiltonian{:dense}(𝑈::AbstractMatrix{<:Union{Function,Nothing}}, xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                                   𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1)),
                                   𝐴_x::Union{Function,Nothing}=nothing, 𝐴_y::Union{Function,Nothing}=nothing) where R <: Real
    return DenseHamiltonian2D(𝑈, xlims, ylims; isperiodic, M, δ, 𝑈_iseven, Γ, 𝐴_x, 𝐴_y)
end

"Single-component constructor accepting a 𝑈 as a function, 𝑈_iseven as a bool, and Γ as a real."
function XSpaceHamiltonian{:dense}(𝑈::Function, xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                                   𝑈_iseven::Bool=false, Γ::R=zero(R),
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
                                    𝑈_iseven::Bool=false, Γ::R=zero(R),
                                    𝐴_x::Union{Function,Nothing}=nothing, 𝐴_y::Union{Function,Nothing}=nothing, fft_threshold::R=√eps(R)) where R <: Real
    return SparseHamiltonian2D([𝑈;;], xlims, ylims; isperiodic, M, δ, 𝑈_iseven=[𝑈_iseven;;], Γ=[Γ], 𝐴_x, 𝐴_y, fft_threshold) # pack `𝑈` and `𝑈_iseven` into matrices, `Γ` into a vector
end

### Functions that are generic for all dimensions

"""
Calculate `nev` lowest eigenvectors and eigenvalues using `ArnoldiMethod`.
Pass `nev=0` for full diagonalisation using `LinearAlgebra`.
The result is written into `xh.ε` and `xh.V`.
"""
function diagonalize!(xh::XSpaceHamiltonian{:dense}; nev::Integer, verbose::Bool=false)
    xh.ε, xh.V = diagonalize(xh; nev, verbose)
end

"""
Calculate `nev` lowest eigenvectors and eigenvalues using `ArnoldiMethod`.
Pass `nev=0` for full diagonalisation using `LinearAlgebra`.
Return a tuple (eigenvalues, eigenvectors).
"""
function diagonalize(xh::XSpaceHamiltonian{:dense}; nev::Integer, verbose::Bool=false)
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
            return ps.eigenvalues, ps.Q
        else
            ps, info = partialschur(dense_linear_map(xh.H); nev, which=:LM)
            verbose && @show info
            ε, V = partialeigen(ps)
            ε .= inv.(ε)
            reverse!(ε) # we want final eigenvalues in ascending order (by abs)
            return ε, V
        end
    end
end

"Helper function for shift-and-invert: construct a linear map that applies the inverse of `A`."
function dense_linear_map(A)
    F = factorize(A) # Bunch-Kaufman for Hermitian `A`, LU otherwise
    LinearMap{eltype(A)}((y, x) -> ldiv!(y, F, x), size(A, 1), ismutating=true)
end