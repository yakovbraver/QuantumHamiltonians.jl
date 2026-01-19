abstract type XSpaceHamiltonian{S} end

"General constructor. If the problem is 1D, 𝐴 may be passed as a vector, whose elements are treated as corresponding to the different components."
function XSpaceHamiltonian{:dense}(xlims::AbstractVector{Tuple{R,R}},
                                   𝑈::AbstractMatrix{<:Union{Function,Nothing}},
                                   𝐴::AbstractVecOrMat{<:Union{Function,Nothing}}=fill(nothing, size(𝑈, 1), length(xlims));
                                   basis::Symbol, M::Integer, δ::R=one(R),
                                   𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1))) where R <: AbstractFloat
    return DenseHamiltonian(xlims, 𝑈, 𝐴; basis, M, δ, 𝑈_iseven, Γ)
end

"""
1-component (but many-D) constructor accepting a 𝑈 as a function, 𝐴 as a vector (with elements treated as corresponding to the different dimensions),
𝑈_iseven as a bool, and Γ as a real.
"""
function XSpaceHamiltonian{:dense}(xlims::AbstractVector{Tuple{R,R}},
                                   𝑈::Union{Function,Nothing},
                                   𝐴::AbstractVector{<:Union{Function,Nothing}}=fill(nothing, length(xlims));
                                   basis::Symbol, M::Integer, δ::R=one(R),
                                   𝑈_iseven::Bool=false, Γ::R=zero(R)) where R <: AbstractFloat
    return DenseHamiltonian(xlims, [𝑈;;], reshape(𝐴, (1, length(xlims))); basis, M, δ, 𝑈_iseven=[𝑈_iseven;;], Γ=[Γ])
end
    
"Constructor accepting a 𝑈 as a matrix of functions and 𝑈_iseven as a matrix of bools."
function XSpaceHamiltonian{:sparse}(𝑈::AbstractMatrix{<:Union{Function,Nothing}}, xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                                    𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1)), fft_threshold::R=√eps(R),
                                    𝐴_x::AbstractVector{<:Union{Function,Nothing}}=fill(nothing, size(𝑈, 1)), 𝐴_y::AbstractVector{<:Union{Function,Nothing}}=fill(nothing, size(𝑈, 1))) where R <: Real
    return SparseHamiltonian2D(𝑈, xlims, ylims; isperiodic, M, δ, 𝑈_iseven, Γ, 𝐴_x, 𝐴_y, fft_threshold)
end

"Single-component constructor accepting a 𝑈, 𝐴_x, and 𝐴_y as functions, 𝑈_iseven as a bool, and Γ as a real."
function XSpaceHamiltonian{:sparse}(𝑈::Function, xlims::Tuple{R,R}, ylims::Tuple{R,R}; isperiodic::Bool, M::Integer, δ::R=one(R),
                                    𝑈_iseven::Bool=false, Γ::R=zero(R), fft_threshold::R=√eps(R),
                                    𝐴_x::Union{Function,Nothing}=nothing, 𝐴_y::Union{Function,Nothing}=nothing) where R <: Real
    return SparseHamiltonian2D([𝑈;;], xlims, ylims; isperiodic, M, δ, 𝑈_iseven=[𝑈_iseven;;], Γ=[Γ], 𝐴_x=[𝐴_x], 𝐴_y=[𝐴_y], fft_threshold)
end

########## Dense

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

########## Sparse

"""
Calculate `nev` lowest eigenvectors and eigenvalues.
The result is written into `xh.ε` and `xh.V`.
"""
function diagonalize!(xh::XSpaceHamiltonian{:sparse}; nev::Integer, verbose::Bool=false)
    xh.ε, xh.V = diagonalize(xh; nev, verbose)
end

"""
Calculate `nev` lowest eigenvectors and eigenvalues.
Return a tuple (eigenvalues, eigenvectors).
"""
function diagonalize(xh::XSpaceHamiltonian{:sparse}; nev::Integer, verbose::Bool=false)
    prob = LS.LinearProblem(xh.H, similar(xh.H, size(xh.H, 1)))
    linsolve = LS.init(prob, LS.UMFPACKFactorization())
    linmap = LinSolveLinMap{eltype(xh.H), typeof(linsolve)}(linsolve, size(xh.H))
    ps, info = partialschur(linmap; nev, which=:LM);
    verbose && @show info
    ε, V = partialeigen(ps)
    if xh.ishermitian # if xh.H is Hermitian but complex, the solver returns complex eigenvalues
        ε .= real.(inv.(ε)) # so we make them real manually
    else
        reverse!(ε)  # we want final eigenvalues in ascending order (by abs)
        ε .= inv.(ε)
    end
    return ε, V
end

"A linear map holding a `LinearSolve` object, used for applying the inverse map."
struct LinSolveLinMap{T,L} <: LinearMaps.LinearMap{T}
    linsolve::L
    size::Dims{2}
end

Base.size(lm::LinSolveLinMap) = lm.size

function LinearMaps._unsafe_mul!(y, lm::LinSolveLinMap, x::AbstractVector)
    copy!(lm.linsolve.b, x)
    copy!(y, LS.solve!(lm.linsolve).u) # `solve!` allocates up to 50 KiB :(
end