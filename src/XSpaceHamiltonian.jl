abstract type XSpaceHamiltonian{S} end

matrix_density(xh::XSpaceHamiltonian) = @error "Matrix density calculation is available for sparse Hamiltonians only."

"General dense constructor. If the problem is 1D, 𝐴 may be passed as a vector, whose elements are treated as corresponding to the different components."
function XSpaceHamiltonian{:dense}(xlims::AbstractVector{Tuple{R,R}},
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
function XSpaceHamiltonian{:dense}(xlims::AbstractVector{Tuple{R,R}},
                                   𝑈::Union{Function,Nothing},
                                   𝐴::AbstractVector{<:Union{Function,Nothing}}=fill(nothing, length(xlims));
                                   basis::Symbol, M::Integer, δ::R=one(R),
                                   𝑈_iseven::Bool=false, Γ::R=zero(R)) where R <: AbstractFloat
    return DenseHamiltonian(xlims, [𝑈;;], reshape(𝐴, (1, length(xlims))); basis, M, δ, 𝑈_iseven=[𝑈_iseven;;], Γ=[Γ])
end

"General sparse constructor. If the problem is 1D, 𝐴 may be passed as a vector, whose elements are treated as corresponding to the different components."
function XSpaceHamiltonian{:sparse}(xlims::AbstractVector{Tuple{R,R}},
                                    𝑈::AbstractMatrix{<:Union{Function,Nothing}},
                                    𝐴::AbstractVecOrMat{<:Union{Function,Nothing}}=fill(nothing, size(𝑈, 1), length(xlims));
                                    basis::Symbol, M::Integer, δ::R=one(R), fft_threshold::R=√eps(R),
                                    𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1))) where R <: AbstractFloat
    if R !== Float64
        @info "Sparse diagonalisation is only supported for Float64, got R = $R. Constructed object Will use Float64."
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
function XSpaceHamiltonian{:sparse}(xlims::AbstractVector{Tuple{R,R}},
                                   𝑈::Union{Function,Nothing},
                                   𝐴::AbstractVector{<:Union{Function,Nothing}}=fill(nothing, length(xlims));
                                   basis::Symbol, M::Integer, δ::R=one(R), fft_threshold::R=√eps(R),
                                   𝑈_iseven::Bool=false, Γ::R=zero(R)) where R <: AbstractFloat
    if R !== Float64
        @info "Sparse diagonalisation is only supported for R = Float64, got R = $R. Constructed object Will use Float64."
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
function make_eigenfunctions(xh::XSpaceHamiltonian; statenos::AbstractVector{<:Integer}, nx::Integer, iqxs::AbstractVector{<:Integer}=Int[])
    (;L, xlims, M, basis, V, V_q, nc) = xh
    Lx = L[1]
    xs = range(0, Lx, nx) # these are the differences `x - xlims[1]`, with `x ∈ xlims`
    ns = isempty(iqxs) ? length(statenos) : length(iqxs)
    R = eltype(L) # real working type
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
                else # nonperiodic
                    @floop for (ix, x) in enumerate(xs)
                        ψ[ix, c, is] = sum(V[(c-1)*M+jx, stateno]sin(π*jx*x/Lx) for jx in 1:M) * √(2/Lx)
                    end
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
Construct 2D coordinate-space wave function `ψ` of eigenstate `stateno` on a grid having `nx` points in `x` and `ny` points in `y` direction.
Return (`xs`, `ys`, `ψ`). If `qx` and `qy` are passed, then construct `ψ` at the corresponding quasimomenta.
"""
function make_eigenfunction(xh::XSpaceHamiltonian, stateno::Integer, nx::Integer, ny::Integer, iqx::Integer=0, iqy::Integer=0)
    (;L, xlims, M, basis, V, V_q, nc) = xh
    Lx, Ly = L
    xs = range(0, Lx, nx) # these are the differences `x - xlims[1][1]`, with `x ∈ xlims[1]`
    ys = range(0, Ly, ny) # these are the differences `y - xlims[2][1]`, with `y ∈ xlims[2]`
    R = eltype(L) # real working type
    ψ_type = basis != :cis && eltype(xh.H) isa Real ? R : complex(R)
    ψ = [Matrix{ψ_type}(undef, nx, ny) for _ in 1:nc] # `ψ` are real if elements of H are real and the basis is real (sin/cos)
    for c in 1:nc
        if basis == :cis
            B = 2M + 1
            if iqx != 0 # if quasimomentum index has been passed
                @floop for (iy, y) in enumerate(ys)
                    for (ix, x) in enumerate(xs)
                        ψ[c][ix, iy] = sum(V_q[(c-1)*B^2+(j-1)B+i, stateno, iqx, iqy]cis(2π*jx*x/Lx + 2π*jy*y/Ly) for (j, jx) in enumerate(-M:M)
                                                                                                                  for (i, jy) in enumerate(-M:M)) / √(Lx*Ly)
                    end
                end
            else # no quasimomentum index
                @floop for (iy, y) in enumerate(ys)
                    for (ix, x) in enumerate(xs)
                        ψ[c][ix, iy] = sum(V[(c-1)*B^2+(j-1)B+i, stateno]cis(2π*jx*x/Lx + 2π*jy*y/Ly) for (j, jx) in enumerate(-M:M)
                                                                                                      for (i, jy) in enumerate(-M:M)) / √(Lx*Ly)
                    end
                end
            end
        else # nonperiodic
            @floop for (iy, y) in enumerate(ys)
                for (ix, x) in enumerate(xs)
                    ψ[c][ix, iy] = sum(V[(c-1)*M^2+(jx-1)M+jy, stateno]sin(π*jx*x/Lx)sin(π*jy*y/Ly) for jx in 1:M for jy in 1:M) * 2 / √(Lx*Ly)
                end
            end
        end
    end
    return xs .+ xlims[1][1], ys .+ xlims[2][1], ψ # return "normal" coordinates, in `x ∈ xlims` and `y ∈ ylims`
end

########## Dense

"""
Calculate `nev` lowest eigenvectors and eigenvalues using `ArnoldiMethod`.
Pass `nev=0` for full diagonalisation using `LinearAlgebra`.
The result is written into `xh.ε` and `xh.V`.
"""
function diagonalize!(xh::XSpaceHamiltonian; nev::Integer, verbose::Bool=false)
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
Return a tuple (eigenvalues, eigenvectors).
"""
function diagonalize(xh::XSpaceHamiltonian{:sparse}; nev::Integer, verbose::Bool=false)
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
struct LinSolveLinMap{T,L} <: LinearMaps.LinearMap{T}
    linsolve::L
    size::Dims{2}
end

Base.size(lm::LinSolveLinMap) = lm.size

function LinearMaps._unsafe_mul!(y, lm::LinSolveLinMap, x::AbstractVector)
    copy!(lm.linsolve.b, x)
    copy!(y, LS.solve!(lm.linsolve).u) # `solve!` allocates up to 50 KiB :(
end