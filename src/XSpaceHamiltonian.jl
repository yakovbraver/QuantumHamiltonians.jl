"""
A type representing a spatial, 𝐷-dimensional, 𝑛-component, possibly quasimomentum-dependent Hamiltonian (𝐻ᵃᵇ)
    𝐻ᵃᵃ(𝐱) = [-i𝛿∇ - 𝐀ᵃ(𝐱) + 𝐪]² + 𝑈ᵃᵃ(𝐱) - iΓᵃ/2
    𝐻ᵃᵇ(𝐱) = 𝑈ᵃᵇ(𝐱)
as a matrix-free 𝑥-space operator. Here  1 ≤ 𝑎, 𝑏 ≤ 𝑛,  𝐱 = (𝑥¹, …, 𝑥ᴰ),  𝐀ᵃ = (𝐴ᵃ¹, …, 𝐴ᵃᴰ),  𝐪 = (𝑞¹, …, 𝑞ᴰ). 
`R` is the underlying real scalar type -- typically `Float64` or `Float32`;
`T` is the eltype of the Hamiltonian map -- real if 𝑈 are real and 𝐴 are not present and Γ are not present; complex otherwise;
`S` is the eltype of the eigenvectors -- real if the Hamiltonian is real; complex otherwise.
Spatial dimensions are treated in the linearised form. A function 𝑓(𝑥, 𝑦) is stored in a linear array correspoding to the natural (column-major) linearisation of `f[x, y]`.
"""
mutable struct XSpaceHamiltonian{R, T, S, FourierTransformer} <: LM.LinearMap{T}
    xlims::Vector{Tuple{R, R}}
    L::Vector{R}
    B::Int # "block size" -- number of points in the contiguous array corresponding to each component
    δ::R    # coefficient of the momentum term: -i𝛿∇ (same for all components)
    nc::Int # number of components
    basis::Symbol
    ishermitian::Bool # the map is nonhermitian if decays Γ are present
    𝑈::Matrix{<:Union{Function, Nothing}} # nc-component matrix containing coordinate-space potentials and couplings. Return type must be `R` or `Complex{R}`
    U::Matrix{Vector{T}} # diagonal elements will contain 𝑈ᵃᵃ + (𝐀ᵃ)² + i𝛿∇𝐀ᵃ - iΓᵃ/2; hence `U` is complex if 𝐴 or Γ is present
    𝐴::Matrix{<:Union{Function, Nothing}} # `𝐴[c, i]` is `i`th projection of the `c`th component of the vector potential
    A::Matrix{Vector{R}} # `A[c, i]` is `i`th projection of the `c`th component of the vector potential
    ∇::Vector{Vector{R}} # `∇[i]` is the p-space flattened vector of 𝑝ᵢ = -i𝛿𝜕ᵢ if `basis=:cis` and 𝛿𝜕ᵢ otherwise. Real in both cases.
    ∇²::Vector{R} # p-space flattened vector of 𝑝² = -𝛿²Δ
    Γ::Vector{R}  # decay rates
    ε::Vector{S}  # eigenvalues, can be complex for nonhermitian Hamiltonian, hence additional type `S`
    V::Matrix{T}  # eigenvectors matrix
    ft::FourierTransformer
    buff_real::Vector{R}
    buff_complex::Vector{Complex{R}}
    buff_complex2::Vector{Complex{R}}
    buff_complex3::Vector{Complex{R}}
    size::Dims{2} # size that the map would have were it a concrete matrix. For us it's `nc*B`
end

Base.size(xh::XSpaceHamiltonian) = xh.size

"""
Convenience 1-component (but many-D) constructor accepting a 𝑈 as a function, 𝐴 as a vector (with elements treated as corresponding to the different dimensions), and Γ as a scalar.
"""
function XSpaceHamiltonian(xlims::AbstractVector{Tuple{R, R}},
                           𝑈::Union{Function, Nothing},
                           𝐴::AbstractVector{<:Union{Function, Nothing}}=fill(nothing, length(xlims));
                           basis::Symbol, M::Integer, δ::R=one(R),
                           Γ::R=zero(R)) where R <: AbstractFloat
    return XSpaceHamiltonian(xlims, [𝑈;;], reshape(𝐴, (1, length(xlims))); basis, M, δ, Γ=[Γ])
end

"""
Construct a `XSpaceHamiltonian` object using the coordinate-space functions stored in `𝑈`, decay rates `Γ`, and gauge fields stored in `𝐴`.
`𝐴[c, i]` is the `i`th projection `𝐴ⁱ` of `c`th component.
`M` is the maximum harmonic number (will use -M+1:M for cis, 1:M for sin, 0:M for cos).
"""
function XSpaceHamiltonian(xlims::AbstractVector{Tuple{R, R}},
                           𝑈::AbstractMatrix{<:Union{Function, Nothing}},
                           𝐴::AbstractVecOrMat{<:Union{Function, Nothing}}=fill(nothing, size(𝑈, 1), length(xlims));
                           basis::Symbol, M::Integer, δ::R=one(R),
                           Γ::Vector{R}=zeros(R, size(𝑈, 1))) where R <: AbstractFloat
    # Warn if M is not optimal
    if basis == :cis || basis == :cos
        !ispow2(M) && println("Maximum harmonic number M = 2ⁿ recommended for $basis basis. Consider changing M.")
    else # basis == :sin
        !ispow2(M+1) && println("Maximum harmonic number M = 2ⁿ - 1 recommended for $basis basis. Consider changing M.")
    end

    nc = size(𝑈, 1) # number of components
    D = length(xlims) # number of spatial dimensions
    L = [lims[2] - lims[1] for lims in xlims]

    # `H_isreal` will show if the resulting Hamiltonian map will be real
    𝐴_present = !all(isnothing.(𝐴))
    # 𝐴 with sin/cos not implemented because that features first-order derivative.
    𝐴_present && basis != :cis && error("XSpaceHamiltonians do not support 𝐴 for $basis basis. Only cis basis is supported.")
    U_isreal = all( 𝑢([xlims[i][1] for i in eachindex(xlims)]...) isa Real for 𝑢 in 𝑈 if !isnothing(𝑢) ) # check if all functions in 𝑈 are real
    H_isreal = U_isreal && !𝐴_present && iszero(Γ) # without checking we assume that all 𝐴 are real. Can be generalised for the exotic cases of complex 𝐴.

    T = H_isreal ? R : Complex{R} # type of elements the Hamiltonian as a linear map

    ft = FourierTransformerX(xlims, M; basis)
    xs = ft.xs
    B = length(ft.plan_forward) # "block size" -- number of points in the contiguous array corresponding to each component

    𝑈_diag_allequal = allequal(diagview(𝑈))
    𝐴ᵢ_allequal = [allequal(𝐴ᵢ) && !isnothing(𝐴ᵢ[1]) for 𝐴ᵢ in eachcol(𝐴)] # 𝐴ᵢ_allequal[i] shows if projection 𝐴ᵢ is the same for all components; note that this also checks if they are nothing

    ∇ = [make_pⁱ(L, M, δ, basis, i) for i in 1:D]
    ∇² = make_p²(L, M, δ, basis)
    # U[i, j] contains a D-dimensional array with diagonal entries representing 𝑈ᵢⱼ and off-diagonal entries representing 𝑈ᵢᵢ + 𝐴ᵢ² + i𝛿∇𝐴ᵢ - iΓᵢ/2
    U = [T[] for _ in 1:nc, _ in 1:nc] # by default, make a rank-D 0×0×… tensor
    # A[c, i] contains a D-dimensional array representing `i`th projection of the `c`th component of 𝐴
    A = [R[] for _ in 1:nc, _ in 1:D]
    
    δ∇ⁱAᵇⁱ = similar(∇²) # temporary buffer

    # allocate buffers needed for application of the map, but also used in the calculation of i𝛿𝐴 below
    buff_real = similar(∇²) # taking ∇² as a real array of the appropriate size
    buff_complex = similar(buff_real, Complex{R})
    buff_complex2 = similar(buff_real, Complex{R})
    buff_complex3 = similar(buff_real, Complex{R})

    for c in 1:nc
        for b in 1:c # only upper triangle is scanned. The lower triangle is filled automatically
            if isnothing(𝑈[b, c])
                if b == c && (any(𝐴[b, :] .!== nothing) || Γ[b] != 0) # if at least one projection for this component is nonzero or if Γ is nonzero
                    U[b, c] = zeros(T, B) # then allocate zeros for storing 𝐴² or Γ
                end
            else
                if D == 1
                    @views U[b, c] = 𝑈[b, c].(xs[:, 1])
                elseif D == 2
                    @views U[b, c] = vec(𝑈[b, c].(xs[:, 1], xs[:, 2]'))
                end
            end
            # for a diagonal block, also add 𝐴² + i𝛿∇𝐴 - iΓ/2
            if b == c
                for i in 1:D # for each projection of 𝐴: 𝐴[c, i] is `i`th projection of the `c`th component
                    if !isnothing(𝐴[b, i]) 
                        if D == 1
                            @views A[b, i] = 𝐴[b, i].(xs[:, 1])
                        elseif D == 2
                            @views A[b, i] = vec(𝐴[b, i].(xs[:, 1], xs[:, 2]'))
                        end
                        
                        # compute 𝛿∇𝐴
                        if basis == :cis
                            copy!(buff_complex2, A[b, i]) # `ft` can only act on complex vectors, so need to copy real `A[b, i]` into a complex buffer
                            transform!(buff_complex, ft, buff_complex2; direction=:forward)
                            @. buff_complex *= im*∇[i] # `∇[i]` holds -i𝛿𝜕ᵢ, but we want to apply just 𝛿𝜕ᵢ (so that we are calculating 𝛿𝜕ᵢ𝐴ᵢ, which is real), hence additional factor of i
                            transform!(buff_complex2, ft, buff_complex; direction=:backward)
                            @. δ∇ⁱAᵇⁱ = real(buff_complex2)
                        else
                            transform!(buff_real, ft, A[b, i]; direction=:forward)
                            @. buff_real *= ∇[i] # `∇[i]` holds 𝛿𝜕ᵢ
                            transform!(δ∇ⁱAᵇⁱ, ft, buff_real; direction=:backward, normalise=true)
                        end
                        # add 𝐴² + i𝛿∇𝐴
                        @. U[b, b] += A[b, i]^2 + im * δ∇ⁱAᵇⁱ
                    end
                end
                if Γ[b] != 0
                    @. U[b, b] -= im*Γ[b]/2
                end
            else # off-diagonal block
                U[c, b] = conj(U[b, c]) # if `U` is real, then `conj` returns a reference, so `U[c, b]` references `U[b, c]`
            end
        end
    end

    ishermitian = iszero(Γ) # if all `Γ`s are zeros, then Hamiltonian is Hermitian and the eigenvalues real
    S = ishermitian ? R : Complex{R} # type of eigenvalues

    # create empty placeholders
    ε = S[] # eigenvalues, can be complex for nonhermitian Hamiltonian, hence additional type `S`
    V = T[;;] # eigenvectors matrix

    return XSpaceHamiltonian(xlims, L, B, δ, nc, basis, ishermitian, 𝑈, U, 𝐴, A, ∇, ∇², Γ, ε, V, ft, buff_real, buff_complex, buff_complex2, buff_complex3, (nc*B, nc*B))
end

"Update `f′` by acting with `xh` on `f`. `f` must be a flattened x-space vector."
function LM._unsafe_mul!(f′, xh::XSpaceHamiltonian{R, T, D}, f::AbstractVector{S}) where {R, T, D, S}
    (;nc, B, basis, U, A, ∇, ∇², ft) = xh
    f_isreal = S <: Real
    f′_isreal = eltype(f′) <: Real

    buff = (basis == :cis || !f_isreal) ? xh.buff_complex : xh.buff_real
    buff2 = xh.buff_complex2
    buff3 = xh.buff_complex3

    # apply Hamiltonian to each component
    for c in 1:nc
        fᶜ = @view f[(c-1)B+1:c*B]
        f′ᶜ = @view f′[(c-1)B+1:c*B]
        ### apply Laplacian -𝛿²Δ ###
        if f_isreal && basis == :cis
            copy!(buff2, fᶜ) # `ft` can only act on complex vectors, so need to copy real `fᶜ` into a complex buffer
            transform!(buff, ft, buff2; direction=:forward)
        else
            transform!(buff, ft, fᶜ; direction=:forward)
        end
        
        # if at least one projection of 𝐴 is present (for `c`th component), then save p-space function into `buff3`. We use it below for calculating 2i𝛿∑ᵢ𝐴ᵢ∇ᵢ𝑓
        𝐴_present = row_has_something(A, c)
        𝐴_present && copy!(buff3, buff)

        @. buff *= ∇²

        if f′_isreal && basis == :cis
            # cannot write directly to `f′ᶜ` because it is real; write into a complex buffer `buff2` instead
            transform!(buff2, ft, buff; direction=:backward)
            @. f′ᶜ = real.(buff2)
        else
            transform!(f′ᶜ, ft, buff; direction=:backward, normalise=true)
        end
        ###

        # appply `U`: diagonal elements U[c, c] also include 𝐴², i𝛿∇𝐴 and decay
        for b in 1:nc
            fᵇ = @view f[(b-1)B+1:b*B]
            if !iszero(U[c, b])
                @. f′ᶜ += U[c, b] * fᵇ
            end
        end
        
        # add 2i𝛿∑ᵢ𝐴ᵢ∇ᵢ𝑓 (sum over projections)
        for i in axes(A, 2)
            if !iszero(A[c, i])
                # `∇[i]` contains 𝑝ᵢ = -i𝛿𝜕ᵢ if `basis=:cis` and 𝛿𝜕ᵢ otherwise
                @. buff2 = ∇[i] * buff3 # `buff3` contains p-space `fᶜ`
                transform!(buff, ft, buff2; direction=:backward, normalise=true)
                if basis == :cis
                    @. f′ᶜ -= 2A[c, i] * buff # if 𝑓 is real, 𝐴ᵢ∇ᵢ𝑓 is also, so we could play around with dropping real/imaginary part after FFT. But if 𝐴 is present, then `f′` is complex anyway, so we don't bother
                else
                    @. f′ᶜ += 2A[c, i] * buff * im
                end
            end
        end
    end
    return
end

"""
Check if `r`th row of a matrix `A` contains at least one element that is not `nothing`.
Same as `any(𝐴[r, :] .!== nothing)` but doesn't allocate.
"""
function row_has_something(A::AbstractMatrix, r::Integer)
    @inbounds for i in axes(A, 2)
        !isnothing(A[r, i]) && return true
    end
    false
end

"""
Calculate `nev` eigenvectors and eigenvalues.
By default, if number of components is bigger than one, then smallest-magnitude eigenvalues are calculated using inversion.
For a single component, the smallest (most negative) eigenvalues are calculated, without using inversion.
Inversion can be set/unset manually using `invert`.
The result is written into `xh.ε` and `xh.V`.
Any additional kwargs (such as `tol`, `mindim`, `maxdim`, `restarts`) will be passed to `partialschur`.
"""
function diagonalize!(xh::XSpaceHamiltonian; nev::Integer, invert::Bool=(xh.nc > 1), verbose::Bool=false, kwargs...)
    xh.ε, xh.V = diagonalize(xh; nev, verbose, invert, kwargs...)
    return
end

"""
Calculate `nev` eigenvectors and eigenvalues.
By default, if number of components is bigger than one, then smallest-magnitude eigenvalues are calculated using inversion.
For a single component, the smallest (most negative) eigenvalues are calculated, without using inversion.
Inversion can be set/unset manually using `invert`.
Any additional kwargs (such as `tol`, `mindim`, `maxdim`, `restarts`) will be passed to `partialschur`. If `tol` is passed, it will be also passed to GMRES as `reltol` in case of inversion.
Return a tuple (eigenvalues, eigenvectors).
"""
function diagonalize(xh::XSpaceHamiltonian{R, T}; invert::Bool=(xh.nc > 1), nev::Integer=0, verbose::Bool=false, kwargs...) where {R, T}
    if invert
        # Here we do shift-invert: we want to diagonalise 𝐻⁻¹, defined by its action 𝑥 = 𝐻⁻¹𝑏; 𝑥 is found by solving 𝐻𝑥 = 𝑏. But `LS.LinearProblem` does not work with LinearMaps, so we wrap `bdg_map` in a SciMLOperator
        xh_op = SciMLOperators.FunctionOperator(XSpaceHamiltonian!, Vector{T}(undef, xh.nc*xh.B); p=xh, isconstant=true)
        prob = LS.LinearProblem(xh_op, Vector{T}(undef, xh.nc*xh.B))
        reltol = haskey(kwargs, :tol) ? kwargs[:tol] : √eps(R) # use user's "tol" if passed; otherwise use LinearSolve's default
        linsolve = LS.init(prob, LS.KrylovJL_GMRES(); reltol)
        linmap = LinSolveLinMap{T, typeof(linsolve)}(linsolve, size(xh))
        ps, info = partialschur(linmap; nev, which=:LM, kwargs...)
    else
        ps, info = partialschur(xh; nev, which=:SR, kwargs...)
    end
    
    verbose && @show info
        
    if xh.ishermitian
        ε, V = real(ps.eigenvalues), ps.Q
        if invert
            ε .= inv.(ε)
        end
    else
        ε, V = partialeigen(ps)
        if invert
            ε .= inv.(ε)
            # here we must reverse to get final eigenvalues in ascending order (by abs)
            reverse!(ε)
            reverse!(V; dims=2)
        end
    end

    # The eigenvectors come out normalised as ∑ᶜ|𝜓ᶜ|² = 1 (sum over components), but we want ∑ᶜ|𝜓ᶜ|²d𝑉 = 1.
    # So normalise by dividing by √d𝑉.
    xs = xh.ft.xs
    dV = prod(xs[2, i] - xs[1, i] for i in axes(xs, 2)) # volume element
    V ./= √dV

    return ε, V
end

"Helper function for wrapping `XSpaceHamiltonian` in a SciMLOperator. `params` will be a `XSpaceHamiltonian` object."
function XSpaceHamiltonian!(w, v, u, params, t)
    mul!(w, params, v)
end

"""
Construct D-dimensional x-space wave function of eigenstate `stateno` having the format `ψ[component][x, y, …]`.
Return a tuple (`xs`, `ys`, …, `ψ`).
"""
function make_eigenfunction(xh::XSpaceHamiltonian{R, T}; stateno::Int) where {R, T}
    make_wavefunction(xh, xh.V[:, stateno])
end

"""
Construct D-dimensional x-space wave function of eigenstate `stateno` by reshaping xh.V[:, stateno] into `ψ` having the format `ψ[component][x, y, …]`.
Return a tuple (`xs`, `ys`, …, `ψ`).
"""
function make_wavefunction(xh::XSpaceHamiltonian{R, T}, v::AbstractVector{T}) where {R, T}
    (;B, nc) = xh
    ψ = map(1:nc) do c
        @views reshape(v[(c-1)B+1:c*B], size(xh.ft.plan_forward))
    end
    return ntuple(i -> xh.ft.xs[:, i], size(xh.ft.xs, 2))..., ψ # first part splits xs into a tuple (𝑥, 𝑦, …)
end

# TODO these two methods work both with PSpaceHamiltonian and XSpaceHamiltonian, move to furute QuantumHamiltonian.jl

"""
Calculate inner product ∫𝑣𝑤d𝑥 (but without d𝑥) between `v` and `w` representing single-component x-space vectors.
For cis and sin basis, this is just a dot, but for cos we have to treat the edges.
The information of the basis and problem dimensions is contained in `qh`.
"""
@views function inner_prod(v::AbstractVector{<:Number}, w::AbstractVector{<:Number}, qh)
    s = if qh.basis != :cos
        dot(v, w)
    else
        D = length(qh.xlims)
        if D == 1
            dot(v, w) - (v[1]w[1] + v[end]w[end]) / 2
        elseif D == 2
            # create reshaped views for convenient indexing
            v_tensor = reshape(v, size(qh.ft.plan_forward))
            w_tensor = reshape(w, size(qh.ft.plan_forward))
            dot.(v, w) -
            ( dot(v_tensor[1, 2:end-1]  , w_tensor[1, 2:end-1])   +
              dot(v_tensor[end, 2:end-1], w_tensor[end, 2:end-1]) +
              dot(v_tensor[2:end-1, 1]  , w_tensor[2:end-1, 1])   +
              dot(v_tensor[2:end-1, end], w_tensor[2:end-1, end])) / 2 -
            (v_tensor[1, 1]'*w_tensor[1, 1] + v_tensor[1, end]'*w_tensor[1, end] + v_tensor[end, 1]'*w_tensor[end, 1] + v_tensor[end, end]'*w_tensor[end, end]) * 3/4
        else
            error("inner_prod not implemented for cos basis in $D dimensions.")
        end
    end
    dV = prod(qh.ft.xs[2, i] - qh.ft.xs[1, i] for i in axes(qh.ft.xs, 2)) # volume element
    return s*dV
end

"""
Calculate the squared norm ‖𝑣‖² = ∫|𝑣|²d𝑉 for `v` representing single-component x-space vectors.
For cis and sin basis, this is `sum(abs2, v)`, but for cos we have to treat the edges.
The information of the basis and problem dimensions is contained in `xh`.
"""
@views function norm²(v::AbstractVector{<:Number}, qh)
    s = if qh.basis != :cos
        sum(abs2, v)
    else
        D = length(qh.xlims)
        if D == 1
            sum(abs2, v) - (abs2(v[1]) + abs2(v[end])) / 2
        elseif D == 2
            # create reshaped views for convenient indexing
            v_tensor = reshape(v, size(qh.ft.plan_forward))
            sum(abs2, v) -
            (sum(abs2, v_tensor[1, 2:end-1]  )  +
             sum(abs2, v_tensor[end, 2:end-1])  +
             sum(abs2, v_tensor[2:end-1, 1]  )  +
             sum(abs2, v_tensor[2:end-1, end])) / 2 -
            (abs2(v_tensor[1, 1]) + abs2(v_tensor[1, end]) + abs2(v_tensor[end, 1]) + abs2(v_tensor[end, end])) * 3/4
        else
            error("norm² not implemented for cos basis in $D dimensions.")
        end
    end
    dV = prod(qh.ft.xs[2, i] - qh.ft.xs[1, i] for i in axes(qh.ft.xs, 2)) # volume element
    return s*dV
end

"""
Calculate the integral ∫𝑣d𝑉 for `v` representing single-component x-space vectors.
For cis and sin basis, this is `sum(v)`, but for cos we have to treat the edges.
The information of the basis and problem dimensions is contained in `xh`.
"""
@views function integrate(v::AbstractVector{<:Number}, qh)
    s = if qh.basis != :cos
        sum(v)
    else
        D = length(qh.xlims)
        if D == 1
            sum(v) - (v[1] + v[end]) / 2
        elseif D == 2
            # create reshaped views for convenient indexing
            v_tensor = reshape(v, size(qh.ft.plan_forward))
            sum(v) -
            (sum(v_tensor[1, 2:end-1]  )  +
             sum(v_tensor[end, 2:end-1])  +
             sum(v_tensor[2:end-1, 1]  )  +
             sum(v_tensor[2:end-1, end])) / 2 -
            (v_tensor[1, 1] + v_tensor[1, end] + v_tensor[end, 1] + v_tensor[end, end]) * 3/4
        else
            error("inner_prod_premultiplied not implemented for cos basis in $D dimensions.")
        end
    end
    dV = prod(qh.ft.xs[2, i] - qh.ft.xs[1, i] for i in axes(qh.ft.xs, 2)) # volume element
    return s*dV
end

"""
For an x-space state `ψ` in the form of a flattened vector, return `E, μ, η`, where `E` is mean energy per particle, `μ` is a vector of chemical potentials of each component,
and `η` is a vector of relative particle numbers of each component.
By default, `makereal=true` so that the returned `E` and `μ` are made real (by dropping imaginary part). Set `makereal=false` if you consider a decaying state, whereby imaginary part is important.
"""
function get_Eμη(xh::XSpaceHamiltonian, ψ::AbstractVector{<:Number}, g::AbstractMatrix{<:Number}=zeros(typeof(xh.δ), xh.nc, xh.nc); makereal=true)
    (;nc, B) = xh
     
    η = [@views norm²(ψ[(c-1)B+1:c*B], xh) for c in 1:nc]
    η_total = sum(η)

    # calculate mean energy of every component 𝑒ᵢ = ⟨𝜓ᵢ|𝐻|𝜓ᵢ⟩
    Hψ = similar(ψ, nc*B) # buffer for storing the result of 𝐻|𝜓ᵢ⟩; specify length manually because `ψ` might contain chemical potentials in the last `nc` elements
    mul!(Hψ, xh, ψ)
    e = [@views inner_prod(Hψ[(c-1)B+1:c*B], ψ[(c-1)B+1:c*B], xh) for c in 1:nc]
    E = sum(e) / η_total
    μ = e ./ η
    
    if !iszero(g)
        # pre-calculate squared wf; a vector of vectors is a bit more convenient
        ψ² = map(1:nc) do c
            @views abs2.(ψ[(c-1)B+1:c*B])
        end

        # for each `i`th component: calculate the sum ∑ⱼ 𝑔ᵢⱼ|𝜓ⱼ|², then multiply by |𝜓ᵢ|², then integrate
        ψ²_sum = similar(ψ²[1])
        for i in 1:nc
            ψ²_sum .= 0
            for j in 1:nc
                g[i, j] == 0 && continue
                @. ψ²_sum += g[i, j] * ψ²[j]
            end
            ψ²_sum .*= ψ²[i]
            U = integrate(ψ²_sum, xh)
            μ[i] += U / η[i]
            E += U / 2η_total
        end
    end

    if makereal
        return real(E), real(μ), η
    else
        return E, μ, η
    end
end


################ StateVector approach ################

# The methods below enable diagonalisation whereby an instance of `XSpaceHamiltonian` acts on an instance of `StateVector`,
# which is a wrapper of an nc-component vector holding D-dimensional tensors. In practice, this is not more convenient than the flattened approach.

"""
Act with `xh` on a `StateVector`, producing a new `StateVector`.
Note that components of `f_in` are D-dimensional tensors, while `xh` uses the flattened approach. Therefore, we use `vec`-reshaping on `f_in`.
"""
function (xh::XSpaceHamiltonian{R, T})(f_in::StateVector{S, D}) where {R, T, D, S}
    (;nc, basis, U, A, ∇, ∇², ft) = xh
    
    f_in_isreal = S <: Real
    f_out_isreal = f_in_isreal && T <: Real
    f_out = similar(f_in, (f_out_isreal ? R : Complex{R}))

    buff = (basis == :cis || !f_in_isreal) ? xh.buff_complex : xh.buff_real
    buff2 = xh.buff_complex2
    buff3 = xh.buff_complex3

    # apply Hamiltonian to each component
    for c in 1:nc
        # --- apply Laplacian -𝛿²Δ
        if f_in_isreal && basis == :cis
            copyto!(buff2, f_in[c]) # `ft` can only act on complex vectors, so need to copy real `f_in[c]` into a complex buffer
            transform!(buff, ft, buff2; direction=:forward)
        else
            transform!(buff, ft, f_in[c]; direction=:forward)
        end

        # if at least one projection of 𝐴 is present (for `c`th component), then save p-space function into `buff3`. We use it below for calculating 2i𝛿∑ᵢ𝐴ᵢ∇ᵢ𝑓
        𝐴_present = row_has_something(A, c)
        𝐴_present && copy!(buff3, buff)

        @. buff *= ∇²

        if f_out_isreal && basis == :cis
            # cannot write directly to `f_out[c]` because it is real; write into a complex buffer `buff2` instead
            transform!(buff2, ft, buff; direction=:backward)
            vec(f_out[c]) .= real.(buff2)
        else
            transform!(f_out[c], ft, buff; direction=:backward, normalise=true)
        end
        # --- 

        # appply `U`: diagonal elements U[c, c] also include 𝐴², i𝛿∇𝐴 and decay
        for b in 1:nc
            if !iszero(U[c, b])
                vec(f_out[c]) .+= U[c, b] .* vec(f_in[b])
            end
        end
        
        # add 2i𝛿∑ᵢ𝐴ᵢ∇ᵢ𝑓 (sum over projections)
        for i in 1:D
            if !iszero(A[c, i])
                # `∇[i]` contains 𝑝ᵢ = -i𝛿𝜕ᵢ if `basis=:cis` and 𝛿𝜕ᵢ otherwise
                @. buff2 = ∇[i] * buff3 # `buff3` contains p-space `f_in[c]`
                transform!(buff, ft, buff2; direction=:backward, normalise=true)
                if basis == :cis
                    vec(f_out[c]) .-= 2 .* A[c, i] .* buff # if 𝑓 is real, 𝐴ᵢ∇ᵢ𝑓 is also, so we could play around with dropping real/imaginary part after FFT. But if 𝐴 is present, then `f_out` is complex anyway, so we don't bother
                else
                    vec(f_out[c]) .+= 2 .* A[c, i] .* buff .* im
                end
            end
        end
    end
    
    return f_out
end

"""
Calculate `nev` eigenvectors and eigenvalues via action on a custom `StateVector` type.
By default, if number of components is bigger than one, then smallest-magnitude eigenvalues are calculated using inversion.
For a single component, the smallest (most negative) eigenvalues are calculated, without using inversion.
Inversion can be set/unset manually using `invert`. Arguments `tol`, `maxiter`, and `krylovdim` will be passed to the eigensolver (and linear solver in case of inversion).
The KrylovKit package's defaults are used except that we set `tol=1e-5` for `Float32`.
Additionally, `ishermitian` will override the default value of `xh.ishermitian`. We use this if 𝐴 is present because solver claims that the map is nonhermitian and yields wrong answer.
Return full KrylovKit output (vals, vecs, info).
"""
function diagonalize_via_statevector(xh::XSpaceHamiltonian{R, T}; nev::Integer, linsolve_verbose=true, invert::Bool=(xh.nc > 1),
                                     tol = (R === Float32 ? 1f-5 : KrylovKit.KrylovDefaults.tol[]), maxiter=KrylovKit.KrylovDefaults.maxiter[],
                                     krylovdim=KrylovKit.KrylovDefaults.krylovdim[], ishermitian=xh.ishermitian) where {R, T}
    verbosity = linsolve_verbose ? KrylovKit.WARN_LEVEL : KrylovKit.SILENT_LEVEL
    # initial guess, mainly needed to convey the storage type of the eigenvector. Type must be `T`: if Hamiltonian is Hermitian but complex, we need complex eigenvectors
    v0 = StateVector{xh.basis}([rand(T, size(xh.ft.plan_forward)) for _ in 1:xh.nc])
    if invert
        K = KrylovKit.eigsolve(v0, nev, :LM; ishermitian, tol, maxiter, krylovdim) do b # `b` is a vector on which the Hamiltonian is acting
            x, _ = KrylovKit.linsolve(xh, b; ishermitian, tol, maxiter, krylovdim, verbosity) # find 𝑥 = 𝐻⁻¹𝑏 by solving 𝐻𝑥 = 𝑏
            return x
        end
        @. K[1] = inv(K[1]) # invert the eigenvalues back
    else
        K = KrylovKit.eigsolve(xh, v0, nev, :SR; ishermitian, tol, maxiter, krylovdim)
    end
    # The eigenvectors come out normalised as ∑ᶜ|𝜓ᶜ|² = 1 (sum over components), but we want ∑ᶜ|𝜓ᶜ|²d𝑉 = 1.
    # So normalise by dividing by √d𝑉.
    xs = xh.ft.xs
    dV = prod(xs[2, i] - xs[1, i] for i in axes(xs, 2)) # volume element
    for v in K[2]
        KrylovKit.VectorInterface.scale!(v, 1/√dV)
    end
    return K
end