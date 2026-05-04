"""
A type representing a spatial, 𝐷-dimensional, 𝑛-component, possibly quasimomentum-dependent Hamiltonian (𝐻ᵃᵇ)
    𝐻ᵃᵃ(𝐱) = [-i𝛿∇ + 𝐪 - 𝐀ᵃ(𝐱)]² + 𝑈ᵃᵃ(𝐱) - iΓᵃ/2
    𝐻ᵃᵇ(𝐱) = 𝑈ᵃᵇ(𝐱)
as a matrix-free 𝑥-space operator. Here  1 ≤ 𝑎, 𝑏 ≤ 𝑛,  𝐱 = (𝑥¹, …, 𝑥ᴰ),  𝐀ᵃ = (𝐴ᵃ¹, …, 𝐴ᵃᴰ),  𝐪 = (𝑞¹, …, 𝑞ᴰ). 
`R` is the underlying real scalar type -- typically `Float64` or `Float32`;
`T` is the eltype of the Hamiltonian map -- real if 𝑈 are real and 𝐴 are not present and Γ are not present; complex otherwise;
`D` is the number of physical dimensions.
"""
struct XSpaceHamiltonian{R, T, D, FourierTransformer}
    xlims::Vector{Tuple{R, R}}
    L::Vector{R}
    M::Int # maximum harmonic number (will use -M+1:M for cis, 1:M for sin, 0:M for cos)
    δ::R # coefficient of the momentum term: -i𝛿∇ (same for all components)
    nc::Int # number of components
    basis::Symbol
    ishermitian::Bool # `H` is nonhermitian if decays Γ are present
    𝑈::Matrix{<:Union{Function, Nothing}} # nc-component matrix containing coordinate-space potentials and couplings. Return type must be `R` or `Complex{R}`
    U::Matrix{Array{T, D}} # diagonal elements will contain 𝑈ᵃᵃ + (𝐀ᵃ)² + i𝛿∇𝐀ᵃ - iΓᵃ/2; hence `U` is complex if 𝐴 or Γ is present
    𝐴::Matrix{<:Union{Function, Nothing}} # 𝐴[c, i] is `i`th projection of the `c`th component of the vector potential
    A::Matrix{Array{R, D}} # `A[c, i]` is `i`th projection of the `c`th component of the vector potential
    ∇::Vector{Array{R, D}} # `∇[i]` is the p-space rank-D tensor of 𝑝ᵢ = -iδ𝜕ᵢ if `basis=:cis` and δ𝜕ᵢ otherwise. Real in both cases.
    ∇²::Array{R, D} # p-space rank-D tensor of 𝑝² = -𝛿²Δ
    Γ::Vector{R} # decay rates
    ft::FourierTransformer
    buff_real::Array{R, D}
    buff_complex::Array{Complex{R}, D}
    buff_complex2::Array{Complex{R}, D}
    buff_complex3::Array{Complex{R}, D}
end

"Show the XSpaceHamiltonian."
function Base.show(io::IO, xh::XSpaceHamiltonian{R, T, D}) where {R, T, D}
    # Examples:
    # "2-component XSpaceHamiltonian{Float64, 1} on a 5-point grid"
    # "3-component XSpaceHamiltonian{Float64, 3} on a 5×6×7 grid"
    print(io, length(xh.xlims), "-component XSpaceHamiltonian{$R, $T, $D} on a ", size(xh.ft.xs, 1),
          (D == 1 ? "-point" : prod("×$(size(xh.ft.xs, 1))" for i in 2:D)), " grid") # only square grids are currently supported
end

"""
Convenience 1-component (but many-D) constructor accepting a 𝑈 as a function, 𝐴 as a vector (with elements treated as corresponding to the different dimensions), and Γ as a real.
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

    # `H_isreal` will show if the resulting `H` will be real
    𝐴_present = !all(isnothing.(𝐴))
    U_isreal = all( 𝑢([xlims[i][1] for i in eachindex(xlims)]...) isa Real for 𝑢 in 𝑈 if !isnothing(𝑢) ) # check if all functions in 𝑈 are real
    H_isreal = U_isreal && !𝐴_present && iszero(Γ) # without checking we assume that all 𝐴 are real. Can be generalised for the exotic cases of complex 𝐴.

    T = H_isreal ? R : Complex{R} # type of elements the Hamiltonian as a linear map

    ft = FourierTransformerX(xlims, M; basis)
    xs = ft.xs
    N = size(xs, 1) # number of grid points

    𝑈_diag_allequal = allequal(diagview(𝑈))
    𝐴ᵢ_allequal = [allequal(𝐴ᵢ) && !isnothing(𝐴ᵢ[1]) for 𝐴ᵢ in eachcol(𝐴)] # 𝐴ᵢ_allequal[i] shows if projection 𝐴ᵢ is the same for all components; note that this also checks if they are nothing

    ∇ = [make_p_i_tensor(L, M, δ, basis, i) for i in 1:D]
    ∇² = make_p²_tensor(L, M, δ, basis)
    # U[i, j] contains a D-dimensional array with diagonal entries representing 𝑈ᵢⱼ and off-diagonal entries representing 𝑈ᵢᵢ + 𝐴ᵢ² + i𝛿∇𝐴ᵢ - iΓᵢ/2
    U = [Array{T, D}(undef, ntuple(Returns(0), D)) for _ in 1:nc, _ in 1:nc] # by default, make a rank-D 0×0×… tensor
    # A[c, i] contains a D-dimensional array representing `i`th projection of the `c`th component of 𝐴
    A = [Array{R, D}(undef, ntuple(Returns(0), D)) for _ in 1:nc, _ in 1:D]
    
    δ∇ⁱAᵇⁱ = similar(∇²) # temporary buffer

    # allocate buffers needed for application of the map, but also used in the calculation of i𝛿𝐴 below
    buff_real = similar(∇²) # taking ∇² as a real array of the appropriate size
    buff_complex = similar(buff_real, Complex{R})
    buff_complex2 = similar(buff_real, Complex{R})
    buff_complex3 = similar(buff_real, Complex{R})

    for c in 1:nc
        for b in 1:c # only upper triangle is scanned. The lower triangle is filled automatically
            if isnothing(𝑈[b, c])
                if b == c && any(𝐴[b, :] .!== nothing) # if at least one projection for this component is nonzero
                    U[b, c] = zeros(T, ntuple(Returns(N), D)) # then allocate zeros for storing 𝐴²
                end
            else
                if D == 1
                    @views U[b, c] = 𝑈[b, c].(xs[:, 1])
                elseif D == 2
                    @views U[b, c] = 𝑈[b, c].(xs[:, 1], xs[:, 2]')
                end
            end
            # for a diagonal block, also add 𝐴² + i𝛿∇𝐴 - iΓ/2
            if b == c
                for i in 1:D # for each projection of 𝐴: 𝐴[c, i] is `i`th projection of the `c`th component
                    if !isnothing(𝐴[b, i]) 
                        if D == 1
                            @views A[b, i] = 𝐴[b, i].(xs[:, 1])
                        elseif D == 2
                            @views A[b, i] = 𝐴[b, i].(xs[:, 1], xs[:, 2]')
                        end
                        
                        # compute 𝛿∇𝐴
                        if basis == :cis
                            copy!(buff_complex2, A[b, i]) # `ft` can only act on complex vectors, so need to copy real `A[b, i]` into a complex buffer
                            transform!(buff_complex, ft, buff_complex2; direction=:forward)
                            @. buff_complex *= im*∇[i] # `∇[i]` holds -i𝛿𝜕ᵢ, but we want to apply just 𝛿𝜕ᵢ (so that we are calculating 𝛿𝜕ᵢ𝐴ᵢ, which is real), hence additional factor of i
                            transform!(buff_complex2, ft, buff_complex; direction=:backward)
                            copyreal!(δ∇ⁱAᵇⁱ, buff_complex2)
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

    ishermitian = iszero(Γ)

    return XSpaceHamiltonian(xlims, L, M, δ, nc, basis, ishermitian, 𝑈, U, 𝐴, A, ∇, ∇², Γ, ft, buff_real, buff_complex, buff_complex2, buff_complex3)
end

function (xh::XSpaceHamiltonian{R, T, D})(f_in::StateVector{S, D}) where {R, T, D, S}
    (;nc, basis, U, A, 𝐴, ∇, ∇², ft) = xh
    
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
            copy!(buff2, f_in[c]) # `ft` can only act on complex vectors, so need to copy real `f_in[c]` into a complex buffer
            transform!(buff, ft, buff2; direction=:forward)
        else
            transform!(buff, ft, f_in[c]; direction=:forward)
        end

        # if at least one projection of 𝐴 is present (for `c`th component), then save p-space function into `buff3`. We use it below for calculating 2i𝛿∑ᵢ𝐴ᵢ∇ᵢ𝑓
        𝐴_present = !all(isnothing.(𝐴[c, :]))
        𝐴_present && copy!(buff3, buff)

        @. buff *= ∇²

        if f_out_isreal && basis == :cis
            # cannot write directly to `f_out[c]` because it is real; write into a complex buffer `buff2` instead
            transform!(buff2, ft, buff; direction=:backward)
            copyreal!(f_out[c], buff2)
        else
            transform!(f_out[c], ft, buff; direction=:backward, normalise=true)
        end
        # --- 

        # appply `U`: diagonal elements U[c, c] also include 𝐴², i𝛿∇𝐴 and decay
        for b in 1:nc
            if !iszero(U[c, b])
                @. f_out[c] += U[c, b] * f_in[b]
            end
        end
        
        # add 2i𝛿∑ᵢ𝐴ᵢ∇ᵢ𝑓 (sum over projections)
        for i in 1:D
            if !iszero(A[c, i])
                # `∇[i]` contains 𝑝ᵢ = -i𝛿𝜕ᵢ if `basis=:cis` and 𝛿𝜕ᵢ otherwise
                @. buff2 = ∇[i] * buff3 # `buff3` contains p-space `f_in[c]`
                transform!(buff, ft, buff2; direction=:backward, normalise=true)
                if basis == :cis
                    @. f_out[c] -= 2A[c, i] * buff # if 𝑓 is real, 𝐴ᵢ∇ᵢ𝑓 is also, so we could play around with dropping real/imaginary part after FFT. But if 𝐴 is present, then `f_out` is complex anyway, so we don't bother
                else
                    @. f_out[c] += 2A[c, i] * buff * im
                end
            end
        end
    end
    
    return f_out
end

"Element-wise copy real part of a complex array `z` into a real array `r`."
function copyreal!(r::AbstractArray{<:Real}, z::AbstractArray{<:Complex})
    @inbounds @simd for i in eachindex(z)
        r[i] = real(z[i])
    end
end

"""
Calculate `nev` eigenvectors and eigenvalues.
By default, if number of components is bigger than one, then smallest-magnitude eigenvalues are calculated using inversion.
For a single component, the smallest (most negative) eigenvalues are calculated, without using inversion.
Inversion can be set/unset manually using `invert`. Arguments `tol`, `maxiter`, and `krylovdim` will be passed to the eigensolver (and linear solver in case of inversion).
The KrylovKit package's defaults are used except that we set `tol=1e-5` for `Float32`.
Additionally, `ishermitian` will override the default value of `xh.ishermitian`. We use this if 𝐴 is present because solver claims that the map is nonhermitian and yields wrong answer.
Return full KrylovKit output (vals, vecs, info).
"""
function diagonalize(xh::XSpaceHamiltonian{R, T, D}; nev::Integer, linsolve_verbose=true, invert::Bool=(xh.nc > 1),
                     tol = (R === Float32 ? 1f-5 : KrylovKit.KrylovDefaults.tol[]), maxiter=KrylovKit.KrylovDefaults.maxiter[],
                     krylovdim=KrylovKit.KrylovDefaults.krylovdim[], ishermitian=xh.ishermitian) where {R, T, D}
    verbosity = linsolve_verbose ? KrylovKit.WARN_LEVEL : KrylovKit.SILENT_LEVEL
    # initial guess, mainly needed to convey the storage type of the eigenvector. Type must be `T`: if Hamiltonian is Hermitian but complex, we need complex eigenvectors
    v0 = StateVector{xh.basis}([rand(T, size(xh.buff_real)...) for _ in 1:xh.nc])
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