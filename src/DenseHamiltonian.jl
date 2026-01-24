"""
A type representing a spatial, 𝐷-dimensional, 𝑛-component, possibly quasimomentum-dependent Hamiltonian (𝐻ᵢⱼ)
    𝐻ᵢᵢ(r) = [-i𝛿∇ + q - Aᵢ(r)]² + 𝑈ᵢᵢ(r) - iΓᵢ/2
    𝐻ᵢⱼ(r) = 𝑈ᵢⱼ(r)
as a dense matrix. Here  1 ≤ 𝑖, 𝑗 ≤ 𝑛,  r = (𝑥₁, …, 𝑥_𝐷),  Aᵢ = (𝐴ᵢ₁, …, 𝐴ᵢ_𝐷),  q = (𝑞₁, …, 𝑞_𝐷).
"""
mutable struct DenseHamiltonian{R,T,S,D1,D2} <: XSpaceHamiltonian{:dense,R,T,S,D1,D2}
    xlims::Vector{Tuple{R, R}}
    L::Vector{R}
    M::Int # maximum harmonic number (will use -M:M for periodic, 1:M for nonperiodic)
    δ::R # coefficient of the momentum term: -iδ∇ (same for all components)
    nc::Int # number of components
    basis::Symbol
    ishermitian::Bool # `H` is nonhermitian if decays Γ are present
    𝑈::Matrix{<:Union{Function,Nothing}} # nc-component matrix containing coordinate-space potentials and couplings. Return type must be R or T
    𝑈_iseven::BitMatrix # nc-component matrix indicating if 𝑈ᵢⱼ is an even function 𝑈ᵢⱼ(r) = 𝑈ᵢⱼ(-r)
    𝐴::Matrix{<:Union{Function,Nothing}} # 𝐴[c, i] is `i`th projection of the `c`th component of hte vector potential
    Γ::Vector{R} # decay rates
    H::Matrix{T} # momentum-space Hamiltonian used for diagonalisation
    ε::Vector{S} # eigenvalues, can be complex for nonhermitian `H`, hence additional type `S`
    V::Matrix{T} # eigenvectors matrix
    ε_q::Array{S,D1} # ε_q[n, iqx, iqy] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    V_q::Array{T,D2} # V_q[:, n, iqx, iqy] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)
    wanniers::Wanniers{R} # wanniers are implemented only for the case of 1-component and 1D
end

"""
Construct a `DenseHamiltonian` object using the coordinate-space functions stored in `𝑈`, decay rates `Γ`, and gauge fields stored in `𝐴`. `𝐴[c, i]` is the `i`th projection `𝐴ᵢ` of cth component.
`M` is the maximum harmonic number. In the cis case, the Hamiltonian will be `nc*(2M+1)²`-by-`nc*(2M+1)²` where `nc` is the number of components. In sin/cos case, the size will be `nc*M²`-by-`nc*M²`.
`𝑈_iseven[i, j]` matters only if `basis=:cis` and shows whether `𝑈[i, j]` is an even function (i.e. whether 𝑢(𝑟) = 𝑢(-𝑟)). If it is, then Fourier transform is real, which is used for better accuracy.
If *all* functions are even (and real), then the resulting Fourier-space Hamiltonian is real (provided also there is no 𝐴 and Γ), giving a speed-up and better accuracy (compared to complex diagonalisation).
If `𝑈[i, j] ≡ nothing` or it is complex, then the value of `𝑈_iseven[i, j]` does not matter.
Currently it is assumed that if 𝐴's are present, then Hamiltonian is necessarily complex, but this is not true in general (it is real in the cis basis if A is real-even, exactly as for 𝑈).
"""
function DenseHamiltonian(xlims::AbstractVector{Tuple{R,R}},
                          𝑈::AbstractMatrix{<:Union{Function,Nothing}},
                          𝐴::AbstractVecOrMat{<:Union{Function,Nothing}}=fill(nothing, size(𝑈, 1), length(xlims));
                          basis::Symbol, M::Integer, δ::R=one(R),
                          𝑈_iseven::AbstractMatrix{Bool}=falses(size(𝑈)), Γ::Vector{R}=zeros(R, size(𝑈, 1))) where R <: AbstractFloat
    nc = size(𝑈, 1) # number of components
    D = length(xlims) # number of spatial dimensions
    L = [lims[2] - lims[1] for lims in xlims]

    # `H_isreal` will show if the resulting `H` will be real
    𝐴ᵢ_present = [any(𝐴ᵢ .!== nothing) for 𝐴ᵢ in eachcol(𝐴)] # `i` numbers projections; 𝐴ᵢ_present[i] = true if 𝐴ᵢ is nonzero for at least one component
    U_isreal = all( 𝑢([xlims[i][1] for i in eachindex(xlims)]...) isa Real for 𝑢 in 𝑈 if !isnothing(𝑢) ) # check if all functions in 𝑈 are real
    H_isreal = U_isreal && all(𝐴ᵢ_present .== false) && iszero(Γ) # without checking we assume that all 𝐴's are real. Can be generalised for the exotic cases of complex 𝐴.
    if basis == :cis # for periodic potential, also check if functions are even 
        H_isreal &= all(𝑈_iseven[𝑈 .!== nothing])
    end

    B = basis == :cis ? (2M+1)^D : M^D # size of each Hamiltonian block

    T = H_isreal ? R : Complex{R} # type of elements of the Hamiltonian
    H = zeros(T, nc*B, nc*B)

    ft = FourierTransformer(xlims, M; basis, target_real=U_isreal) # `target_real` will allocate a buffer for the imaginary part of the sin/cos-transform if some of 𝑈's are complex

    𝑈_diag_allequal = allequal(diagview(𝑈))
    𝐴ᵢ_allequal = [allequal(𝐴ᵢ) && !isnothing(𝐴ᵢ[1]) for 𝐴ᵢ in eachcol(𝐴)] # 𝐴ᵢ_allequal[i] shows if projection 𝐴ᵢ is the same for all components; note that this also checks if they are nothing

    makereal = (basis == :cis && H_isreal) # in this case the transform is actually real, but is stored in a complex array `ft.buff`; this will be passed to `fft_to_matrix` to drop imaginary part of `ft.buff`

    # treat diagonal blocks, adding the diagonal potentials 𝑈ᵢᵢ and 𝑝² (conditionally)
    for jH in 1:nc
        h = @view H[(jH-1)*B+1:jH*B, (jH-1)*B+1:jH*B] # a view of the `jH`th diagonal block
        h_set = false # shows if `h` has been set to something (i.e. etiher/both next two if's have been entered)
        if !isnothing(𝑈[jH, jH])
            transform!(ft, 𝑈[jH, jH])
            fft_to_matrix!(h, ft; makereal)
            h_set = true
            # @debug "Wrote 𝑈[$jH, $jH] into H[$jH, $jH]" # H[iH, jH] schematically means the block (`iH`, `jH`)
        end
        # Add 𝑝² if basis is sin/cos. But if there are no 𝐴's at all, add in the cis case too (if 𝐴's are present, then 𝑝ᵢ²'s will be added together with 𝐴ᵢ's)
        if basis != :cis || all(𝐴ᵢ_present .== false)
            h .+= make_p²(L, M, δ, basis)
            h_set = true
            # @debug "Added 𝑝² to H[$jH, $jH]"
        end
        # If all 𝑈 are equal, then copy the just-calculated first diagonal block into all other diagonal blocks, and break.
        # This can be triggered on the first iteration only, and only if 𝑈's are not all nothing
        if 𝑈_diag_allequal && h_set
            for iH in 2:nc
                copyto!(H, CartesianIndices(((iH-1)*B+1:iH*B, (iH-1)*B+1:iH*B)), h, CartesianIndices(h))
                # @debug "Copied H[1, 1] to H[$iH, $iH]"
            end
            break
        end
    end

    # treat diagonal blocks, adding the kinetic terms (𝑝ᵢ - 𝐴ᵢ)²
    if any(𝐴ᵢ_present)
        A_buff = Matrix{T}(undef, B, B)
        A_buff2 = similar(A_buff)
        for i in 1:D # iterate over projections of 𝐴
            if !𝐴ᵢ_present[i] && basis != :cis # if the projection 𝐴ᵢ is zero for all components, then skip 𝐴ᵢ. However, if basis is cis, we cannot skip because also need to add 𝑝ᵢ²
                continue
            end
            pᵢ = make_p_i(L, M, δ, basis, i)
            for c in 1:nc
                if isnothing(𝐴[c, i]) # then there is nothing to do, except adding 𝑝ᵢ² in the cis case
                    if basis == :cis
                        pᵢ .^= 2 # in-place squaring
                        H[(c-1)*B+1:c*B, (c-1)*B+1:c*B] .+= pᵢ
                        # @debug "Added p_$i^2 to H[$c, $c]"
                    end
                    continue
                end
                transform!(ft, 𝐴[c, i])
                fft_to_matrix!(A_buff, ft)

                if basis == :cis
                    A_buff .= pᵢ .- A_buff
                    mul!(A_buff2, A_buff, A_buff) # after this multiplication, `A_buff2` contains (𝑝ᵢ - 𝐴ᵢ)²
                else
                    A_buff2 .= im*(A_buff*pᵢ + pᵢ*A_buff) + A_buff^2 # The perfect square for (𝑝ᵢ - 𝐴ᵢ)² is much less accurate. TODO optimise multiplications
                end
                H[(c-1)*B+1:c*B, (c-1)*B+1:c*B] .+= A_buff2 # add to the curent block
                # @debug begin basis == :cis ? "Added (p_$i - 𝐴[$c, $i])^2 to H[$c, $c]" : "Added im(𝐴[$c, $i]*p_$i + p_$i*𝐴[$c, $i] + 𝐴[$c, $i]^2) to H[$c, $c]"; end
                if 𝐴ᵢ_allequal[i] # then add `A_buff2` to all other diagonal blocks and break
                    for iH in 2:nc
                        H[(iH-1)*B+1:iH*B, (iH-1)*B+1:iH*B] .+= A_buff2
                        # @debug begin basis == :cis ? "Added (p_$i - 𝐴[$c, $i])^2 to H[$iH, $iH]" : "Added im(𝐴[$c, $i]*p_$i + p_$i*𝐴[$c, $i] + 𝐴[$c, $i]^2) to H[$iH, $iH]"; end
                    end
                    break
                end
            end
        end
    end
    # add -iΓ/2
    for c in 1:nc
        if Γ[c] != 0
            H[diagind(H)[(c-1)*B+1:c*B]] .-= im*Γ[c]/2
        end
    end
    # treat off-diagonal blocks (will not be run for a single component)
    for jH in 2:nc
        for iH in 1:jH-1 # only upper triangle is scanned. The lower triangle is filled only if Γ is present
            isnothing(𝑈[iH, jH]) && continue
            transform!(ft, 𝑈[iH, jH])
            wi = (iH-1)*B+1:iH*B
            wj = (jH-1)*B+1:jH*B
            h = @view H[wi, wj] # a view of the required block
            fft_to_matrix!(h, ft; makereal)
            # @debug "Wrote 𝑈[$iH, $jH] into H[$iH, $jH]"

            if !iszero(Γ) # fill conjugate block if Γ is present (then we cannot use Hermitian view)
                H[wj, wi] .= h' # TODO possible to use `copyto!` ?
                # @debug "Copied H[$iH, $jH]' into H[$jH, $iH]"
            end
        end
    end

    # determine the type of eigenvalues 
    ishermitian = iszero(Γ) # if all `Γ`s are zeros, then Hamiltonian is Hermitian and the eigenvalues real
    S = ishermitian ? R : Complex{R} # type of eigenvalues
    
    # create empty placeholders
    ε = S[] # eigenvalues, can be complex for nonhermitian `H`, hence additional type `S`
    V = T[;;] # eigenvectors matrix
    ε_q = Array{S}(undef, ntuple(Returns(0), D+1)) # ε_q[n, iqx, iqy, ...] = `n`th band eigenvalue at momentum at indices (`iqx`, `iqy`)
    V_q = Array{T}(undef, ntuple(Returns(0), D+2)) # V_q[:, n, iqx, iqy, ...] = `n`th band eigenvector at momentum at indices (`iqx`, `iqy`)

    return DenseHamiltonian(xlims, L, M, δ, nc, basis, ishermitian, 𝑈, BitMatrix(𝑈_iseven), 𝐴, Γ, H, ε, V, ε_q, V_q, Wanniers{R}())
end

# """
# More efficient way of calculating
#     H = -Δ + im*(A_x*∂_x + A_y*∂_y + ∂_x*A_x + ∂_y*A_y) + A_x^2 + A_y^2 + U
# Currently intended for non-periodic calculation.
# """
# function sum_parts(A_x::Matrix{<:Real}, A_y::Matrix{<:Real}, ∂_y::Matrix{<:Real}, ∂_x::Matrix{<:Real}, U::Matrix{<:Real}, Δ)
#     H = complex.(U)
#     H += -Δ
#     unit = one(eltype(A_x))
#     null = zero(eltype(A_x))
#     symm!('L', 'L', unit, A_x, ∂_x, null, U) # we start using `U` as a buffer
#     symm!('R', 'L', unit, A_x, ∂_x, unit, U)
#     symm!('L', 'L', unit, A_y, ∂_y, unit, U)
#     symm!('R', 'L', unit, A_y, ∂_y, unit, U)
#     H .+= U .* im
#     symm!('L', 'L', unit, A_x, A_x, null, U)
#     H += U
#     symm!('L', 'L', unit, A_y, A_y, null, U)
#     H += U
#     return H
# end

"Helper function for q-diagonalisation that updates the diagonal blocks of `xh.H`."
function update_diag!(xh::DenseHamiltonian, U, K, QS, 𝑈_diag_allequal, 𝐴ᵢ_allequal, D, buff1, buff2)
    (;nc, M, Γ, H) = xh
    B = (2M + 1)^D
    for c in 1:nc
        H_block = @view H[(c-1)*B+1:c*B, (c-1)*B+1:c*B]
        for i in 1:D
            which_K = 𝐴ᵢ_allequal[i] ? 1 : c
            copy!(buff1, K[which_K, i])
            buff1 += LA.I*QS[i]
            mul!(buff2, buff1, buff1)
            if i == 1
                copyto!(H, CartesianIndices(((c-1)*B+1:c*B, (c-1)*B+1:c*B)), buff2, CartesianIndices(buff2))
                # @debug "Copied (K[$which_K, $i] + QS[$i])^2 into H[$c, $c]"
            else
                H_block .+= buff2
                # @debug "Added (K[$which_K, $i] + QS[$i])^2 to H[$c, $c]"
            end
        end

        if 𝑈_diag_allequal
            H_block .+= U[1]
            # @debug "Added U[1] to H[$c, $c]"
        elseif !isnothing(U[c])
            H_block .+= U[c]
            # @debug "Added U[$c] to H[$c, $c]"
        end
        if Γ[c] != 0
            H_block -= LA.I * im*Γ[c]/2
            # @debug "Added -im*Γ[$c]/2 to H[$c, $c]"
        end
    end
    return
end

"Convenience caller for the 1-component case, where `𝜓₀` is a function, `g` is a number, and `𝜓₀_iseven` is a Bool."
function propagate(xh::XSpaceHamiltonian{Storage,R}, 𝜓₀::Union{Function,AbstractVector}, g::R=zero(R);
                   𝜓₀_iseven::Bool=false, T_max::R, dt::R, itime::Bool=false, solver=(iszero(g) ? DE.LinearExponential() : DE.Tsit5()), nsaves::Integer=0) where {Storage,R}
    propagate(xh, [𝜓₀], [g;;]; 𝜓₀_iseven=[𝜓₀_iseven], T_max, dt, itime, solver, nsaves)
end

"""
Propagate the time-dependent Schrödinger or Gross-Pitaevskii (with nonlinearity matrix `g`) equation for the initial wave function `𝜓₀`.
`𝜓₀` is either a vector of functions (for each component) or a vector of vectors representing discretised functions.
 Set `itime=true` for imaginary time propagation.
`𝜓₀_iseven[c]` matters only if `xh.basis=:cis` and shows whether `𝜓₀[c]` is an even function (i.e. whether 𝜓(x) = 𝜓(-x)).
If it is, then Fourier transform is real; if `xh.H` is also real, the imaginary time propagation can be done for a real type.
`solver` is a solver from DifferentialEquations.jl. For SE, recommended are `LinearExponential` (default) or state-independent ones from https://docs.sciml.ai/DiffEqDocs/stable/solvers/nonautonomous_linear_ode/.
For GPE, the default is `Tsit5`.
Return the DifferentialEquations solution object. 
"""
function propagate(xh::XSpaceHamiltonian{Storage,R}, 𝜓₀::Union{AbstractVector{<:Function},AbstractVector{<:AbstractVector}}, g::AbstractMatrix{R}=zeros(R, xh.nc, xh.nc);
                   𝜓₀_iseven::AbstractVector{Bool}=falses(length(𝜓₀)), T_max::R, dt::R, itime::Bool=false, solver=(iszero(g) ? DE.LinearExponential() : DE.Tsit5()), nsaves::Integer=0) where {Storage,R}
    (;xlims, L, M, basis, nc) = xh
    D = length(xlims)
    B = basis == :cis ? (2M+1)^D : M^D # size of each Hamiltonian block

    # prepare p-space wave function
    if 𝜓₀ isa AbstractVector{<:Function} # `𝜓₀` a vector of analytic functions
        𝜓₀_isreal = [ 𝜓([xlims[i][1] for i in eachindex(xlims)]...) isa Real for 𝜓 in 𝜓₀ ]
    else # `𝜓₀` is a vector of vectors of discretised functions
        𝜓₀_isreal = [eltype(𝜓) isa Real for 𝜓 in 𝜓₀]
    end
    ψ₀_isreal = all(𝜓₀_isreal) # will show if `ψ₀` should be constructed real
    if basis == :cis # also check if functions are even 
        ψ₀_isreal &= all(𝜓₀_iseven)
    end
    ψ₀ = Vector{ψ₀_isreal && itime ? R : Complex{R}}(undef, nc*B)

    # transform each component's wf and put into ψ₀
    ft = FourierTransformer(xlims, M; basis, target_real=ψ₀_isreal, target_rank=1) # `target_real` will allocate a buffer for the imaginary part of the sin/cos-transform if 𝜓₀ is complex
    for c in 1:nc
        transform!(ft, 𝜓₀[c])
        ψ₀_block = @view ψ₀[(c-1)*B+1:c*B]
        fft_to_vector!(ψ₀_block, ft; makereal=(𝜓₀_iseven[c] && 𝜓₀_isreal[c]))
        normalize!(ψ₀_block)
    end

    if itime # propagation in imaginary time: equation is real if `xh.H` and `ψ₀` are real
        H = !ψ₀_isreal && eltype(xh.H) <: Real ? -complex(xh.H) : -xh.H # if `ψ₀` is complex then solver needs complex matrix. So cast `xh.H` to complex if it is real
        G = -g ./ L[1]
    else # propagation in real time: equation is always complex
        H = -im * xh.H
        G = -im * g
    end

    # initialise the problem
    tspan = (zero(R), T_max)
    if iszero(g) # nonlinearity absent
        prob = DE.ODEProblem(SciMLOperators.MatrixOperator(H), ψ₀, tspan)
    else # nonlinearity present
        # treat each case separately beacuse we don't want any loops in update functions
        # if nc == 1
        #     params = (diag(H), G[1]/L[1])
        #     A = SciMLOperators.MatrixOperator(H, update_func! = update_A_1comp!)
        # elseif nc == 2
        #     # first two ranges take two components from the solution vector; next two take the corresponding diagonal elements
        #     params = (diag(H), G, 1:B, B+1:2B, diagind(H)[1:B], diagind(H)[B+1:2B])
        #     A = SciMLOperators.MatrixOperator(H, update_func! = update_A_2comp!)
        # end
        # prob = DE.ODEProblem(A, ψ₀, tspan, params)
        params = (H, G[1], B)
        prob = DE.ODEProblem(gpe!, ψ₀, tspan, params)
    end

    # when `saveat` is set, saving happens at points `tspan[1]:saveat:tspan[2]`
    saveat = nsaves == 0 ? T_max+1 : (tspan[2] - tspan[1]) / nsaves

    if itime
        # prepare the callback that remormalises wave function at every step
        condition = Returns(true) # condition is checked at the end of each time step; we want this to be always true
        affect!(integrator) = normalize!(integrator.u)
        cb = DE.DiscreteCallback(condition, affect!) 
        return DE.solve(prob, solver; callback=cb, save_everystep=false, save_start=false, dt, saveat)
    else
        return DE.solve(prob, solver; save_everystep=false, save_start=true, dt, saveat)
    end
end

"Update the 𝑢′ matrix of the GPE."
function gpe!(du, u, params, t)
    H, g, B = params
    mul!(du, H, u)
    for p in eachindex(u)
        du[p] += g * sum(u[k] * sum(u[k′] * u[k′+k-p]' for k′ in max(1, 1+p-k):min(B, B+p-k)) for k in eachindex(u))
    end
    return
end

# "Update the 𝐴 matrix of the nonlinear equation 𝑢′(𝑡) = 𝐴(𝑢)𝑢(𝑡) in the 1-component case."
# function update_A_1comp!(A, u, params, t)
#     d, g = params # original diagonal of the Hamiltonian and the coupling strength
#     A[diagind(A)] .= d .+ g .* abs2.(u)
#     return
# end

# "Update the 𝐴 matrix of the nonlinear equation 𝑢′(𝑡) = 𝐴(𝑢)𝑢(𝑡) in the 2-component case."
# function update_A_2comp!(A, u, params, t)
#     d, g, w1, w2, dw1, dw2 = params
#     u₁ = @view u[w1]
#     u₂ = @view u[w2]
#     A[dw1] .= d[dw1] .+ g[1, 1].*abs2.(u₁) .+ g[1, 2].*abs2.(u₂)
#     A[dw2] .= d[dw2] .+ g[2, 1].*abs2.(u₁) .+ g[2, 2].*abs2.(u₂)
#     return
# end

"Return mean energy and chemical potential for a p-space state `v`. (currently, 1-component only)"
function get_ε_μ(xh, v, g=zeros(typeof(xh.δ), xh.nc, xh.nc))
    # (; nc) = xh
    L = xh.L[1]
    B = length(v)    
    ε = dot(v, xh.H, v)
    μ = ε
    if !iszero(g)
        U = zero(μ)
        for p′ in eachindex(v), p in eachindex(v), k in eachindex(v)
            k′ = k+p-p′
            (k′ < 1 || k′ > B) && continue
            U += conj(v[p′]*v[k′]) * v[p]*v[k] |> real
        end
        U *= g[1]/L
        μ += U
        ε += U / 2
    end
    return ε, μ
end