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

    # size of each Hamiltonian block
    B = basis == :cis ? (2M+1)^D :
        basis == :sin ?      M^D : (M+1)^D

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

            # copy adjoint of H[wi, wj] into H[wj, wi]. Needed for nonhermitian diagonalisation (cannot use Hermitian view) and also for GPE
            copy_adjoint!(H, wj, wi, H, wi, wj)
            # @debug "Copied H[$iH, $jH]' into H[$jH, $iH]"
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

"Convenience caller for the 1-component case, where `ψ₀` is an analytic function or a vector representing discretised functions, `g` is a number, and `ψ₀_iseven` is a Bool."
function propagate(xh::XSpaceHamiltonian{Storage, R}, ψ₀::Union{Function, AbstractVector}, g::R=zero(R);
                   ψ₀_iseven::Bool=false, T_max::R, dt::R, itime::Bool=false,
                   solver=(iszero(g) ? ODE.LinearExponential() : itime ? ODE.LawsonEuler() : ODE.ETDRK4()), nsaves::Integer=0) where {Storage, R}
    propagate(xh, [ψ₀], [g;;]; ψ₀_iseven=[ψ₀_iseven], T_max, dt, itime, solver, nsaves)
end

"""
Propagate the time-dependent Schrödinger or Gross-Pitaevskii (with nonlinearity matrix `g`) equation (SE and GPE, respectively) for the initial wave function `ψ₀`.
`ψ₀` can be:
    * a vector of x-space analytic functions (one for each component)
    * a vector of vectors (one for each component) representing discretised x-space functions
    * a vector representing discretised p-space functions, all lumped together 
Set `itime=true` for imaginary time propagation.
`ψ₀_iseven[c]` matters only if basis is cis, `g`s are zero (SE case), and `ψ₀` is given in x-space. It shows whether `ψ₀[c]` is an even function (i.e. whether ψ(x) = ψ(-x)).
If it is, then if `xh.H` is also real, the imaginary time propagation will be done for a real type.
`solver` is a solver from DifferentialEquations.jl. For SE, recommended are `LinearExponential` (default) or state-independent ones from https://docs.sciml.ai/DiffEqDocs/stable/solvers/nonautonomous_linear_ode/.
For GPE, recommended are the Semilinear Split ODE Solvers from https://docs.sciml.ai/DiffEqDocs/stable/solvers/split_ode_solve/.
For imaginary-time GPE, the default is `LawsonEuler`, which is first order (and hence fast), but is sufficient when the time step is small.
For real-time GPE, the default is `ETDRK4`; lower order variants can also be used for quick results. `HochOst4` seems to conserve the norm even better, but is a bit slower.
Return the DifferentialEquations solution object. 
"""
function propagate(xh::XSpaceHamiltonian{Storage, R, T}, ψ₀::Union{AbstractVector{<:Function}, AbstractVector{<:AbstractVector}, AbstractVector{<:Number}}, g::AbstractMatrix{R}=zeros(R, xh.nc, xh.nc);
                   ψ₀_iseven::AbstractVector{Bool}=falses(length(ψ₀)), T_max::R, dt::R, itime::Bool=false,
                   solver=(iszero(g) ? ODE.LinearExponential() : itime ? ODE.LawsonEuler() : ODE.ETDRK4()), nsaves::Integer=0) where {Storage, R, T}
    (;xlims, L, M, basis, nc) = xh
    D = length(xlims)
    # size of each Hamiltonian block
    B = basis == :cis ? (2M+1)^D :
        basis == :sin ?      M^D : (M+1)^D

    # determine if equation can be solved using real types. Note that for cis with nonzero `g` it cannot because intermediate FFT's will be yielding complex results
    eq_isreal = itime && T <: Real && !(basis == :cis && !iszero(g)) # below `eq_isreal` might change if initial state is complex

    # Prepare the p-space wf. We don't normalise it. E.g. in p-space the user might want to remove one component and propagate the rest, meaning that total norm is not one.
    # In x-space the user might use a stationary state calculated for a fixed 𝜇 and not necessarily unit norm.
    # Normalisation can have consequences since equation is nonlinear.
    if ψ₀ isa AbstractVector{<:Number} # `ψ₀` is given in p-space
        ψ₀_isreal = eltype(ψ₀) <: Real
        eq_isreal &= ψ₀_isreal
        ψ₀ₚ = ψ₀_isreal && !eq_isreal ? complex(ψ₀) : ψ₀  # if the passed initial is real but equation is not, then convert; otherwise take as-is
    else # `ψ₀` is given in x-space
        if ψ₀ isa AbstractVector{<:Function} # `ψ₀` a vector of analytic functions
            ψ₀_arereal = [ ψ([xlims[i][1] for i in eachindex(xlims)]...) isa Real for ψ in ψ₀ ]
        else # `ψ₀` is a vector of vectors of discretised functions
            ψ₀_arereal = [eltype(ψ) <: Real for ψ in ψ₀]
        end
        eq_isreal &= all(ψ₀_arereal)
        if basis == :cis && iszero(g) # then also check if functions are even 
            eq_isreal &= all(ψ₀_iseven)
        end
        ψ₀ₚ = Vector{eq_isreal ? R : Complex{R}}(undef, nc*B)

        # transform each component's wf and put into ψ₀ₚ
        ft = FourierTransformer(xlims, M; basis, target_real=all(ψ₀_arereal), target_rank=1) # `target_real=false` will allocate a buffer for the imaginary part of the sin/cos-transform if ψ₀ is complex
        for c in 1:nc
            transform!(ft, ψ₀[c])
            ψ₀ₚ_block = @view ψ₀ₚ[(c-1)*B+1:c*B]
            fft_to_vector!(ψ₀ₚ_block, ft; makereal=(ψ₀_iseven[c] && ψ₀_arereal[c]))
        end
    end

    # initialise the Hamiltonian and coupling matrix `G` with the appropriate sign and `im` factor
    if itime # propagation in imaginary time: equation is real if `xh.H` and `ψ₀ₚ` are real
        H = T <: Real && !eq_isreal ? -complex(xh.H) : -xh.H # `xh.H` is real but equation is not, then convert the Hamiltonian to complex. Solving then proceeds faster TODO: figure out why
        G = -g
    else # propagation in real time: equation is always complex
        H = -im * xh.H
        G = basis == :cis ? -im * g : -g # in the sin/cos case, do not include `im`
    end

    # Combine in `G` all fft normalisation factors so that this multiplication can be done just once at each step
    if basis == :cis
        N = 2M + 1 # number of points in each dimension
        dx = L ./ N
        G .*= prod(@. dx / L^2) # After bfft, resulting `u` must be divided by √𝐿; since we have `u^3`, we must divide by 𝐿√𝐿. Then, after fft the result must be multiplied by Δ𝑥/√𝐿. So Δ𝑥/𝐿² in total.
    elseif basis == :sin
        N = M
        dx = L ./ (N+1)
        G .*= prod(@. dx / (2L)^2) # Same as for cis but with √(2𝐿) instead of √𝐿
    else # basis == :cos
        N = M + 1
        dx = L ./ (N-1)
        G .*= prod(@. dx / (2L)^2) # Same as for cis but with √(2𝐿) instead of √𝐿
    end
    G_input = nc == 1 || allequal(G) ? G[1] : G # the `gpe_*` functions specialise on the equal-g case

    # initialise the problem
    tspan = (zero(R), T_max)
    # prepare the SciMLOperator based on the Hamiltonian. If `xh` describes a free system (no 𝑈 or 𝐴), then use `Diagonal`. Then matrix exponential is trivial, leading to immense speed-up
    H_op = all(isnothing.(xh.𝑈)) && all(isnothing.(xh.𝐴)) ? SciMLOperators.MatrixOperator(Diagonal(H)) : SciMLOperators.MatrixOperator(H)

    if iszero(g) # nonlinearity absent
        prob = ODE.ODEProblem(H_op, ψ₀ₚ, tspan)
    else # nonlinearity present
        ψ₀ₚ_block = @view ψ₀ₚ[1:B] # for constructing FFT plans and various buffers
        if basis == :cis
            # Both plans will operate on views in `gpe_cis_*`, and they seem to fail for Float32 on Windows/Intel due to alignment issues. The workaround is to pass `flags=FFTW.UNALIGNED`, see https://github.com/JuliaMath/FFTW.jl/issues/67
            bfft_plan = FFTW.plan_bfft(ψ₀ₚ_block) # will first use this and write the result into `du`
            fft_plan! = FFTW.plan_fft!(ψ₀ₚ_block) # then will use this in-place on `du`
            if nc == 1 # the 1-component case can be treated more efficiently
                params = (G_input, bfft_plan, fft_plan!)
                prob = ODE.SplitODEProblem(H_op, gpe_cis_realsin_1comp!, ψ₀ₚ, tspan, params) # use sepcialisation `ODEProblem{true, SciMLBase.FullSpecialize}` for production!
            else
                buff = [similar(ψ₀ₚ_block) for _ in 1:nc]
                params = (G, B, nc, buff, similar(ψ₀ₚ_block), bfft_plan, fft_plan!, basis)
                prob = ODE.SplitODEProblem(H_op, gpe_cis_realsincos!, ψ₀ₚ, tspan, params) # use sepcialisation `ODEProblem{true, SciMLBase.FullSpecialize}` for production!
            end
        else # basis == :sin || basis == :cos
            ft_type = basis == :sin ? FFTW.RODFT00 : FFTW.REDFT00
            if eq_isreal # basically, if solving imaginary-time GPE with a real Hamiltonian
                rft_plan  = FFTW.plan_r2r(ψ₀ₚ_block, ft_type)  # will first use this and write the result to the buffer
                rft_plan! = FFTW.plan_r2r!(ψ₀ₚ_block, ft_type) # then will use this in-place on that buffer
                if nc == 1 # the 1-component case can be treated more efficiently
                    if basis == :sin
                        params = (G_input, rft_plan, rft_plan!)
                        prob = ODE.SplitODEProblem(H_op, gpe_cis_realsin_1comp!, ψ₀ₚ, tspan, params) # use sepcialisation `ODEProblem{true, SciMLBase.FullSpecialize}` for production!
                    else # basis == :cos
                        params = (G_input, similar(ψ₀ₚ_block), rft_plan, rft_plan!)
                        prob = ODE.SplitODEProblem(H_op, gpe_realcos_1comp!, ψ₀ₚ, tspan, params) # use sepcialisation `ODEProblem{true, SciMLBase.FullSpecialize}` for production!
                    end
                else # arbitrary number of components
                    buff = [similar(ψ₀ₚ_block) for _ in 1:nc]
                    params = (G, B, nc, buff, similar(ψ₀ₚ_block), rft_plan, rft_plan!, basis)
                    prob = ODE.SplitODEProblem(H_op, gpe_cis_realsincos!, ψ₀ₚ, tspan, params) # use sepcialisation `ODEProblem{true, SciMLBase.FullSpecialize}` for production!
                end
            else # solving complex equation
                rft_plan! = FFTW.plan_r2r!(real(ψ₀ₚ_block), ft_type)
                if nc == 1
                    params = (G_input, similar(ψ₀ₚ_block, R), similar(ψ₀ₚ_block, R), similar(ψ₀ₚ_block, R), rft_plan!, basis)
                    prob = ODE.SplitODEProblem(H_op, gpe_complexsincos_1comp!, ψ₀ₚ, tspan, params) # use sepcialisation `ODEProblem{true, SciMLBase.FullSpecialize}` for production!
                else
                    # using vectors of vectors instead of contiguous vectors is ~10% faster and x1000 less memory
                    u_re = [similar(ψ₀ₚ_block, R) for _ in 1:nc]
                    u_im = [similar(ψ₀ₚ_block, R) for _ in 1:nc]
                    u² = [similar(ψ₀ₚ_block, R) for _ in 1:nc]
                    params = (G_input, B, nc, u_re, u_im, u², similar(ψ₀ₚ_block, R), rft_plan!, basis)
                    prob = ODE.SplitODEProblem(H_op, gpe_complexsincos!, ψ₀ₚ, tspan, params) # use sepcialisation `ODEProblem{true, SciMLBase.FullSpecialize}` for production!
                end
            end
        end
    end

    if itime
        # prepare the callback that remormalises wf at every step
        condition = Returns(true) # condition is checked at the end of each time step; we want this to be always true
        affect!(integrator) = normalize!(integrator.u)
        cb = ODE.DiscreteCallback(condition, affect!) # will save every step before and after the callback (`save_positions=(true, true)`); docs say this is mandatory when change of `u` is discontinuous
        sol = ODE.solve(prob, solver; callback=cb, save_everystep=false, save_start=true, dt)
        normalize!(sol.u[end]) # the final step is saved only before the callback, so normalise manually
        return sol
    else
        # when `saveat` is set, saving happens at points `tspan[1]:saveat:tspan[2]`
        saveat = nsaves == 0 ? T_max+1 : (tspan[2] - tspan[1]) / nsaves
        return ODE.solve(prob, solver; save_everystep=false, save_start=true, dt, saveat)
    end
end

"""
Update the 𝑢′ vector of the nonlinear part of the 1-component GPE
    𝑢′ = 𝑔|𝑢|²𝑢
The 𝑔 must contain `im` (for real-time propagation) and the proper sign.
Suitable for cases: (1) basis is cis; (2) basis is sin and equation is real.
"""
function gpe_cis_realsin_1comp!(du, u, params, t)
    g, bft_plan, ft_plan! = params
    mul!(du, bft_plan, u) # transform `u` and write into `du`
    @. du *= g * abs2(du)
    ft_plan! * du # in-place transform of `du`
    return
end

"""
Update the 𝑢′ vector of the nonlinear part of the GPE
    𝑢′ᵢ = ∑ⱼ 𝑔ᵢⱼ|𝑢ⱼ|²𝑢ᵢ
The 𝑔ᵢⱼ's must contain `im` (for real-time propagation) and the proper sign.
Suitable for cases: (1) basis is cis; (2) basis is sin/cos and equation is real.
"""
function gpe_cis_realsincos!(du, u, params, t)
    g, B, nc, u², u²_sum, bfft_plan, fft_plan!, basis = params
    # for each `i`th component, transform 𝑢ᵢ to x-space and write into `du`. Also, calculate |𝑢ᵢ|² and store in `u²`
    @views for i in 1:nc
        window = (i-1)B+1:i*B
        du_i = du[window] # view into the relevant component
        if basis == :cos # then must normalise 0th and last harmonics before transforming; will use `u²_sum` as a buffer
            copyto!(u²_sum, 1, u, (i-1)B+1, B) # copy `B` elements of `u`, starting from `(i-1)B+1`th into `u²_sum`, starting from index 1
            u²_sum[1] *= √2; u²_sum[end] *= √2
            mul!(du_i, bfft_plan, u²_sum) # transform `u²_sum` and write into `du`
        else
            mul!(du_i, bfft_plan, u[window]) # transform `u` and write into `du`
        end
        @. u²[i] = abs2(du_i)
    end
    # for each `i`th component, calculate 𝑢ᵢ∑ⱼ𝑔ᵢⱼ|𝑢ⱼ|² storing the result in the appropriate block of `du`.
    u∑gu²_complex!(du, u², u²_sum, g, nc, B)
    # transform `du` to p-space in-place
    for i in 1:nc
        @views fft_plan! * du[(i-1)B+1:i*B]
        basis == :cos && (du[(i-1)B+1] /= √2; du[i*B] /= √2)
    end
    return
end

"""
For each `i`th component, calculate 𝑢ᵢ∑ⱼ𝑔ᵢⱼ|𝑢ⱼ|² storing the result in the appropriate block of `du`.
"""
function u∑gu²_complex!(du, u², u²_sum, g::AbstractMatrix{<:Number}, nc, B)
    for i in 1:nc
        @. u²_sum = g[i, 1] * u²[1]
        for j in 2:nc
            g[i, j] == 0 && continue
            @. u²_sum += g[i, j] * u²[j]
        end
        @views du[(i-1)B+1:i*B] .*= u²_sum
    end
    return
end

"""
For each `i`th component, calculate 𝑢ᵢ𝑔∑ⱼ|𝑢ⱼ|² storing the result in the appropriate block of `du`.
"""
function u∑gu²_complex!(du, u², u²_sum, g::Number, nc, B) # `u²_sum` is not used but kept for compatibility with the other method
    for i in 2:nc
        @turbo u²[1] .+= u²[i]
    end
    u²[1] .*= g
    for i in 1:nc
        @views du[(i-1)B+1:i*B] .*= u²[1]
    end
    return
end

"""
Update the 𝑢′ vector of the nonlinear part of the 1-component GPE
    𝑢′ = 𝑔|𝑢|²𝑢
The 𝑔 must contain the proper sign.
Suitable for the case: basis is cos and equation is real.
"""
function gpe_realcos_1comp!(du, u, params, t)
    g, u_buff, rft_plan, rft_plan! = params
    copy!(u_buff, u) # because of the next step; cannot do it for `u` (not allowed to change `u`)
    u_buff[1] *= √2; u_buff[end] *= √2
    mul!(du, rft_plan, u_buff)
    @turbo @. du *= g * du^2
    rft_plan! * du
    du[1] /= √2; du[end] /= √2
    return
end

"""
Update the 𝑢′ vector of the nonlinear part of the 1-component GPE
    𝑢′ = i𝑔|𝑢|²𝑢
The 𝑔 must contain the proper sign but must NOT contain `im`.
Suitable for the case: basis is sin/cos and equation is complex.
"""
function gpe_complexsincos_1comp!(du, u, params, t)
    g, u_re, u_im, u², rft_plan!, basis = params
    # split re and im
    for i in eachindex(u)
        u_re[i], u_im[i] = reim(u[i])
    end
    basis == :cos && (u_re[1] *= √2; u_re[end] *= √2; u_im[1] *= √2; u_im[end] *= √2)
    # transform to x-space
    rft_plan! * u_re
    rft_plan! * u_im
    # calculate |𝑢(𝑥)|²
    @turbo @. u² = u_re^2 + u_im^2
    # calculate 𝑢(𝑥)|𝑢(𝑥)|²
    @turbo @. u_re *= u²
    @turbo @. u_im *= u²
    # transform to p-space
    rft_plan! * u_re
    rft_plan! * u_im
    # add re and im
    @. du = g * (im * u_re - u_im) # recall additional `im` from the equation, not contained in `g`
    basis == :cos && (du[1] /= √2; du[end] /= √2)
    return
end

"""
Update the 𝑢′ vector of the nonlinear part of the multi-component GPE
    𝑢′ᵢ = i ∑ⱼ 𝑔ᵢⱼ|𝑢ⱼ|²𝑢ᵢ
The 𝑔ᵢⱼ's must contain the proper sign but must NOT contain `im`.
Suitable for the cases: basis is sin/cos and equation is complex.
"""
function gpe_complexsincos!(du, u, params, t)
    # u_re, u_im, u² -- vectors of `nc` vectors. `u²_sum` -- 1-component length, and it's real
    g, B, nc, u_re, u_im, u², u²_sum, rft_plan!, basis = params
    # for each `i`th component, transform 𝑢ᵢ to x-space and write into `uᵢ_re` and `uᵢ_im`. Also, calculate |𝑢ᵢ|² and write into `u²`
    for i in 1:nc
        # split re and im
        for b in 1:B
            u_re[i][b], u_im[i][b] = reim(u[(i-1)B+b])
        end
        # transform to x-space
        basis == :cos && (u_re[i][1] *= √2; u_re[i][end] *= √2; u_im[i][1] *= √2; u_im[i][end] *= √2)
        rft_plan! * u_re[i]
        rft_plan! * u_im[i]
        @turbo @. u²[i] = u_re[i]^2 + u_im[i]^2
    end
    # for each `i`th component, calculate 𝑢ᵢ∑ⱼ𝑔ᵢⱼ|𝑢ⱼ|² overwriting 𝑢ᵢ, stored in `u_re[i]` and `u_im[i]`
    u∑gu²_real!(u_re, u_im, u², u²_sum, g, nc) # this specialises for the cases when `g` is a number (this is the case when all are 𝑔's equal) or an array
    # transform `du` to p-space in-place
    for i in 1:nc
        # transform to p-space
        rft_plan! * u_re[i]
        rft_plan! * u_im[i]
        # add re and im
        basis == :cos && (u_re[i][1] /= √2; u_re[i][end] /= √2; u_im[i][1] /= √2; u_im[i][end] /= √2)
        @. du[(i-1)B+1:i*B] = Complex(-u_im[i], u_re[i])  # recall additional `im` from the equation, not contained in `g`. So we do `im * u_re - u_im`
    end
    return
end

"""
For each `i`th component, calculate 𝑢ᵢ∑ⱼ𝑔ᵢⱼ|𝑢ⱼ|² overwriting 𝑢ᵢ, stored in `u_re[i]` and `u_im[i]`.
"""
function u∑gu²_real!(u_re, u_im, u², u²_sum, g::AbstractMatrix{<:Number}, nc)
    for i in 1:nc
        @turbo @. u²_sum = g[i, 1] * u²[1]
        for j in 2:nc
            g[i, j] == 0 && continue
            @turbo @. u²_sum += g[i, j] * u²[j]
        end
        @turbo u_re[i] .*= u²_sum
        @turbo u_im[i] .*= u²_sum
    end
    return
end

"""
For each `i`th component, calculate 𝑢ᵢ𝑔∑ⱼ|𝑢ⱼ|² overwriting 𝑢ᵢ, stored in `u_re[i]` and `u_im[i]`.
"""
function u∑gu²_real!(u_re, u_im, u², u²_sum, g::Number, nc) # `u²_sum` is not used but kept for compatibility with the other method
    for i in 2:nc
        @turbo u²[1] .+= u²[i]
    end
    @turbo u²[1] .*= g
    for i in 1:nc
        @turbo u_re[i] .*= u²[1]
        @turbo u_im[i] .*= u²[1]
    end
    return
end

"""
For a state `v`, return `E, μ, η`, where `E` is mean energy per particle, μ is a vector of chemical potentials of each compoenent,
and `η` is a vector of relative particle numbers of each compoenent.
`v_is_pspace=true` means that `v` is given in p-space, and x-space otherwise.
By default, `makereal=true` so that the returned `E` and `μ` are made real (by dropping imaginary part). Set `makereal=false` if you consider a decaying state, whereby imaginary part is important.
"""
function get_Eμη(xh::XSpaceHamiltonian{Storage, R}, v::AbstractVector{<:Number}, g::AbstractMatrix{<:Number}=zeros(typeof(xh.δ), xh.nc, xh.nc);
                 v_is_pspace=true, makereal=true) where {Storage, R}
    (;xlims, M, nc, basis) = xh
    B = size(xh.H, 1) ÷ nc  
     
    if v_is_pspace # if `v` is in p-space, then make `vₚ` point to `v`
        vₚ = v
    else # if `v` is in x-space, then perform FT to transition to p-space
        v_isreal = eltype(v) <: Real
        ft = FourierTransformer(xlims, M; basis, target_real=v_isreal, target_rank=1, isforward=true)
        v_type = !v_isreal ? Complex{R} : eltype(ft.buff) # if v in x-space is complex, then result will be complex; otherwise the same as determined in `ft`
        vₚ = Vector{v_type}(undef, length(v))
        @views for c in 1:nc
            window = (c-1)B+1:c*B
            transform!(ft, v[window])
            fft_to_vector!(vₚ[window], ft)
        end
    end

    η = [@views sum(abs2, vₚ[(c-1)B+1:c*B]) for c in 1:nc]
    η_total = sum(η)
    e = [@views dot(vₚ[(c-1)B+1:c*B], xh.H[(c-1)B+1:c*B, :], vₚ[1:nc*B]) for c in 1:nc] # using `vₚ[1:nc*B]` instead of just `vₚ` because it might contain chemical potentials as the last `nc` elements
    E = sum(e) / η_total
    μ = e ./ η
    
    if !iszero(g)
        # we need `ft` object to get the coordinates
        v_isreal = eltype(v) <: Real
        ft = FourierTransformer(xlims, M; basis, target_real=v_isreal, target_rank=1, isforward=false)
        dV = prod(ft.xs[2, i] - ft.xs[1, i] for i in axes(ft.xs, 2)) # volume element
        if v_is_pspace # if `v` is in p-space, then perform FT to x-space
            # create an array of arrays holding squared x-space densities |𝜓(𝑥)|² for each component
            ψ² = map(1:nc) do c
                @views transform!(ft, v[(c-1)B+1:c*B])
                ψ = fft_to_vector(ft)
                ψ .= abs2.(ψ)
                return ψ
            end
        else # if `v` is in x-space, then calculate abs2 directly, but we need a vector of vectors instead of contiguous
            ψ² = map(1:nc) do c
                @views abs2.(v[(c-1)B+1:c*B])
            end
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
            U = sum(ψ²_sum) * dV # for sin, endpoints are not included but are zero, so this is equivalent to the trapezoid rule. For cis, rectangle rule is more appropriate because there is no boundary
            basis == :cos && (U -= (ψ²_sum[1] + ψ²_sum[end])/2 * dV)
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

"""
Find the stationary state of the Gross-Pitaevskii equation (with nonlinearity matrix `g`) starting from the initial guess `ψ₀`.
Namely, solve the 𝑛-component system
    (𝐻𝑢)ᵢ + (∑ⱼ 𝑔ᵢⱼ|𝑢ⱼ|² - 𝜇ᵢ) 𝑢ᵢ = 0
The solver support 3 modes:
    1. (Default). Number of atoms is not fixed, chemical potentials are fixed.
        Then `natoms=nothing`, while `μ` is a vector of 𝜇ᵢ's of each component. In the 1-component case, `μ` is a scalar (a vector of one element will also work, but scalar is preferred).
    2. Number of atoms in each component is fixed, chemical potentials are not fixed. In the 1-component case, this is not applicable -- use mode 3 instead.
        Then `natoms` is a vector of number of atoms in each component, while `μ` is a vector of guesses of 𝜇ᵢ of each component.
        The system is augmented with 𝑛 equations
            ∫𝑢ᵢ²d𝑥 - 𝑁ᵢ = 0
    3. Total number of atoms is fixed, chemical potentials are not fixed (but will be the same for all components).
        Then `natoms` is the total number of atoms (a scalar), while `μ` is an initial guess of 𝜇 (also a scalar). This works in the 1-component case as well.
        The system is augmented with the equation
            ∑ᵢ∫𝑢ᵢ²d𝑥 - 𝑁 = 0
`ψ₀` can be:
    * a vector of x-space analytic functions (one for each component);
    * a vector of vectors (one for each component) representing discretised x-space functions.
`solver` is a solver from NonlinearSolvers.jl. We do not construct a concrete Jacobian but rather declare its action on a vector. Autodiff will fail because it doesn't work with FFT, which we are using.
Therefore, when passing the solver, always turn off concrete Jacobian and/or set linear solving to an iterative method.
Default solver is `NewtonRaphson(;linsolve=KrylovJL_GMRES())`.
You can also try using BICSTAB and/or Eisenstat-Walker forcing as in `NewtonRaphson(;linsolve=KrylovJL_BICGSTAB(), forcing=EisenstatWalkerForcing2())`.
Forcing might fail to converge, but it accelerates solving since otherwise the linear system is solved to the same accuracy as the nonlinear system, which is often (but not always!) reundant.
Default termination mode is `AbsNormSafeBestTerminationMode` with L-inf norm with `abstol` defined in NonlinearSolve.
Any additional keyword arguments will be passed directly to `NonlinearSolve.solve()`.
Return the tuple consisting of the coordinates and the NonlinearSolution object.
"""
function find_stationary(xh::XSpaceHamiltonian{Storage, R, T}, ψ₀::Union{AbstractVector{<:Function}, AbstractVector{<:AbstractVector}},
                         g::AbstractMatrix{R}, μ::Union{R, AbstractVector{<:R}}, natoms::Union{Nothing, R, AbstractVector{<:R}}=nothing;
                         solver=NLS.NewtonRaphson(;linsolve=LS.KrylovJL_GMRES()), kwargs...) where {Storage, R, T}
    (;xlims, M, basis, nc) = xh

    # determine if equation is real
    if ψ₀ isa AbstractVector{<:Function} # `ψ₀` is a vector of analytic functions: need to sample them
        ψ₀_arereal = all( ψ([xlims[i][1] for i in eachindex(xlims)]...) isa Real for ψ in ψ₀ )
    else # `ψ₀` is a vector of vectors of discretised functions: simply put them into a contiguous vector
        ψ₀_arereal = all(eltype(ψ) <: Real for ψ in ψ₀)
    end
    eq_isreal = ψ₀_arereal && all(isnothing.(xh.𝐴)) # equations are real (in x-space) if Hamiltonian and wfs are real (in x-space)
    ft_forward  = FourierTransformer(xlims, M; basis, target_real=eq_isreal, target_rank=1, isforward=true) # `target_real=false` will allocate a buffer for the imaginary part of the sin/cos-transform if ψ₀ is complex
    ft_backward = FourierTransformer(xlims, M; basis, target_real=eq_isreal, target_rank=1, isforward=false) # `target_real=false` will allocate a buffer for the imaginary part of the sin/cos-transform if ψ₀ is complex
    nx = length(ft_forward.xs)
    
    # Prepare the input wf `ψ_input`. By default, its length is `nc*nx`, but if `natoms` is passed then we need additional `nc` elements to represent the μ's that are being optimised.
    # Even if only total 𝑁 is fixed (and hence there is only one 𝜇 to be optimised), we still add `nc` elements to keep the general structure
    if ψ₀ isa AbstractVector{<:Function} # `ψ₀` is a vector of analytic functions: need to sample them
        if eq_isreal
            # sample each function in ψ₀ at points `ft_forward.xs`
            ψ_input = Vector{R}(undef, nc*(nx + !isnothing(natoms)))
            for c in 1:nc
                ψ_input[(c-1)*nx+1:c*nx] .= ψ₀[c].(ft_forward.xs)
            end
        else # equations in x-space are complex
            ψ_input = Vector{R}(undef, nc*(2nx+!isnothing(natoms)))
            for c in 1:nc, ix in 1:nx
                ψ_input[(c-1)*2nx + ix], ψ_input[(c-1)*2nx + nx+ix] = reim(ψ₀[c].(ft_forward.xs[ix]))
            end
        end
    else # `ψ₀` is a vector of vectors of discretised functions: put them into a contiguous vector
        ψ_input = Vector{eq_isreal ? R : Complex{R}}(undef, nc*(nx+!isnothing(natoms)))
        for c in 1:nc
            ψ_input[(c-1)*nx+1:c*nx] .= ψ₀[c]
        end
    end
    if !isnothing(natoms)
        ψ_input[end-nc+1:end] .= μ # use the passed `μ` as the initial guess (a single number if total 𝑁 is fixed or a vector otherwise; broadcast handles both cases)
    end

    # prepare the buffers needed in momentum space
    if basis == :cis || eq_isreal # in the cis case, the p-space buffer must be complex, so copy `ft_forward.buff` -- it's always complex for cis. If `eq_isreal` and basis is sin/cos, then also copy `ft_forward.buff` -- it's always real for sin/cos
        uₚ_buff  = similar(ft_forward.buff, nc*length(ft_forward.buff))
        uₚ_buff2 = similar(ft_forward.buff, nc*length(ft_forward.buff))
    else # !eq_isreal and basis is sin/cos
        uₚ_buff = similar(ft_forward.buff, Complex{R}, nc*length(ft_forward.buff))
        uₚ_buff2 = similar(ft_forward.buff, Complex{R}, nc*length(ft_forward.buff))
    end

    if isnothing(natoms) # = numbers of atoms are not fixed, but chemical potentials are
        μs_or_Ns = μ # so just pass the fixed chemical potentials
    else # total number of atoms or number of atoms in each component is fixed
        μs_or_Ns = natoms # pass the fixed numbers of atoms (a single number if total 𝑁 is fixed or a vector otherwise)
    end

    B = length(ft_forward.buff)
    if nc == 1 # the 1-component case can be treated more efficiently
        params = (xh.H, g[1], μs_or_Ns, B, uₚ_buff, uₚ_buff2, ft_forward, ft_backward)
        nlfunction = NLS.NonlinearFunction(nls_gpe_1comp!; jvp=jvp_gpe_1comp!)
        prob = NLS.NonlinearProblem(nlfunction, ψ_input, params)
    else
        # initialise the buffers for holding all double products
        u²_sum = Vector{R}(undef, B)
        u² = [similar(u²_sum) for _ in 1:nc]
        uⱼvⱼ = [similar(u²_sum) for _ in 1:nc]
        params = (xh.H, g, μs_or_Ns, B, nc, uₚ_buff, uₚ_buff2, u², u²_sum, uⱼvⱼ, ft_forward, ft_backward)
        nlfunction = NLS.NonlinearFunction(gpe_real_xspace!; jvp=jvp_gpe_real_xspace!)
        prob = NLS.NonlinearProblem(nlfunction, ψ_input, params) # use sepcialisation `NonlinearProblem{true, SciMLBase.FullSpecialize}` for production!
    end

    # we will pass on user's kwargs to NLS.solve, but we override some of NLS's defaults. User's kwargs will in turn override ours.
    finalkwargs = (;verbose=false, termination_condition=NLS.AbsNormSafeBestTerminationMode(Base.Fix1(maximum, abs)), kwargs...)
    
    return ft_forward.xs, NLS.solve(prob, solver; finalkwargs...)
end

"""
Update the x-space 𝑢′ vector of the 1-component GPE
    𝑢′ = 𝐻𝑢 + 𝑔𝑢²𝑢 - 𝜇𝑢
    ∫𝑢²d𝑥 - 𝑁 = 0     [present if the number of atoms is fixed]
Used for finding the steady state with nonlinear solve.
Suitable for the case when 𝑢 is real in x-space.
"""
function nls_gpe_1comp!(du, u, params)
    H, g, μ_or_N, B, uₚ_buff, uₚ_buff2, ft_forward, ft_backward = params
    # make `μ` point to the chemical potentials: those contained in `μs_or_Ns` if μ's are fixed, or last elements of `u` otherwise
    μ_isfixed = length(u) == B # is 𝜇's are not fixed, then `length(u)` exceeds `B` because `u` then also contains 𝜇
    μ = μ_isfixed ? μ_or_N : u[end] # if 𝜇 is fixed, the `μ_or_N` contains the fixed chemical potential
    # transform `u` to p-space, multiply by `H` and transform back
    transform!(ft_forward, @view u[1:B])
    fft_to_vector!(uₚ_buff, ft_forward)
    mul!(uₚ_buff2, H, uₚ_buff)
    transform!(ft_backward, uₚ_buff2)
    @views fft_to_vector!(du[1:B], ft_backward; makereal=true) # we assume that `u` is real, so the result here must be real: pass `make_real=true` to drop imaginary part in the cis case
    # add g and μ terms
    @views @. du[1:B] += (g * abs2(u[1:B]) - μ) * u[1:B]
    if !μ_isfixed # then update the last element of `du` representing the residual ∫𝑢²d𝑥 - 𝑁. In this case, `μ_or_N` contains 𝑁.
        dx = ft_forward.xs[2] - ft_forward.xs[1]
        @views du[end] = sum(abs2, u[1:B])*dx - μ_or_N
        ft_forward.basis == :cos && (du[end] -= (u[1]^2 + u[B]^2)*dx/2)
    end
    return
end

"""
Describes the action of the Jacobian of the 1-component GPE on an x-space vector 𝑣:
    𝐽𝑣 = 𝐻𝑣 + 3𝑔𝑢²𝑣 - 𝜇𝑣
If the number of atoms is fixed, then the last element of `v` and `u` is assumed to contain the chemical potential. Then, equation is
    𝐽𝑣 = 𝐻𝑣 + 3𝑔𝑢²𝑣 - 𝜇𝑣 - 𝑀𝑢
where 𝑀 is the last elements of `v`, 𝜇 is the last elements of `u`, and an additional equation reads
    𝐽𝑀 = 2∫d𝑥 𝑢𝑣
Used for finding the steady state with nonlinear solve.
Suitable for the case when 𝑢 and 𝑣 are real in x-space.
"""
function jvp_gpe_1comp!(Jv, v, u, params)
    H, g, μ_or_N, B, vₚ_buff, vₚ_buff2, ft_forward, ft_backward = params
    # make `μ` point to the chemical potentials: those contained in `μs_or_Ns` if μ's are fixed, or last elements of `u` otherwise
    μ_isfixed = length(u) == B # is 𝜇's are not fixed, then `length(u)` exceeds `B` because `u` then also contains 𝜇
    μ = μ_isfixed ? μ_or_N : u[end] # if 𝜇 is fixed, the `μ_or_N` contains the fixed chemical potential
    # transform `v` to p-space, multiply by `H` and transform back
    transform!(ft_forward, @view v[1:B])
    fft_to_vector!(vₚ_buff, ft_forward)
    mul!(vₚ_buff2, H, vₚ_buff)
    transform!(ft_backward, vₚ_buff2)
    @views fft_to_vector!(Jv[1:B], ft_backward; makereal=true)
    # add g and μ terms
    @views @. Jv[1:B] += (3g * abs2(u[1:B]) - μ) * v[1:B]
    if !μ_isfixed # then subtract the additional term 𝑀𝑢 and update the last element of `Jv` representing the additional equation. In this case, `μ_or_N` contains 𝑁.
        @views @. Jv[1:B] -= v[end] * u[1:B] # subtract 𝑀𝑢
        # set 𝐽𝑀 = 2∫d𝑥 𝑢𝑣
        dx = ft_forward.xs[2] - ft_forward.xs[1]
        @views Jv[end] = 2dot(u[1:B], v[1:B]) * dx
        ft_forward.basis == :cos && (Jv[end] -= (u[1]*v[1] + u[B]*v[B])*dx)
    end
    return
end

"""
Update the x-space 𝑢′ vector of the multi-component GPE
    𝑢′ᵢ = (𝐻𝑢)ᵢ + (∑ⱼ 𝑔ᵢⱼ𝑢ⱼ² - 𝜇ᵢ)𝑢ᵢ
    ∫𝑢ᵢ²d𝑥 - 𝑁ᵢ = 0     [present if the numbers of atoms are fixed]
Used for finding the steady state with nonlinear solve.
Suitable for the case when 𝑢 is real in x-space.
"""
function gpe_real_xspace!(du, u, params)
    H, g, μs_or_Ns, B, nc, uₚ_buff, uₚ_buff2, u², u²_sum, uⱼvⱼ, ft_forward, ft_backward = params
    # make `μ` point to the chemical potentials: those contained in `μs_or_Ns` if μ's are fixed, or last elements of `u` otherwise
    μs_arefixed = length(u) == B*nc # is 𝜇's are not fixed, then `length(u)` exceeds `B*nc` because `u` then also contains the 𝜇's
    μ = μs_arefixed ? μs_or_Ns : @view u[end-nc+1:end] # if total number of atoms is fixed, then these elements will all be the same
    ### Linear part
    # transform `u` to p-space, write into `uₚ_buff`
    @views for i in 1:nc
        window = (i-1)B+1:i*B
        transform!(ft_forward, u[window])
        fft_to_vector!(uₚ_buff[window], ft_forward)
    end
    mul!(uₚ_buff2, H, uₚ_buff)
    # transform `uₚ_buff2` back to x-space, write into `du`
    @views for i in 1:nc
        window = (i-1)B+1:i*B
        transform!(ft_backward, uₚ_buff2[window])
        fft_to_vector!(du[window], ft_backward; makereal=true)
    end
    ### Nonlinear part
    # pre-calculate 𝑢ᵢ² for each component and store in `u²`
    @views for i in 1:nc
        @turbo @. u²[i] = u[(i-1)B+1:i*B]^2
    end
    # add (∑ⱼ𝑔ᵢⱼ𝑢ⱼ² - 𝜇)𝑢ᵢ to `duᵢ`
    @views for i in 1:nc
        @turbo @. u²_sum = g[i, 1] * u²[1]
        for j in 2:nc
            g[i, j] == 0 && continue
            @turbo @. u²_sum += g[i, j] * u²[j]
        end
        window = (i-1)B+1:i*B
        duᵢ = du[window] # must create a view separately for @turbo to work in the next line
        @turbo @. duᵢ += (u²_sum - μ[i]) * u[window]
    end
    if !μs_arefixed # then update last `nc` elements of `du` representing residuals ∫𝑢ᵢ²d𝑥 - 𝑁ᵢ. In this case, `μs_or_Ns` contains 𝑁ᵢ's.
        dx = ft_forward.xs[2] - ft_forward.xs[1]
        if μs_or_Ns isa Number # then only total number of atoms is fixed
            u²_sum = zero(μs_or_Ns) # for storing the sum ∑ᵢ∫𝑢ᵢ²d𝑥
            for i in 1:nc
                u²_sum += sum(u²[i])*dx
                ft_forward.basis == :cos && (u²_sum -= (u²[i][end] + u²[i][1])*dx/2)
            end
            du[end-nc+1:end] .= u²_sum - μs_or_Ns # place the sum in the residuals array; we have `nc` identical elements to keep the general structure
        else # numbers of atoms in each compoenent are fixed
            for i in 1:nc
                du[end-nc+i] = sum(u²[i])*dx - μs_or_Ns[i]
                ft_forward.basis == :cos && (du[end-nc+i] -= (u²[i][end] + u²[i][1])*dx/2)
            end
        end
    end
    return
end

"""
Describes the action of the Jacobian of the multi-component GPE on an x-space vector 𝑣:
    (𝐽𝑣)ᵢ = (𝐻𝑣)ᵢ + 2𝑢ᵢ∑ⱼ𝑔ᵢⱼ𝑢ⱼ𝑣ⱼ + (3𝑔ᵢᵢ𝑢ᵢ² + ∑ⱼ𝑔ᵢⱼ𝑢ⱼ² - 𝜇ᵢ)𝑣ᵢ   [∑ⱼ excludes i]
If the numbers of atoms in each component are fixed, then last 𝑛 elements of `v` and `u` are assumed to contain the chemical potentials. Then, equations are
    (𝐽𝑣)ᵢ = (𝐻𝑣)ᵢ + 2𝑢ᵢ∑ⱼ𝑔ᵢⱼ𝑢ⱼ𝑣ⱼ - 𝑀ᵢ𝑢ᵢ + (3𝑔ᵢᵢ𝑢ᵢ² + ∑ⱼ𝑔ᵢⱼ𝑢ⱼ² - 𝜇ᵢ)𝑣ᵢ   [∑ⱼ excludes i]
where 𝑀ᵢ are 𝑛 last elements of `v`, 𝜇ᵢ's are 𝑛 last elements of `u`, and additional 𝑛 equations read
    𝐽𝑀ᵢ = 2∫d𝑥 𝑢ᵢ𝑣ᵢ
If the total number of atoms is fixed, then there is a single 𝜇, so that 𝑛 last elements of `v` are indentical, and 𝑛 last elements of `u` also.
Equations (𝐽𝑣)ᵢ are the same, while the additional 𝑛 (identical) equations read
    𝐽𝑀ᵢ = 2∑ᵢ∫d𝑥 𝑢ᵢ𝑣ᵢ
Used for finding the steady state with nonlinear solve.
"""
function jvp_gpe_real_xspace!(Jv, v, u, params)
    H, g, μs_or_Ns, B, nc, vₚ_buff, vₚ_buff2, u², u²_sum, uⱼvⱼ, ft_forward, ft_backward = params
    # make `μ` point to the chemical potentials: those contained in `μs_or_Ns` if μ's are fixed, or last elements of `u` otherwise
    μs_arefixed = length(u) == B*nc # is 𝜇's are not fixed, then `length(u)` exceeds `B*nc` because `u` then also contains the 𝜇's
    μ = μs_arefixed ? μs_or_Ns : @view u[end-nc+1:end] # if total number of atoms is fixed, then these elements will all be the same
    ### Linear part
    # transform `v` to p-space, write into `vₚ_buff`
    @views for i in 1:nc
        window = (i-1)B+1:i*B
        transform!(ft_forward, v[window])
        fft_to_vector!(vₚ_buff[window], ft_forward)
    end
    mul!(vₚ_buff2, H, vₚ_buff)
    # transform `vₚ_buff2` back to x-space, write into `Jv`
    @views for i in 1:nc
        window = (i-1)B+1:i*B
        transform!(ft_backward, vₚ_buff2[window])
        fft_to_vector!(Jv[window], ft_backward; makereal=true)
    end
    ### Nonlinear part
    # pre-calculate products `uⱼvⱼ` and `uⱼ²`
    @views for j in 1:nc
        uⱼ = u[(j-1)B+1:j*B]
        vⱼ = v[(j-1)B+1:j*B]
        @turbo @. uⱼvⱼ[j] = uⱼ * vⱼ
        @turbo @. u²[j] = uⱼ^2
    end
    # add to `Jv` all nonlinear terms
    @views for i in 1:nc
        vᵢ = v[(i-1)B+1:i*B]
        uᵢ = u[(i-1)B+1:i*B]
        Jvᵢ = Jv[(i-1)B+1:i*B]
        # add 2𝑢ᵢ∑ⱼ𝑔ᵢⱼ𝑢ⱼ𝑣ⱼ to `Jvᵢ`
        j′ = i == 1 ? 2 : 1 # determine the first allowed index of the sum (1 by default, but 2 if i is 1)
        @turbo @. u²_sum = g[i,j′] * uⱼvⱼ[j′] # treat the first term of the sum separately to initialise `u²_sum`
        for j in j′+1:nc # add the remaining terms
            (j == i || g[i,j] == 0) && continue
            @turbo @. u²_sum += g[i,j] * uⱼvⱼ[j]
        end
        !μs_arefixed && (@. u²_sum -= v[end-nc+i]/2) # additional term if μ's are not fixed; divide by 2 to compensate the overall factor in the next line
        @turbo @. Jvᵢ += 2uᵢ * u²_sum
        # add (3𝑔ᵢᵢ𝑢ᵢ² + ∑ⱼ𝑔ᵢⱼ𝑢ⱼ² - 𝜇)𝑣ᵢ to `Jvᵢ`
        @turbo @. u²_sum = 3g[i,i] * u²[i]
        for j in 1:nc
            (j == i || g[i,j] == 0) && continue
            @turbo @. u²_sum += g[i,j] * u²[j]
        end
        @turbo @. Jvᵢ += (u²_sum - μ[i]) * vᵢ
    end
    if !μs_arefixed # then update last `nc` elements of `Jv` corresponding to the chemical potentials. In this case, `μs_or_Ns` contains 𝑁ᵢ's.
        dx = ft_forward.xs[2] - ft_forward.xs[1]
        if μs_or_Ns isa Number # then only total number of atoms is fixed
            uᵢvᵢ_sum = zero(μs_or_Ns) # for storing the sum 2∑ᵢ∫𝑢ᵢvᵢd𝑥
            for i in 1:nc
                uᵢvᵢ_sum += 2sum(uⱼvⱼ[i])*dx
                ft_forward.basis == :cos && (uᵢvᵢ_sum -= (uⱼvⱼ[i][end] + uⱼvⱼ[i][1])*dx) # no division by 2 because of the overall factor in the line above
            end
            Jv[end-nc+1:end] .= uᵢvᵢ_sum # put the sum into place; we have `nc` identical elements to keep the general structure
        else # numbers of atoms in each compoenent are fixed
            for i in 1:nc
                Jv[end-nc+i] = 2sum(uⱼvⱼ[i])*dx
                ft_forward.basis == :cos && (Jv[end-nc+i] -= (uⱼvⱼ[i][end] + uⱼvⱼ[i][1])*dx) # no division by 2 because of the overall factor in the line above
            end
        end
    end
    return
end

############ p-space approach
# Stationary states calculation using NewtonRaphson and BdG fully in momentum space.
# Not superior to x-space and much more cumbersome: 
#   * cis case is complex in p-space for real wave function;
#   * calculation of derivatives in the Jacobian, with respect to re and im part is super cumbersome for the nonlinear term;
#   * constructing the BdG matrix requires double the number of harmonics
# All of the following is subject to removal. Not fully tested, except `bdg_spectrum_pspace`, which is correct and included in the test suite.

"""
Find the stationary state of the Gross-Pitaevskii equation (with nonlinearity matrix `g`) starting from the trial function `ψ₀`.
`ψ₀` can be:
    * a vector of x-space analytic functions (one for each component)
    * a vector of vectors (one for each component) representing discretised x-space functions
    * a vector representing discretised p-space functions, all lumped together 
`solver` is a solver from NonlinearSolvers.jl. We do not construct a concrete Jacobian but rather declare its action on a vector. Autodifferentiation will fail because it doesn't work with FFT that we are using.
Therefore, when passing the solver, always turn off concrete Jacobian and set linear solving to an iterative method.
Return the NonlinearSolution object. 
"""
function find_stationary_pspace(xh::XSpaceHamiltonian{Storage, R, T}, ψ₀::Union{AbstractVector{<:Function}, AbstractVector{<:AbstractVector}, AbstractVector{<:Number}}, μ::R, g::AbstractMatrix{R}=zeros(R, xh.nc, xh.nc);
                 ψ₀_iseven::AbstractVector{Bool}=falses(length(ψ₀)), solver=NLS.NewtonRaphson(;concrete_jac=false, linsolve=NLS.KrylovJL_GMRES())) where {Storage, R, T}
    (;xlims, L, M, basis, nc) = xh
    D = length(xlims)
    # size of each Hamiltonian block
    B = basis == :cis ? (2M+1)^D :
        basis == :sin ?      M^D : (M+1)^D

    # determine if equation can be solved using real types. Note that for cis with nonzero `g` it cannot because intermediate FFT's will be yielding complex results
    eq_isreal = T <: Real && !(basis == :cis && !iszero(g)) # below `eq_isreal` might change if initial state is complex

    # prepare the p-space wf
    if ψ₀ isa AbstractVector{<:Number} # `ψ₀` is given in p-space
        ψ₀_isreal = eltype(ψ₀) <: Real
        eq_isreal &= ψ₀_isreal
        ψ₀ₚ = ψ₀_isreal && !eq_isreal ? complex(ψ₀) : ψ₀  # if the passed initial is real but equation is not, then convert; otherwise take as-is
        # In the p-space case we don't normalise ψ₀ₚ. E.g. the user might want to remove one component and propagate the rest. Normalisation can have consequences since equation is nonlinear
    else # `ψ₀` is given in x-space
        if ψ₀ isa AbstractVector{<:Function} # `ψ₀` a vector of analytic functions
            ψ₀_arereal = [ ψ([xlims[i][1] for i in eachindex(xlims)]...) isa Real for ψ in ψ₀ ]
        else # `ψ₀` is a vector of vectors of discretised functions
            ψ₀_arereal = [eltype(ψ) <: Real for ψ in ψ₀]
        end
        eq_isreal &= all(ψ₀_arereal)
        if basis == :cis && iszero(g) # then also check if functions are even 
            eq_isreal &= all(ψ₀_iseven)
        end
        ψ₀ₚ = Vector{eq_isreal ? R : Complex{R}}(undef, nc*B)

        # transform each component's wf and put into ψ₀ₚ
        ft = FourierTransformer(xlims, M; basis, target_real=all(ψ₀_arereal), target_rank=1) # `target_real=false` will allocate a buffer for the imaginary part of the sin/cos-transform if ψ₀ is complex
        for c in 1:nc
            transform!(ft, ψ₀[c])
            ψ₀ₚ_block = @view ψ₀ₚ[(c-1)*B+1:c*B]
            fft_to_vector!(ψ₀ₚ_block, ft; makereal=(ψ₀_iseven[c] && ψ₀_arereal[c]))
        end
        # In the x-space case we normalise because the initial state is likely just some approximate state.
        normalize!(ψ₀ₚ)
    end

    H = T <: Real && !eq_isreal ? complex(xh.H) : xh.H # `xh.H` is real but equation is not, then convert the Hamiltonian to complex. Solving then proceeds faster TODO: figure out why

    # Combine in `G` all fft normalisation factors so that this multiplication can be done just once at each step
    if basis == :cis
        N = 2M + 1 # number of points in each dimension
        dx = L ./ N
        G = g .* prod(@. dx / L^2) # After bfft, resulting `u` must be divided by √𝐿; since we have `u^3`, we must divide by 𝐿√𝐿. Then, after fft the result must be multiplied by Δ𝑥/√𝐿. So Δ𝑥/𝐿² in total.
    elseif basis == :sin
        N = M
        dx = L ./ (N+1)
        G = g .* prod(@. dx / (2L)^2) # Same as for cis but with √(2𝐿) instead of √𝐿
    else # basis == :cos
        N = M + 1
        dx = L ./ (N-1)
        G = g .* prod(@. dx / (2L)^2) # Same as for cis but with √(2𝐿) instead of √𝐿
    end
    G_input = nc == 1 || allequal(G) ? G[1] : G # the `gpe_*` functions specialise on the equal-g case

    ψ₀ₚ_block = @view ψ₀ₚ[1:B] # for constructing FFT plans and various buffers
    if basis == :cis
        # Both plans will operate on views in `gpe_*`, and they seem to fail for Float32 on Windows/Intel due to alignment issues. The workaround is to pass `flags=FFTW.UNALIGNED`, see https://github.com/JuliaMath/FFTW.jl/issues/67
        bfft_plan! = FFTW.plan_bfft!(ψ₀ₚ_block)
        fft_plan!  = FFTW.plan_fft!(ψ₀ₚ_block)
        if nc == 1 # the 1-component case can be treated more efficiently
            params = (H, μ, G_input, similar(ψ₀ₚ), similar(ψ₀ₚ), bfft_plan!, fft_plan!) # the buffers are complex and hence same length as `ψ₀ₚ`
            nlfunction = NLS.NonlinearFunction(nls_gpe_cis_1comp!; jvp=jvp_gpe_cis_1comp!)
            prob = NLS.NonlinearProblem(nlfunction, complex_to_real(ψ₀ₚ), params)
        else
            buff = [similar(ψ₀ₚ_block) for _ in 1:nc]
            params = (G, B, nc, buff, similar(ψ₀ₚ_block), bfft_plan!, fft_plan!, basis)
            # prob = ODE.SplitODEProblem(H_op, gpe_cis_realsincos!, ψ₀ₚ, tspan, params)
        end
    else # basis == :sin || basis == :cos
        ft_type = basis == :sin ? FFTW.RODFT00 : FFTW.REDFT00
        if eq_isreal # basically, if solving imaginary-time GPE with a real Hamiltonian
            rft_plan  = FFTW.plan_r2r(ψ₀ₚ_block, ft_type)  # will first use this and write the result to the buffer
            rft_plan! = FFTW.plan_r2r!(ψ₀ₚ_block, ft_type) # then will use this in-place on that buffer
            if nc == 1 # the 1-component case can be treated more efficiently
                if basis == :sin
                    params = (H, μ, G_input, similar(ψ₀ₚ), similar(ψ₀ₚ), rft_plan, rft_plan!)
                    nlfunction = NLS.NonlinearFunction(nls_gpe_realsin_1comp!; jvp=jvp_gpe_realsin_1comp!)
                    prob = NLS.NonlinearProblem(nlfunction, ψ₀ₚ, params)
                else # basis == :cos
                    params = (G_input, similar(ψ₀ₚ_block), rft_plan, rft_plan!)
                    # prob = ODE.SplitODEProblem(H_op, gpe_realcos_1comp!, ψ₀ₚ, tspan, params)
                end
            else # arbitrary number of components
                buff = [similar(ψ₀ₚ_block) for _ in 1:nc]
                params = (G, B, nc, buff, similar(ψ₀ₚ_block), rft_plan, rft_plan!, basis)
                # prob = ODE.SplitODEProblem(H_op, gpe_cis_realsincos!, ψ₀ₚ, tspan, params)
            end
        else # solving complex equation
            rft_plan! = FFTW.plan_r2r!(real(ψ₀ₚ_block), ft_type)
            if nc == 1
                params = (G_input, similar(ψ₀ₚ_block, R), similar(ψ₀ₚ_block, R), similar(ψ₀ₚ_block, R), rft_plan!, basis)
                # prob = ODE.SplitODEProblem(H_op, gpe_complexsincos_1comp!, ψ₀ₚ, tspan, params)
            else
                # using vectors of vectors instead of contiguous vectors is ~10% faster and x1000 less memory
                u_re = [similar(ψ₀ₚ_block, R) for _ in 1:nc]
                u_im = [similar(ψ₀ₚ_block, R) for _ in 1:nc]
                u² = [similar(ψ₀ₚ_block, R) for _ in 1:nc]
                params = (G_input, B, nc, u_re, u_im, u², similar(ψ₀ₚ_block, R), rft_plan!, basis)
                # prob = ODE.SplitODEProblem(H_op, gpe_complexsincos!, ψ₀ₚ, tspan, params)
            end
        end
    end

    return NLS.solve(prob, solver)
end

"Return a real vector [real(v); imag(v)]"
function complex_to_real(v::AbstractVector{Complex{R}}) where R <: Real
    n = length(v)
    result = Vector{R}(undef, 2n)
    for i in eachindex(v)
        result[i], result[i+n] = reim(v[i])
    end
    result
end

"""
Update the 𝑢′ vector of the 1-component GPE
    𝑢′ = 𝐻𝑢 + 𝑔|𝑢|²𝑢 - 𝜇𝑢
We use this for finding the steady state with nonlinear solve.
Suitable for the case: wave function is real and basis is sin.
"""
function nls_gpe_realsin_1comp!(du, u, params)
    H, μ, g, u_buff, v_buff, bft_plan, ft_plan! = params # `v_buff` is not neede here but the same set of parameters is also used in `jvp_` function
    mul!(du, H, u)
    @turbo @. du -= u * μ
    # transform `u` to x-space
    mul!(u_buff, bft_plan, u)
    @turbo @. u_buff *= g * abs2(u_buff)
    ft_plan! * u_buff
    du .+= u_buff
    return
end

"""
Describes the action of the Jacobian of the 1-component GPE on a vector 𝑣:
    𝐽𝑣 = 𝐻𝑣 + 3𝑔|𝑢|²𝑣 - 𝜇𝑣
We use this for finding the steady state with nonlinear solve.
Suitable for the case: wave function is real and basis is sin.
"""
function jvp_gpe_realsin_1comp!(Jv, v, u, params)
    H, μ, g, u_buff, v_buff, bft_plan, ft_plan! = params
    mul!(Jv, H, v)
    @turbo @. Jv -= μ * v
    # transform `v` and `u` to x-space
    mul!(u_buff, bft_plan, u)
    mul!(v_buff, bft_plan, v)
    @turbo @. v_buff *= 3g * u_buff^2
    ft_plan! * v_buff
    @turbo @. Jv += v_buff
    return
end

"""
Update the 𝑢′ vector of the 1-component GPE
    𝑢′ = 𝐻𝑢 + 𝑔|𝑢|²𝑢 - 𝜇𝑢
We use this for finding the steady state with nonlinear solve.
Suitable for the case: basis is cis.
"""
function nls_gpe_cis_1comp!(du, u, params)
    H, μ, g, u_complex, v_complex, bft_plan!, ft_plan! = params
    b = length(u) ÷ 2
    # For real and imaginary parts, calculate 𝐻𝑢 - 𝜇𝑢 and write into appropriate halves of `du`
    @views for window in (1:b, b+1:2b)
        mul!(du[window], H, u[window]) # `H` is assumed real
        @. du[window] -= μ * u[window] # all quantities are real but @turbo cannot be applied for views
    end
    # assemble complex `u`
    @views @. u_complex = Complex(u[1:b], u[b+1:2b])
    # apply nonlinearity
    bft_plan! * u_complex
    @. u_complex *= g * abs2(u_complex)
    ft_plan! * u_complex
    # add re and im parts of the nonlinear term to the final `du`
    for i in eachindex(u_complex)
        u_re, u_im = reim(u_complex[i])
        du[i] += u_re
        du[i+b] += u_im
    end
    return
end

"""
Describes the action of the Jacobian of the 1-component GPE on a vector 𝑣:
    𝐽𝑣 = 𝐻𝑣 + 3𝑔|𝑢|²𝑣 - 𝜇𝑣
We use this for finding the steady state with nonlinear solve.
Suitable for the case: wave function is real and basis is sin.
"""
function jvp_gpe_cis_1comp!(Jv, v, u, params)
    H, μ, g, u_complex, v_complex, bft_plan!, ft_plan! = params
    b = length(u) ÷ 2
    # For real and imaginary parts, calculate 𝐻𝑣 - 𝜇𝑣 and write into appropriate halves of `Jv`
    @views for window in (1:b, b+1:2b)
        mul!(Jv[window], H, v[window]) # `H` is assumed real
        @. Jv[window] -= μ * v[window]
    end
    # assemble complex `u` and `v`
    @views @. u_complex = Complex(u[1:b], u[b+1:2b])
    @views @. v_complex = Complex(v[1:b], v[b+1:2b])
    # transform `v` and `u` to x-space
    bft_plan! * u_complex
    bft_plan! * v_complex
    @. v_complex *= 3g * abs2(u_complex)
    ft_plan! * v_complex
    # add re and im parts of the nonlinear term to the final `Jv`
    for i in eachindex(v_complex)
        v_re, v_im = reim(v_complex[i])
        Jv[i] += v_re
        Jv[i+b] += v_im
    end
    return
end

"""
Compute BdG stability spectrum and eigenfunctions for an x-space state `ψ` (1-component case).
If `nev > 0`, calculate only `nev` eigenvalues of of type `whichvals` (`:LI` = largest imaginary by default).
`xh` must contain half the number of harmonics of `ψ` (because having N points in `ψ` we can only construct a p-space operator of size N/2).
`ψ` can be a vector or a N×1 matrix (where N is the number of x points).
"""
function bdg_spectrum_pspace(xh::XSpaceHamiltonian{Storage, R}, ψ::AbstractVecOrMat{<:Union{R, Complex{R}}}, g::AbstractFloat, μ::AbstractFloat; ψ_iseven=false, nev::Integer=0, whichvals::Symbol=:LI, verbose::Bool=false) where {Storage, R}
    (;xlims, M, basis) = xh
    # transform `ψ2` to p-space
    ψ_isreal = eltype(ψ) <: Real
    ψ2 = g .* ψ.^2
    ft = FourierTransformer(xlims, M; basis, target_real=ψ_isreal, target_rank=2, isforward=true) # the constructed matrix will correspond to `M`
    transform!(ft, ψ2)
    v2 = fft_to_matrix(ft; makereal=(ψ_iseven && ψ_isreal))
    if ψ_isreal
        vconj2 = v2
        vabs2 = 2 .* v2
    else
        # transform `conj(ψ2)` to p-space
        transform!(ft, conj(ψ2))
        vconj2 = fft_to_matrix(ft)
        # transform `ψabs2` to p-space
        ψabs2 = abs2.(ψ)
        ft = FourierTransformer(xlims, M; basis, target_real=true, target_rank=2, isforward=true)
        transform!(ft, ψabs2)
        vabs2 = fft_to_matrix(ft)
    end
    # construct the matrix
    A11 = xh.H - μ*LA.I + vabs2
    A = [A11     v2
         -vconj2 -A11]
    if nev == 0
        vals, vecs = eigen(A)
    else
        ps, info = partialschur(A; nev, which=whichvals, restarts=200)
        verbose && @show info
        vals, vecs = partialeigen(ps)
    end
    return vals, vecs
end

"""
Compute BdG stability spectrum and eigenfunctions for an x-space state `ψ` (2-component case).
If `nev > 0`, calculate only `nev` eigenvalues of smallest magnitude.
`xh` must contain half the number of harmonics of `ψ`.
`ψ` must be a N×nc matrix, where `N` is the number of x points and `nc` is the number of components.
The chemical potential `μ` can be passed as a vector, or a number if it is the same for all components.
`q` is the quasimomentum vector [qx, qy, …], zero by default.
Note that the off-diagonal blocks of `xh.H` are not taken into account at all (because one needs to figure out conjugation).
"""
function bdg_spectrum_pspace(xh::XSpaceHamiltonian{Storage, R}, ψ::AbstractMatrix{<:Union{R, Complex{R}}}, g::AbstractMatrix{<:AbstractFloat}, μ::Union{R, AbstractVector{<:R}}, q=zeros(R, length(xh.xlimits));
                             nev::Integer=0, verbose::Bool=false) where {Storage, R}
    (;xlims, M, basis, H, nc) = xh
    μs = μ isa R ? fill(μ, xh.nc) : μ # if only one μ is passed, then construct a vector of same values
    ψ_isreal = eltype(ψ) <: Real
    ft = FourierTransformer(xlims, M; basis, target_real=ψ_isreal, target_rank=2, isforward=true) # the constructed matrix will correspond to `2M` -- internally it will use twice because target_rank=2
    transform!(ft, ψ[:, 1].^2)
    v₁² = fft_to_matrix(ft)
    transform!(ft, ψ[:, 2].^2)
    v₂² = fft_to_matrix(ft)
    transform!(ft, ψ[:, 1].*ψ[:, 2])
    v₁v₂ = fft_to_matrix(ft)
    if ψ_isreal
        v₁⁺² = v₁² # "+" means conjugate
        V₁²  = v₁² # uppercase means modulus
        v₂⁺² = v₂²  
        V₂²  = v₂²
        v₁⁺v₂⁺ = v₁v₂
        v₁v₂⁺ = v₁v₂
        v₁⁺v₂ = v₁v₂
    else
        transform!(ft, conj(ψ[:, 1]).^2)
        v₁⁺² = fft_to_matrix(ft)
        transform!(ft, conj(ψ[:, 2]).^2)
        v₂⁺² = fft_to_matrix(ft)
        
        ft_real = FourierTransformer(xlims, M; basis, target_real=true, target_rank=2, isforward=true)
        transform!(ft_real, abs2.(ψ[:, 1]))
        V₁² = fft_to_matrix(ft_real)
        transform!(ft_real, abs2.(ψ[:, 2]))
        V₂² = fft_to_matrix(ft_real)

        transform!(ft, conj.(ψ[:, 1]) .* conj.(ψ[:, 2]))
        v₁⁺v₂⁺ = fft_to_matrix(ft)
        transform!(ft, ψ[:, 1] .* conj.(ψ[:, 2]))
        v₁v₂⁺ = fft_to_matrix(ft)
        transform!(ft, conj.(ψ[:, 1]) .* ψ[:, 2])
        v₁⁺v₂ = fft_to_matrix(ft)
    end
    B = size(v₁², 1) # our usual blocksize -- number of points corresponding to each (of the two) components -- of xh_half
    A = Matrix{eltype(v₁²)}(undef, 4B, 4B)
    block(a, b) = CartesianIndices(((a-1)B+1:a*B, (b-1)B+1:b*B))

    # from the diagonal of each diagonal block of `H`, extract (𝑈ᵢᵢ)₀ (the 0th harmonic of 𝑈ᵢᵢ) plus decay -iΓ/2
    U_diags = [H[(c-1)B + B÷2+1, (c-1)B + B÷2+1] for c in 1:nc] # generally, `Hᵢᵢ = -Δᵢᵢ + Uᵢᵢ - iΓ/2`, but Δᵢᵢ = 0 for the central element of the diagonal

    copyto!(A, block(1, 1), H, block(1, 1))
    p² = make_p²(xh.L, xh.M, xh.δ, :cis, q) |> parent
    A₁₁ = @view A[block(1, 1)]
    A₁₁[diagind(A₁₁)] .= p² .+ U_diags[1] .- μs[1]
    @. A₁₁ += 2g[1,1]V₁² + g[1,2]V₂²

    @. A[block(1, 2)] = g[1,1]v₁²
    @. A[block(1, 3)] = g[1,2]v₁v₂⁺
    @. A[block(1, 4)] = g[1,2]v₁v₂
    @. A[block(2, 1)] = -g[1,1]v₁⁺²
    @. A[block(2, 2)] = -A₁₁ # assumes real `xh.H`
    @. A[block(2, 3)] = -g[1,2]v₁⁺v₂⁺
    @. A[block(2, 4)] = -g[1,2]v₁⁺v₂
    @. A[block(3, 1)] = g[2,1]v₁⁺v₂
    @. A[block(3, 2)] = g[2,1]v₁v₂

    copyto!(A, block(3, 3), H, block(2, 2))
    A₃₃ = @view A[block(3, 3)]
    A₃₃[diagind(A₃₃)] .= p² .+ U_diags[2] .- μs[2]
    @. A₃₃ += 2g[2,2]V₂² + g[2,1]V₁²

    @. A[block(3, 4)] = g[2,2]v₂²
    @. A[block(4, 1)] = -g[2,1]v₁⁺v₂⁺
    @. A[block(4, 2)] = -g[2,1]v₁v₂⁺
    @. A[block(4, 3)] = -g[2,2]v₂⁺²
    @. A[block(4, 4)] = -A₃₃ # assumes real `xh.H`

    if nev == 0
        vals, vecs = eigen(A)
    else
        prob = LS.LinearProblem(A, similar(A, size(A, 1)))
        linsolve = LS.init(prob, LS.LUFactorization())
        linmap = LinSolveLinMap{Complex{R}, typeof(linsolve)}(linsolve, size(A))
        ps, info = partialschur(linmap; nev, which=:LM)
        verbose && @show info
        vals, vecs = partialeigen(ps)
        return inv.(vals), vecs
    end
    return vals, vecs
end