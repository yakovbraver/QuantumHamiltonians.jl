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
                   solver=(iszero(g) ? DE.LinearExponential() : itime ? DE.LawsonEuler() : DE.ETDRK4()), nsaves::Integer=0) where {Storage, R}
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
                   solver=(iszero(g) ? DE.LinearExponential() : itime ? DE.LawsonEuler() : DE.ETDRK4()), nsaves::Integer=0) where {Storage, R, T}
    (;xlims, L, M, basis, nc) = xh
    D = length(xlims)
    # size of each Hamiltonian block
    B = basis == :cis ? (2M+1)^D :
        basis == :sin ?      M^D : (M+1)^D

    # determine if equation can be solved using real types. Note that for cis with nonzero `g` it cannot because intermediate FFT's will be yielding complex results
    eq_isreal = itime && T <: Real && !(basis == :cis && !iszero(g)) # below `eq_isreal` might change if initial state is complex

    # prepare the p-space wf
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
        normalize!(ψ₀ₚ)
    end

    # initialise the Hamiltonian and coupling matrix `G` with the appropriate sign and `im` factor
    if itime # propagation in imaginary time: equation is real if `xh.H` and `ψ₀ₚ` are real
        H = T <: Real && !eq_isreal ? -complex(xh.H) : -xh.H # `xh.H` is real but equation is not, then convert the Hamiltonian to complex. Solving then proceeds faster TODO: figure out why
        G = -g
    else # propagation in real time: equation is always complex
        H = -im * xh.H
        G = -im * g
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

    # initialise the problem
    tspan = (zero(R), T_max)
    # prepare the SciMLOperator based on the Hamiltonian. If `xh` describes a free system (no 𝑈 or 𝐴), then use `Diagonal`. Then matrix exponential is trivial, leading to immense speed-up
    H_op = all(isnothing.(xh.𝑈)) && all(isnothing.(xh.𝐴)) ? SciMLOperators.MatrixOperator(Diagonal(H)) : SciMLOperators.MatrixOperator(H)

    if iszero(g) # nonlinearity absent
        prob = DE.ODEProblem(H_op, ψ₀ₚ, tspan)
    else # nonlinearity present
        ψ₀ₚ_block = @view ψ₀ₚ[1:B] # for constructing FFT plans and various buffers
        if basis == :cis
            # Both plans will operate on views in `gpe_cis_*`, and they seem to fail for Float32 on Windows/Intel due to alignment issues. The workaround is to pass `flags=FFTW.UNALIGNED`, see https://github.com/JuliaMath/FFTW.jl/issues/67
            bfft_plan = FFTW.plan_bfft(ψ₀ₚ_block) # will first use this and write the result into `du`
            fft_plan! = FFTW.plan_fft!(ψ₀ₚ_block) # then will use this in-place on `du`
            if nc == 1 # the 1-component case can be treated more efficiently
                params = (G[1], bfft_plan, fft_plan!)
                prob = DE.SplitODEProblem(H_op, gpe_cis_realsin_1comp!, ψ₀ₚ, tspan, params)
            else
                buff = [similar(ψ₀ₚ_block) for _ in 1:nc]
                params = (G, B, nc, buff, similar(ψ₀ₚ_block), bfft_plan, fft_plan!, basis)
                prob = DE.SplitODEProblem(H_op, gpe_cis_realsincos!, ψ₀ₚ, tspan, params)
            end
        else # basis == :sin || basis == :cos
            ft_type = basis == :sin ? FFTW.RODFT00 : FFTW.REDFT00
            if eq_isreal # basically, if solving imaginary-time GPE with a real Hamiltonian
                rft_plan  = FFTW.plan_r2r(ψ₀ₚ_block, ft_type)  # will first use this and write the result to the buffer
                rft_plan! = FFTW.plan_r2r!(ψ₀ₚ_block, ft_type) # then will use this in-place on that buffer
                if nc == 1 # the 1-component case can be treated more efficiently
                    if basis == :sin
                        params = (G[1], rft_plan, rft_plan!)
                        prob = DE.SplitODEProblem(H_op, gpe_cis_realsin_1comp!, ψ₀ₚ, tspan, params)
                    else # basis == :cos
                        params = (G[1], similar(ψ₀ₚ_block), rft_plan, rft_plan!)
                        prob = DE.SplitODEProblem(H_op, gpe_realcos_1comp!, ψ₀ₚ, tspan, params)
                    end
                else # arbitrary number of components
                    buff = [similar(ψ₀ₚ_block) for _ in 1:nc]
                    params = (G, B, nc, buff, similar(ψ₀ₚ_block), rft_plan, rft_plan!, basis)
                    prob = DE.SplitODEProblem(H_op, gpe_cis_realsincos!, ψ₀ₚ, tspan, params)
                end
            else # solving complex equation
                rft_plan! = FFTW.plan_r2r!(real(ψ₀ₚ_block), ft_type)
                params = (G[1], similar(ψ₀ₚ_block, R), similar(ψ₀ₚ_block, R), similar(ψ₀ₚ_block, R), rft_plan!, basis)
                prob = DE.SplitODEProblem(H_op, gpe_sincos_complex_1comp!, ψ₀ₚ, tspan, params)
            end
        end
    end

    if itime
        # prepare the callback that remormalises wf at every step
        condition = Returns(true) # condition is checked at the end of each time step; we want this to be always true
        affect!(integrator) = normalize!(integrator.u)
        cb = DE.DiscreteCallback(condition, affect!) # will save every step before and after the callback (`save_positions=(true, true)`); docs say this is mandatory when change of `u` is discontinuous
        sol = DE.solve(prob, solver; callback=cb, save_everystep=false, save_start=true, dt)
        normalize!(sol.u[end]) # the final step is saved only before the callback, so normalise manually
        return sol
    else
        # when `saveat` is set, saving happens at points `tspan[1]:saveat:tspan[2]`
        saveat = nsaves == 0 ? T_max+1 : (tspan[2] - tspan[1]) / nsaves
        return DE.solve(prob, solver; save_everystep=false, save_start=true, dt, saveat)
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
    du .*= g .* abs2.(du)
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
    # for each `i`th component, transform 𝑢ᵢ to x-space and write into `du`. Also, calculate |𝑢ᵢ|²
    for i in 1:nc
        window = (i-1)B+1:i*B
        du_i = @view du[window] # view into the relevant component
        if basis == :cos # then must normalise 0th and last harmonics before transforming; will use `u²_sum` as a buffer
            copyto!(u²_sum, 1, u, (i-1)B+1, B) # copy `B` elements of `u`, starting from `(i-1)B+1`th into `u²_sum`, starting from index 1
            u²_sum[1] *= √2; u²_sum[end] *= √2
            mul!(du_i, bfft_plan, u²_sum) # transform `u²_sum` and write into `du`
        else
            mul!(du_i, bfft_plan, @view(u[window])) # transform `u` and write into `du`
        end
        @. u²[i] = abs2(du_i)
    end
    # for each `i`th component, calculate the sum ∑ⱼ 𝑔ᵢⱼ|𝑢ⱼ|² an multiply by 𝑢ᵢ, stored in `du`
    for i in 1:nc
        @. u²_sum = g[i, 1] * u²[1]
        for j in 2:nc
            g[i, j] == 0 && continue
            @. u²_sum += g[i, j] * u²[j]
        end
        du[(i-1)B+1:i*B] .*= u²_sum
    end
    # transform `du` to p-space in-place
    for i in 1:nc
        fft_plan! * @view(du[(i-1)B+1:i*B])
        basis == :cos && (du[(i-1)B+1] /= √2; du[i*B] /= √2)
    end
    return
end

"""
Update the 𝑢′ vector of the nonlinear part of the 1-component GPE
    𝑢′ = 𝑔|𝑢|²𝑢
The 𝑔 must contain `im` (for real-time propagation) and the proper sign.
Suitable for the case: basis is cos and equation is real.
"""
function gpe_realcos_1comp!(du, u, params, t)
    g, u_buff, rft_plan, rft_plan! = params
    copy!(u_buff, u) # because of the next step; cannot do it for `u` (not allowed to change `u`)
    u_buff[1] *= √2; u_buff[end] *= √2
    mul!(du, rft_plan, u_buff)
    du .*= g .* du.^2
    rft_plan! * du
    du[1] /= √2; du[end] /= √2
    return
end

"Update the 𝑢′ matrix of the complex GPE in the sin/cos basis."
function gpe_sincos_complex_1comp!(du, u, params, t)
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
    @. u² = u_re^2 + u_im^2
    # calculate 𝑢(𝑥)|𝑢(𝑥)|²
    @. u_re *= u²
    @. u_im *= u²
    # transform to p-space
    rft_plan! * u_re
    rft_plan! * u_im
    # add re and im
    @. du = g * (u_re + im * u_im)
    basis == :cos && (du[1] /= √2; du[end] /= √2)
    return
end

"Return mean energy (per particle) and chemical potential for a p-space state `v`."
function get_ε_μ(xh::XSpaceHamiltonian{Storage,R}, v, g::AbstractMatrix{<:Number}=zeros(typeof(xh.δ), xh.nc, xh.nc)) where {Storage,R}
    (;xlims, M, nc, basis) = xh
    ε = dot(v, xh.H, v)
    μ = ε
    if !iszero(g)
        B = length(v) ÷ nc  
        v_isreal = eltype(v) <: Real
        ft = FourierTransformer(xlims, M; basis, target_real=v_isreal, target_rank=1, forward=false)
        dV = prod(ft.xs[2, i] - ft.xs[1, i] for i in axes(ft.xs, 2)) # volume element
        U = ε # initialise integral
        # create an array of arrays holding squared x-space wfs |𝜓(𝑥)|² for each component
        ψ² = map(1:nc) do c
            v_input = v[(c-1)B+1:c*B] # a copy is needed only in the cos case, but we always make it
            basis == :cos && (v_input[1] *= √2; v_input[end] *= √2) # proper normalisation of the 0th and last harmonics
            transform!(ft, v_input)
            ψ = fft_to_vector(ft)
            basis == :cos && (ψ[1] *= √2; ψ[end] *= √2)  # undo what is done in `fft_to_vector!` (that assumes p-space while we actually got back to x-space)
            basis == :cis && (ψ = FFTW.ifftshift(ψ)) # undo what is done in `fft_to_vector`
            ψ .= abs2.(ψ)
            return ψ
        end
        ψ²_sum = similar(ψ²[1])
        # for each `i`th component: calculate the sum ∑ⱼ 𝑔ᵢⱼ|𝜓ⱼ|², then multiply by |𝜓ᵢ|², then integrate
        for i in 1:nc
            ψ²_sum .= 0
            for j in 1:nc
                g[i, j] == 0 && continue
                @. ψ²_sum += g[i, j] * ψ²[j]
            end
            ψ²_sum .*= ψ²[i]
            if basis == :cos # 𝑥 is discretised with both endpoints included
                U = (sum(ψ²_sum) - ψ²_sum[end]) * dV # so final point is redundant (assuming rectangle rule)
            else
                U = sum(ψ²_sum) * dV # for sin, endpoints are not included but are zero, so this is equivalent to the trapezoid rule. For cis, rectangle rule is more appropriate because there is no boundary
            end
            μ += U
            ε += U / 2
        end
    end
    return ε, μ
end