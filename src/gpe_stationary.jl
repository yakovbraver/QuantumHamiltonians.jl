"A version of `find_stationary` accepting `ψ₀` as a vector of analytic functions (one for each component)."
function find_stationary(qh::Union{PSpaceHamiltonian{Storage, R, T}, XSpaceHamiltonian{R, T}}, ψ₀::AbstractVector{<:Function},
                         g::AbstractMatrix{R}, μ::Union{R, AbstractVector{R}}, natoms::Union{Nothing, R, AbstractVector{R}}=nothing;
                         solver=NLS.NewtonRaphson(;linsolve=LS.KrylovJL_GMRES()), kwargs...) where {Storage, R, T}
    (;xlims, B, nc, ft) = qh

    # determine if equation is real
    ψ₀_arereal = all( ψ([xlims[i][1] for i in eachindex(xlims)]...) isa Real for ψ in ψ₀ )
    eq_isreal = ψ₀_arereal && all(isnothing.(qh.𝐴)) # equations are real if Hamiltonian and wfs are real
    
    # Prepare the input wf `ψ_input`. By default, its length is `nc*B`, but if `natoms` is passed then we need additional `nc` elements to represent the 𝜇s that are being optimised.
    # Even if only total 𝑁 is fixed (and hence there is only one 𝜇 to be optimised), we still add `nc` elements to keep the general structure
    if eq_isreal
        # sample each function in ψ₀ at points `ft.xs`
        ψ_input = Vector{R}(undef, nc*(B + !isnothing(natoms)))
        for c in 1:nc
            @views sample!(ψ_input[(c-1)B+1:c*B], ψ₀[c], ft.xs)
        end
    else # equations in x-space are complex
        ψ_input = Vector{R}(undef, nc*(2B+!isnothing(natoms)))
        for c in 1:nc
            @views sample!(ψ_input[(c-1)*2B+1:(c-1)*2B+B], ψ_input[(c-1)*2B+B+1:c*2B], ft.xs)
        end
    end

    find_stationary(qh, ψ_input, g, μ, natoms; solver, kwargs...)
end

"A version of `find_stationary` accepting `ψ₀` as a vector of etiher D-dimensional arrays or flattened vectors, one for each component, representing discretised x-space functions."
function find_stationary(qh::Union{PSpaceHamiltonian{Storage, R, T}, XSpaceHamiltonian{R, T}}, ψ₀::AbstractVector{<:AbstractArray},
                         g::AbstractMatrix{R}, μ::Union{R, AbstractVector{R}}, natoms::Union{Nothing, R, AbstractVector{R}}=nothing;
                         solver=NLS.NewtonRaphson(;linsolve=LS.KrylovJL_GMRES()), kwargs...) where {Storage, R, T}
    (;B, nc) = qh

    # determine if equation is real
    ψ₀_arereal = all(eltype(ψ) <: Real for ψ in ψ₀)
    eq_isreal = ψ₀_arereal && all(isnothing.(qh.𝐴)) # equations are real if Hamiltonian and wfs are real
    
    # Prepare the input wf `ψ_input`. By default, its length is `nc*B`, but if `natoms` is passed then we need additional `nc` elements to represent the 𝜇s that are being optimised.
    # Even if only total 𝑁 is fixed (and hence there is only one 𝜇 to be optimised), we still add `nc` elements to keep the general structure
    ψ_input = Vector{eq_isreal ? R : Complex{R}}(undef, nc*(B+!isnothing(natoms)))
    for c in 1:nc
        copyto!(ψ_input, (c-1)B+1, ψ₀[c], 1, B) # copy `B` (= all) elements of `ψ₀[c]`, starting at 1st, to `ψ_input`, starting at element (c-1)B+1. `ψ₀[c]` might be D-dimensional, but `copyto!` automatically flattens it (i.e. treats it as a contiguous vector)
    end

    find_stationary(qh, ψ_input, g, μ, natoms; solver, kwargs...)
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
    1. A vector of x-space analytic functions (one for each component);
    2. A vector of etiher D-dimensional arrays or flattened vectors, one for each component, representing discretised x-space functions.
    3. An x-space flattened vector, such as one obtained during diagonalisation. This is the format using throughout the solving; inputs of types 1 and 2 are converted to this type.
`solver` is a solver from NonlinearSolvers.jl. We do not construct a concrete Jacobian but rather declare its action on a vector. Autodiff will fail because it doesn't work with FFT, which we are using.
Therefore, when passing the solver, always turn off concrete Jacobian and/or set linear solving to an iterative method.
Default solver is `NewtonRaphson(;linsolve=KrylovJL_GMRES())`.
You can also try using BICSTAB and/or Eisenstat-Walker forcing as in `NewtonRaphson(;linsolve=KrylovJL_BICGSTAB(), forcing=EisenstatWalkerForcing2())`.
Forcing might fail to converge, but it accelerates solving since otherwise the linear system is solved to the same accuracy as the nonlinear system, which is often (but not always!) reundant.
Default termination mode is `AbsNormSafeBestTerminationMode` with L-inf norm with `abstol` defined in NonlinearSolve.
Any additional keyword arguments will be passed directly to `NonlinearSolve.solve()`.
Return the tuple consisting of the coordinates and the NonlinearSolution object.
"""
function find_stationary(qh::Union{PSpaceHamiltonian{Storage, R, T}, XSpaceHamiltonian{R, T}}, ψ₀::AbstractVector{<:Number},
                         g::AbstractMatrix{R}, μ::Union{R, AbstractVector{R}}, natoms::Union{Nothing, R, AbstractVector{<:R}}=nothing;
                         solver=NLS.NewtonRaphson(;linsolve=LS.KrylovJL_GMRES()), kwargs...) where {Storage, R, T}
    (;B, nc, ft) = qh

    # make `μs_or_Ns` point to the right thing and prepare input state
    if isnothing(natoms) # = numbers of atoms are not fixed, but chemical potentials are
        μs_or_Ns = μ # so just pass the fixed chemical potentials

        ψ_input = ψ₀ # make a reference
    else  # total number of atoms or number of atoms in each component is fixed
        μs_or_Ns = natoms # pass the fixed numbers of atoms (a single number if total 𝑁 is fixed or a vector otherwise)

        if length(ψ₀) == nc*B + nc # `ψ₀` already has `nc` extra elements for 𝜇s
            ψ_input = ψ₀ # then just make a reference
        else # `ψ₀` does not have the required `nc` extra elements for 𝜇s
            ψ_input = similar(ψ₀, nc*B + nc) # make an array of required length
            copyto!(ψ_input, ψ₀)
        end
        # now can set final `nc` elements
        ψ_input[end-nc+1:end] .= μ # use the passed `μ` as the initial guess (a single number if total 𝑁 is fixed or a vector otherwise; broadcast handles both cases)
    end

    if nc == 1 # the 1-component case can be treated more efficiently
        params = (qh, g[1], μs_or_Ns)
        nlfunction = NLS.NonlinearFunction(nls_gpe_real_1comp!; jvp=jvp_gpe_real_1comp!)
        prob = NLS.NonlinearProblem(nlfunction, ψ_input, params)
    else
        # initialise the buffers for holding all double products
        u²_sum = Vector{R}(undef, B)
        u² = [similar(u²_sum) for _ in 1:nc]
        uⱼvⱼ = [similar(u²_sum) for _ in 1:nc]
        params = (qh, g, μs_or_Ns, nc, u², u²_sum, uⱼvⱼ)
        nlfunction = NLS.NonlinearFunction(nls_gpe_real!; jvp=jvp_gpe_real!)
        prob = NLS.NonlinearProblem(nlfunction, ψ_input, params) # use specialisation `NonlinearProblem{true, SciMLBase.FullSpecialize}` for production!
    end

    # we will pass on user's kwargs to NLS.solve, but we override some of NLS's defaults. User's kwargs will in turn override ours.
    finalkwargs = (;verbose=false, termination_condition=NLS.AbsNormSafeBestTerminationMode(Base.Fix1(maximum, abs)), kwargs...)
    
    return ft.xs, NLS.solve(prob, solver; finalkwargs...)
end

"""
Update the x-space 𝑢′ vector of the 1-component GPE
    𝑢′ = 𝐻𝑢 + 𝑔𝑢²𝑢 - 𝜇𝑢

If the number of atoms 𝑁 is fixed, then 𝜇 is unknown and is contained in the last element of `u`. The last element of `du` contains the residual 𝜇′ for the chemical potential.
An additional equation reads
    ∫𝑢²d𝑥 - 𝑁 = 0
which is coded as
    𝜇′ = ∫𝑢²d𝑥 - 𝑁

Used for finding the steady state with nonlinear solve (by solving for 𝑢′ = 𝜇′ = 0).
Suitable for the case when 𝐻, and hence also 𝑢, is real in x-space.
"""
@views function nls_gpe_real_1comp!(du, u, params)
    qh, g, μ_or_N = params
    B = qh.B
    # make `μ` point to the chemical potentials: those contained in `μs_or_Ns` if 𝜇 is fixed, or last element of `u` otherwise
    μ_isfixed = length(u) == B # is 𝜇s are not fixed, then `length(u)` exceeds `B` because `u` then also contains 𝜇
    μ = μ_isfixed ? μ_or_N : u[end] # if 𝜇 is fixed, the `μ_or_N` contains the fixed chemical potential
    mul!(du[1:B], qh, u[1:B])
    # add g and μ terms
    @. du[1:B] += (g * abs2(u[1:B]) - μ) * u[1:B]
    if !μ_isfixed # then update the last element of `du` representing the residual ∫𝑢²d𝑥 - 𝑁. In this case, `μ_or_N` contains 𝑁.
        du[end] = norm²(u[1:B], qh) - μ_or_N
    end
    return
end

"""
Describes the action of the Jacobian of the 1-component GPE on an x-space vector 𝑣:
    𝐽𝑣 = 𝐻𝑣 + 3𝑔𝑢²𝑣 - 𝜇𝑣

If the number of atoms is fixed, then the last element of `v` and `u` is assumed to contain the chemical potential. Then, equation is
    𝐽𝑣 = 𝐻𝑣 + 3𝑔𝑢²𝑣 - 𝜇𝑣 - 𝑀𝑢
where 𝑀 is the last element of `v`, 𝜇 is the last element of `u`, and an additional equation reads
    𝐽𝑀 = 2∫𝑢𝑣d𝑥
Used for finding the steady state with nonlinear solve.
Suitable for the case when 𝐻, and hence also 𝑢 and 𝑣, is real in x-space.
"""
@views function jvp_gpe_real_1comp!(Jv, v, u, params)
    qh, g, μ_or_N = params
    B = qh.B
    # make `μ` point to the chemical potentials: those contained in `μs_or_Ns` if 𝜇 is fixed, or last element of `u` otherwise
    μ_isfixed = length(u) == B # if 𝜇s are not fixed, then `length(u)` exceeds `B` because `u` then also contains 𝜇
    μ = μ_isfixed ? μ_or_N : u[end] # if 𝜇 is fixed, the `μ_or_N` contains the fixed chemical potential
    mul!(Jv[1:B], qh, v[1:B])
    # add g and μ terms
    @. Jv[1:B] += (3g * abs2(u[1:B]) - μ) * v[1:B]
    if !μ_isfixed # then subtract the additional term 𝑀𝑢 and update the last element of `Jv` representing the additional equation. In this case, `μ_or_N` contains 𝑁.
        @. Jv[1:B] -= v[end] * u[1:B] # subtract 𝑀𝑢
        # set 𝐽𝑀 = 2∫𝑢𝑣d𝑥
        Jv[end] = 2inner_prod(u[1:B], v[1:B], qh)
    end
    return
end

"""
Update the x-space 𝑢′ vector of the multi-component GPE
    𝑢′ᵢ = (𝐻𝑢)ᵢ + (∑ⱼ 𝑔ᵢⱼ𝑢ⱼ² - 𝜇ᵢ)𝑢ᵢ

If the numbers of atoms 𝑁ᵢ in each component are fixed, then 𝜇ᵢ are unknown and are contained in 𝑛 last elements of `u`. Last 𝑛 elements of `du` contain the residuals 𝜇ᵢ′ for the chemical potentials.
𝑛 additional equations read
    𝜇ᵢ′ = ∫𝑢ᵢ²d𝑥 - 𝑁

If only the total number of atoms 𝑁 is fixed, then there is a single 𝜇, so that 𝑛 last elements of `u` are indentical, and 𝑛 last elements of `du` also.
𝑛 (identical) additional equations read
    𝜇′ = ∑ᵢ∫𝑢ᵢ²d𝑥 - 𝑁

Used for finding the steady state with nonlinear solve.
Suitable for the case when 𝐻, and hence also 𝑢ᵢ, is real in x-space.
"""
@views function nls_gpe_real!(du, u, params)
    qh, g, μs_or_Ns, nc, u², u²_sum, uⱼvⱼ = params
    B = qh.B
    # make `μ` point to the chemical potentials: those contained in `μs_or_Ns` if 𝜇s are fixed, or last elements of `u` otherwise
    μs_arefixed = length(u) == B*nc # is 𝜇s are not fixed, then `length(u)` exceeds `B*nc` because `u` then also contains the 𝜇s
    μ = μs_arefixed ? μs_or_Ns : u[end-nc+1:end] # if total number of atoms is fixed, then these elements will all be the same
    
    ### Linear part
    mul!(du[1:B*nc], qh, u[1:B*nc])
    
    ### Nonlinear part
    # pre-calculate 𝑢ᵢ² for each component and store in `u²`
    for i in 1:nc
        @turbo @. u²[i] = u[(i-1)B+1:i*B]^2
    end
    # add (∑ⱼ𝑔ᵢⱼ𝑢ⱼ² - 𝜇ᵢ)𝑢ᵢ to `duᵢ`
    for i in 1:nc
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
        if μs_or_Ns isa Number # then only total number of atoms is fixed -- will place (∑ᵢ∫𝑢ᵢ²d𝑥 - 𝑁) into `nc` last elements of `du`
            u²_sum = zero(μs_or_Ns) # for storing the sum ∑ᵢ∫𝑢ᵢ²d𝑥
            for i in 1:nc
                u²_sum += integrate(u²[i], qh)
            end
            du[end-nc+1:end] .= u²_sum - μs_or_Ns # place the sum in the residuals array; we have `nc` identical elements to keep the general structure
        else # numbers of atoms in each component are fixed -- will place (∫𝑢ᵢ²d𝑥 - 𝑁ᵢ) into `nc` last elements of `du` respectively
            for i in 1:nc
                du[end-nc+i] = integrate(u²[i], qh) - μs_or_Ns[i]
            end
        end
    end
    return
end

"""
Describes the action of the Jacobian of the multi-component GPE on an x-space vector 𝑣:
    (𝐽𝑣)ᵢ = (𝐻𝑣)ᵢ + 2𝑢ᵢ∑ⱼ𝑔ᵢⱼ𝑢ⱼ𝑣ⱼ + (3𝑔ᵢᵢ𝑢ᵢ² + ∑ⱼ𝑔ᵢⱼ𝑢ⱼ² - 𝜇ᵢ)𝑣ᵢ   [∑ⱼ excludes 𝑖]

If the numbers of atoms in each component are fixed, then last 𝑛 elements of `v` and `u` are assumed to contain the chemical potentials. Then, equations are
    (𝐽𝑣)ᵢ = (𝐻𝑣)ᵢ + 2𝑢ᵢ∑ⱼ𝑔ᵢⱼ𝑢ⱼ𝑣ⱼ - 𝑀ᵢ𝑢ᵢ + (3𝑔ᵢᵢ𝑢ᵢ² + ∑ⱼ𝑔ᵢⱼ𝑢ⱼ² - 𝜇ᵢ)𝑣ᵢ   [∑ⱼ excludes 𝑖]
where 𝑀ᵢ are 𝑛 last elements of `v`, 𝜇ᵢ are 𝑛 last elements of `u`, and additional 𝑛 equations read
    𝐽𝑀ᵢ = 2∫d𝑥 𝑢ᵢ𝑣ᵢ
If the total number of atoms is fixed, then there is a single 𝜇, so that 𝑛 last elements of `v` are indentical, and 𝑛 last elements of `u` also.
Equations (𝐽𝑣)ᵢ are the same, while the additional 𝑛 (identical) equations read
    𝐽𝑀ᵢ = 2∑ᵢ∫d𝑥 𝑢ᵢ𝑣ᵢ

Used for finding the steady state with nonlinear solve.
Suitable for the case when 𝐻, and hence also 𝑢ᵢ and 𝑣ᵢ, is real in x-space.
"""
@views function jvp_gpe_real!(Jv, v, u, params)
    qh, g, μs_or_Ns, nc, u², u²_sum, uⱼvⱼ = params
    B = qh.B
    # make `μ` point to the chemical potentials: those contained in `μs_or_Ns` if 𝜇s are fixed, or last elements of `u` otherwise
    μs_arefixed = length(u) == B*nc # is 𝜇s are not fixed, then `length(u)` exceeds `B*nc` because `u` then also contains the 𝜇s
    μ = μs_arefixed ? μs_or_Ns : u[end-nc+1:end] # if total number of atoms is fixed, then these elements will all be the same
    
    ### Linear part
    mul!(Jv[1:B*nc], qh, v[1:B*nc])
    
    ### Nonlinear part
    # pre-calculate products `uⱼvⱼ` and `uⱼ²`
    for j in 1:nc
        uⱼ = u[(j-1)B+1:j*B]
        vⱼ = v[(j-1)B+1:j*B]
        @turbo @. uⱼvⱼ[j] = uⱼ * vⱼ
        @turbo @. u²[j] = uⱼ^2
    end
    # add to `Jv` all nonlinear terms
    for i in 1:nc
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
        !μs_arefixed && (@. u²_sum -= v[end-nc+i]/2) # additional term if 𝜇s are not fixed; divide by 2 to compensate the overall factor in the next line
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
        if μs_or_Ns isa Number # then only total number of atoms is fixed -- will place 2∑ᵢ∫𝑢ᵢvᵢd𝑥 into `nc` last elements of `Jv`
            uᵢvᵢ_sum = zero(μs_or_Ns) # for storing the sum ∑ᵢ∫𝑢ᵢvᵢd𝑥
            for i in 1:nc
                uᵢvᵢ_sum += integrate(uⱼvⱼ[i], qh)
            end
            Jv[end-nc+1:end] .= 2uᵢvᵢ_sum # put the sum into place; we have `nc` identical elements to keep the general structure
        else # numbers of atoms in each component are fixed -- will place 2∫𝑢ᵢvᵢd𝑥 into `nc` last elements of `du` respectively
            for i in 1:nc
                Jv[end-nc+i] = 2integrate(uⱼvⱼ[i], qh)
            end
        end
    end
    return
end
