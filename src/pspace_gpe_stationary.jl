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
function find_stationary(xh::PSpaceHamiltonian{Storage, R, T}, ψ₀::Union{AbstractVector{<:Function}, AbstractVector{<:AbstractVector}},
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
    ft = FourierTransformerP(xlims, M; basis, target_real=eq_isreal, target_rank=1) # single transformer supporting both directions
    nx = length(ft.xs) # TODO this assumes 1D, change to size(ft.xs, 1), also in other places
    
    # Prepare the input wf `ψ_input`. By default, its length is `nc*nx`, but if `natoms` is passed then we need additional `nc` elements to represent the μ's that are being optimised.
    # Even if only total 𝑁 is fixed (and hence there is only one 𝜇 to be optimised), we still add `nc` elements to keep the general structure
    if ψ₀ isa AbstractVector{<:Function} # `ψ₀` is a vector of analytic functions: need to sample them
        if eq_isreal
            # sample each function in ψ₀ at points `ft.xs`
            ψ_input = Vector{R}(undef, nc*(nx + !isnothing(natoms)))
            for c in 1:nc
                ψ_input[(c-1)*nx+1:c*nx] .= ψ₀[c].(ft.xs)
            end
        else # equations in x-space are complex
            ψ_input = Vector{R}(undef, nc*(2nx+!isnothing(natoms)))
            for c in 1:nc, ix in 1:nx
                ψ_input[(c-1)*2nx + ix], ψ_input[(c-1)*2nx + nx+ix] = reim(ψ₀[c].(ft.xs[ix]))
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
    if basis == :cis || eq_isreal # in the cis case, the p-space buffer must be complex, so copy `ft.buff` -- it's always complex for cis. If `eq_isreal` and basis is sin/cos, then also copy `ft.buff` -- it's always real for sin/cos
        uₚ_buff  = similar(ft.buff, nc*length(ft.buff))
        uₚ_buff2 = similar(ft.buff, nc*length(ft.buff))
    else # !eq_isreal and basis is sin/cos
        uₚ_buff = similar(ft.buff, Complex{R}, nc*length(ft.buff))
        uₚ_buff2 = similar(ft.buff, Complex{R}, nc*length(ft.buff))
    end

    if isnothing(natoms) # = numbers of atoms are not fixed, but chemical potentials are
        μs_or_Ns = μ # so just pass the fixed chemical potentials
    else # total number of atoms or number of atoms in each component is fixed
        μs_or_Ns = natoms # pass the fixed numbers of atoms (a single number if total 𝑁 is fixed or a vector otherwise)
    end

    B = length(ft.buff)
    if nc == 1 # the 1-component case can be treated more efficiently
        params = (xh.H, g[1], μs_or_Ns, B, uₚ_buff, uₚ_buff2, ft)
        nlfunction = NLS.NonlinearFunction(nls_gpe_real_1comp!; jvp=jvp_gpe_real_1comp!)
        prob = NLS.NonlinearProblem(nlfunction, ψ_input, params)
    else
        # initialise the buffers for holding all double products
        u²_sum = Vector{R}(undef, B)
        u² = [similar(u²_sum) for _ in 1:nc]
        uⱼvⱼ = [similar(u²_sum) for _ in 1:nc]
        params = (xh.H, g, μs_or_Ns, B, nc, uₚ_buff, uₚ_buff2, u², u²_sum, uⱼvⱼ, ft)
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
    ∫𝑢²d𝑥 - 𝑁 = 0     [present if the number of atoms is fixed]
Used for finding the steady state with nonlinear solve.
Suitable for the case when 𝐻, and hence also 𝑢, is real in x-space.
"""
function nls_gpe_real_1comp!(du, u, params)
    H, g, μ_or_N, B, uₚ_buff, uₚ_buff2, ft = params
    # make `μ` point to the chemical potentials: those contained in `μs_or_Ns` if μ's are fixed, or last elements of `u` otherwise
    μ_isfixed = length(u) == B # is 𝜇's are not fixed, then `length(u)` exceeds `B` because `u` then also contains 𝜇
    μ = μ_isfixed ? μ_or_N : u[end] # if 𝜇 is fixed, the `μ_or_N` contains the fixed chemical potential
    # transform `u` to p-space, multiply by `H` and transform back
    transform!(ft, @view u[1:B]; direction=:forward)
    fft_to_state!(uₚ_buff, ft; direction=:forward)
    mul!(uₚ_buff2, H, uₚ_buff)
    transform!(ft, uₚ_buff2; direction=:backward)
    @views fft_to_state!(du[1:B], ft; direction=:backward, makereal=true) # we assume that `u` is real, so the result here must be real: pass `make_real=true` to drop imaginary part in the cis case
    # add g and μ terms
    @views @. du[1:B] += (g * abs2(u[1:B]) - μ) * u[1:B]
    if !μ_isfixed # then update the last element of `du` representing the residual ∫𝑢²d𝑥 - 𝑁. In this case, `μ_or_N` contains 𝑁.
        dx = ft.xs[2] - ft.xs[1]
        @views du[end] = sum(abs2, u[1:B])*dx - μ_or_N
        ft.basis == :cos && (du[end] -= (u[1]^2 + u[B]^2)*dx/2)
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
Suitable for the case when 𝐻, and hence also 𝑢 and 𝑣, is real in x-space.
"""
function jvp_gpe_real_1comp!(Jv, v, u, params)
    H, g, μ_or_N, B, vₚ_buff, vₚ_buff2, ft = params
    # make `μ` point to the chemical potentials: those contained in `μs_or_Ns` if μ's are fixed, or last elements of `u` otherwise
    μ_isfixed = length(u) == B # is 𝜇's are not fixed, then `length(u)` exceeds `B` because `u` then also contains 𝜇
    μ = μ_isfixed ? μ_or_N : u[end] # if 𝜇 is fixed, the `μ_or_N` contains the fixed chemical potential
    # transform `v` to p-space, multiply by `H` and transform back
    transform!(ft, @view v[1:B]; direction=:forward)
    fft_to_state!(vₚ_buff, ft; direction=:forward)
    mul!(vₚ_buff2, H, vₚ_buff)
    transform!(ft, vₚ_buff2; direction=:backward)
    @views fft_to_state!(Jv[1:B], ft; direction=:backward, makereal=true)
    # add g and μ terms
    @views @. Jv[1:B] += (3g * abs2(u[1:B]) - μ) * v[1:B]
    if !μ_isfixed # then subtract the additional term 𝑀𝑢 and update the last element of `Jv` representing the additional equation. In this case, `μ_or_N` contains 𝑁.
        @views @. Jv[1:B] -= v[end] * u[1:B] # subtract 𝑀𝑢
        # set 𝐽𝑀 = 2∫d𝑥 𝑢𝑣
        dx = ft.xs[2] - ft.xs[1]
        @views Jv[end] = 2dot(u[1:B], v[1:B]) * dx
        ft.basis == :cos && (Jv[end] -= (u[1]*v[1] + u[B]*v[B])*dx)
    end
    return
end

"""
Update the x-space 𝑢′ vector of the multi-component GPE
    𝑢′ᵢ = (𝐻𝑢)ᵢ + (∑ⱼ 𝑔ᵢⱼ𝑢ⱼ² - 𝜇ᵢ)𝑢ᵢ
    ∫𝑢ᵢ²d𝑥 - 𝑁ᵢ = 0     [present if the numbers of atoms are fixed]
Used for finding the steady state with nonlinear solve.
Suitable for the case when 𝐻, and hence also 𝑢ᵢ, is real in x-space.
"""
function nls_gpe_real!(du, u, params)
    H, g, μs_or_Ns, B, nc, uₚ_buff, uₚ_buff2, u², u²_sum, uⱼvⱼ, ft = params
    # make `μ` point to the chemical potentials: those contained in `μs_or_Ns` if μ's are fixed, or last elements of `u` otherwise
    μs_arefixed = length(u) == B*nc # is 𝜇's are not fixed, then `length(u)` exceeds `B*nc` because `u` then also contains the 𝜇's
    μ = μs_arefixed ? μs_or_Ns : @view u[end-nc+1:end] # if total number of atoms is fixed, then these elements will all be the same
    ### Linear part
    # transform `u` to p-space, write into `uₚ_buff`
    @views for i in 1:nc
        window = (i-1)B+1:i*B
        transform!(ft, u[window]; direction=:forward)
        fft_to_state!(uₚ_buff[window], ft; direction=:forward)
    end
    mul!(uₚ_buff2, H, uₚ_buff)
    # transform `uₚ_buff2` back to x-space, write into `du`
    @views for i in 1:nc
        window = (i-1)B+1:i*B
        transform!(ft, uₚ_buff2[window]; direction=:backward)
        fft_to_state!(du[window], ft; direction=:backward, makereal=true)
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
        dx = ft.xs[2] - ft.xs[1]
        if μs_or_Ns isa Number # then only total number of atoms is fixed
            u²_sum = zero(μs_or_Ns) # for storing the sum ∑ᵢ∫𝑢ᵢ²d𝑥
            for i in 1:nc
                u²_sum += sum(u²[i])*dx
                ft.basis == :cos && (u²_sum -= (u²[i][end] + u²[i][1])*dx/2)
            end
            du[end-nc+1:end] .= u²_sum - μs_or_Ns # place the sum in the residuals array; we have `nc` identical elements to keep the general structure
        else # numbers of atoms in each compoenent are fixed
            for i in 1:nc
                du[end-nc+i] = sum(u²[i])*dx - μs_or_Ns[i]
                ft.basis == :cos && (du[end-nc+i] -= (u²[i][end] + u²[i][1])*dx/2)
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
Suitable for the case when 𝐻, and hence also 𝑢ᵢ and 𝑣ᵢ, is real in x-space.
"""
function jvp_gpe_real!(Jv, v, u, params)
    H, g, μs_or_Ns, B, nc, vₚ_buff, vₚ_buff2, u², u²_sum, uⱼvⱼ, ft = params
    # make `μ` point to the chemical potentials: those contained in `μs_or_Ns` if μ's are fixed, or last elements of `u` otherwise
    μs_arefixed = length(u) == B*nc # is 𝜇's are not fixed, then `length(u)` exceeds `B*nc` because `u` then also contains the 𝜇's
    μ = μs_arefixed ? μs_or_Ns : @view u[end-nc+1:end] # if total number of atoms is fixed, then these elements will all be the same
    ### Linear part
    # transform `v` to p-space, write into `vₚ_buff`
    @views for i in 1:nc
        window = (i-1)B+1:i*B
        transform!(ft, v[window]; direction=:forward)
        fft_to_state!(vₚ_buff[window], ft; direction=:forward)
    end
    mul!(vₚ_buff2, H, vₚ_buff)
    # transform `vₚ_buff2` back to x-space, write into `Jv`
    @views for i in 1:nc
        window = (i-1)B+1:i*B
        transform!(ft, vₚ_buff2[window]; direction=:backward)
        fft_to_state!(Jv[window], ft; direction=:backward, makereal=true)
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
        dx = ft.xs[2] - ft.xs[1]
        if μs_or_Ns isa Number # then only total number of atoms is fixed
            uᵢvᵢ_sum = zero(μs_or_Ns) # for storing the sum 2∑ᵢ∫𝑢ᵢvᵢd𝑥
            for i in 1:nc
                uᵢvᵢ_sum += 2sum(uⱼvⱼ[i])*dx
                ft.basis == :cos && (uᵢvᵢ_sum -= (uⱼvⱼ[i][end] + uⱼvⱼ[i][1])*dx) # no division by 2 because of the overall factor in the line above
            end
            Jv[end-nc+1:end] .= uᵢvᵢ_sum # put the sum into place; we have `nc` identical elements to keep the general structure
        else # numbers of atoms in each compoenent are fixed
            for i in 1:nc
                Jv[end-nc+i] = 2sum(uⱼvⱼ[i])*dx
                ft.basis == :cos && (Jv[end-nc+i] -= (uⱼvⱼ[i][end] + uⱼvⱼ[i][1])*dx) # no division by 2 because of the overall factor in the line above
            end
        end
    end
    return
end