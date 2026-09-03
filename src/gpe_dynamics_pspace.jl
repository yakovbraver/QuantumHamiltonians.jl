"Convenience caller for the 1-component case, where `ψ₀` is an analytic function or a vector representing discretised functions, `g` is a number, and `ψ₀_iseven` is a Bool."
function propagate(xh::PSpaceHamiltonian{Storage, R}, ψ₀::Union{Function, AbstractVector}, g::R=zero(R);
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
function propagate(xh::PSpaceHamiltonian{Storage, R, T}, ψ₀::Union{AbstractVector{<:Function}, AbstractVector{<:AbstractVector}, AbstractVector{<:Number}}, g::AbstractMatrix{R}=zeros(R, xh.nc, xh.nc);
                   ψ₀_iseven::AbstractVector{Bool}=falses(length(ψ₀)), T_max::R, dt::R, itime::Bool=false,
                   solver=(iszero(g) ? ODE.LinearExponential() : itime ? ODE.LawsonEuler() : ODE.ETDRK4()), nsaves::Integer=0) where {Storage, R, T}
    (;xlims, L, B, basis, nc, ft) = xh

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
        for c in 1:nc
            transform!(ft, ψ₀[c])
            ψ₀ₚ_block = @view ψ₀ₚ[(c-1)*B+1:c*B]
            fft_to_state!(ψ₀ₚ_block, ft; makereal=(ψ₀_iseven[c] && ψ₀_arereal[c]))
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
        N = 2ft.M + 1 # number of points in each dimension
        dx = L ./ N
        G .*= prod(@. dx / L^2) # After bfft, resulting `u` must be divided by √𝐿; since we have `u^3`, we must divide by 𝐿√𝐿. Then, after fft the result must be multiplied by Δ𝑥/√𝐿. So Δ𝑥/𝐿² in total.
    elseif basis == :sin
        N = ft.M
        dx = L ./ (N+1)
        G .*= prod(@. dx / (2L)^2) # Same as for cis but with √(2𝐿) instead of √𝐿
    else # basis == :cos
        N = ft.M + 1
        dx = L ./ (N-1)
        G .*= prod(@. dx / (2L)^2) # Same as for cis but with √(2𝐿) instead of √𝐿
    end
    # if all 𝑔ᵢⱼ are equal, then make the final `g_input` a scalar; a specialisation of `u∑gu²_complex!` exists for this case
    G_input = nc == 1 || allequal(G) ? G[1] : G

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
    𝑢′ᵢ = 𝑢ᵢ∑ⱼ 𝑔ᵢⱼ|𝑢ⱼ|²
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
    u∑gu²_complex!(du, u², u²_sum, g, nc, B) # if all 𝑔ᵢⱼ are equal, then `g` is a number, and a special method will be called.
    # transform `du` to p-space in-place
    for i in 1:nc
        @views fft_plan! * du[(i-1)B+1:i*B]
        basis == :cos && (du[(i-1)B+1] /= √2; du[i*B] /= √2)
    end
    return
end

"""
For each `i`th component, calculate ∑ⱼ𝑔ᵢⱼ|𝑢ⱼ|² and multiply by the 𝑖th block of `du`, assumed to contain 𝑢ᵢ.
"""
function u∑gu²_complex!(du, u², u²_sum, g::AbstractMatrix{<:Number}, nc, B)
    for i in 1:nc
        @. u²_sum = g[i, 1] * u²[1] # better to check whether `g[i, 1]` is zero and fill with zeros?
        for j in 2:nc
            g[i, j] == 0 && continue
            @. u²_sum += g[i, j] * u²[j]
        end
        @views du[(i-1)B+1:i*B] .*= u²_sum
    end
    return
end

"""
Calculate 𝑔∑ⱼ|𝑢ⱼ|² and multiply by each 𝑖th block of `du`, assumed to contain 𝑢ᵢ. Here `g` is the same for all components and is a number.
"""
function u∑gu²_complex!(du, u², u²_sum, g::Number, nc, B) # `u²_sum` is not used but kept for compatibility with the other method
    # accumulate ∑ⱼ|𝑢ⱼ|² in `u²[1]`
    for j in 2:nc
        @turbo u²[1] .+= u²[j]
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
For each `i`th component, calculate ∑ⱼ𝑔ᵢⱼ|𝑢ⱼ|² and multiply by `u_re[i]` and `u_im[i]`, assumed to contain ℜ𝑢ᵢ and ℑ𝑢ᵢ.
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
Calculate 𝑔∑ⱼ|𝑢ⱼ|² and multiply by `u_re[i]` and `u_im[i]`, assumed to contain ℜ𝑢ᵢ and ℑ𝑢ᵢ. Here `g` is the same for all components and is a number.
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
