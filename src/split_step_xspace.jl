"A type for storing the solution of the time-dependent GPE at a number of time points."
struct GPESolution{R, T}
    t::Vector{R}
    u::Vector{Vector{T}}
end

"""
Propagate the time-dependent GPE using split-step method, with `solver` being `:strang2` or `:yoshida4`.
The initial state `u` will be mutated during solving.
The equation is 𝜕ₜ𝜓 = 𝑟𝐻𝜓 + 𝑟𝑔|𝜓|²𝜓 with `rhs_factor` 𝑟 = -1 for imaginary-time solving and 𝑟 = -i for real-time.
We split as 𝜕ₜ𝜓 = 𝐿𝜓 + 𝑀𝜓 where 𝐿 = 𝑟(𝐩 - 𝐀)² while 𝑀 = 𝑟(𝑈 + 𝑔|𝜓|²𝜓).
Return a solution object `sol` with the intermediate solutions stored in `sol.u` and time moments in `sol.t`.
Intermediate solutions are saved `nsaves` times in addition to intial and final.
The less you save the faster it goes because not saving allows for combining the exponents between successive steps.
For nonzero vector potential, `:strang2` requires `𝛬=[𝛬ˣ, 𝛬ʸ]` analytic functions satisfying `𝜕ⁱ𝛬ⁱ = 𝐴ⁱ`.
"""
function split_step!(xh::XSpaceHamiltonian{R}, u::AbstractVector{T}, g::AbstractMatrix{R}, 𝛬::AbstractVector{<:Function}=[zero, zero];
                     T_max::R, dt::R, itime::Bool=false, nsaves::Integer=0, solver::Symbol) where {R, T}
    # prepare time moments
    ts = zero(R):dt:T_max
    n_t = length(ts)
    nsaves += 2 # add first and last moments
    nsaves = min(nsaves, n_t) # safeguard in case `nsaves > n_t`
    save_indices = round.(Int, range(1, n_t, length=nsaves)) # indices of time moments at which to save the solution
    (save_indices[end] == save_indices[end-1]) && pop!(save_indices) # last and pre-last might coincide, then remove one
    ts_saved = ts[save_indices] # time moments at which the solution will be saved
    # prepare solution saving. `T` is the type of `u`, and it reflects the type (real/complex) of equation since this was determined in `propagate!`
    us = [T[] for _ in 1:length(ts_saved)]
    us[1] = copy(u)

    # prepare the p-space buffer
    uₚ = T <: Real && xh.basis != :cis ? xh.buff_real : xh.buff_complex
    # `u` is a copy of the input state, so it will be directly used for propagation

    if any(!isempty, xh.A) # if at least one vector potential 𝐴 is present
        xh.basis == :cis || throw(ArgumentError("magnetic split-step propagation requires `basis=:cis`"))
        xh.nc == 1 || throw(ArgumentError("magnetic split-step propagation currently supports one component"))
        solver == :strang2 || throw(ArgumentError("magnetic split-step propagation currently supports only `solver=:strang2`"))
        any(==(zero), 𝛬[findall(!isempty, xh.A)]) && throw(ArgumentError("Each nonzero 𝐴ⁱ must have a matching nonzero 𝛬ⁱ"))
        return strang2_magnetic(xh, u, uₚ, us, g, 𝛬; ts, save_indices, itime, dt)
    else 
        if solver == :strang2
            return strang2!(xh, u, uₚ, us, g; ts, save_indices, itime, dt)
        else
            return yoshida4!(xh, u, uₚ, us, g; ts, save_indices, itime, dt)
        end 
    end    
    return GPESolution(ts_saved, us) 
end

"""
Propagate `u` using 2nd order Strang splitting. Save intermediate solutions to `us` and return a `GPESolution` object.
Applies exp(ℎ𝐿/2)exp(ℎ𝑀)exp(ℎ𝐿/2) but combines adjacent 𝐿-exponents in real-time propagation when not saving during the present step.
"""
function strang2!(xh, u, uₚ, us, g; ts, save_indices, itime, dt)
    exp_cis = itime ? exp : cis
    n_t = length(ts)
    nsave = 2

    exp_L = @. exp_cis(-dt*xh.∇²)
    exp_L_2 = @. exp_cis(-dt*xh.∇²/2)

    if itime # we need to renormalise after every step, so combining steps is not possible. Therefore it is cleaner to have the itime case separately
        for it in 2:n_t
            apply_exp_p²!(u, uₚ, xh.ft, exp_L_2)
            @. u *= exp_cis(-dt * (xh.U[1] + g[1] * abs2(u))) # apply exp(ℎ𝑀)
            apply_exp_p²!(u, uₚ, xh.ft, exp_L_2)
            normalize!(u, xh)
            if it == save_indices[nsave]
                us[nsave] = copy(u)
                nsave += 1
            end
        end
    else # real-time propagation
        apply_exp_p²!(u, uₚ, xh.ft, exp_L_2)
        for it in 2:n_t
            @. u *= exp_cis(-dt * (xh.U[1] + g[1] * abs2(u))) # apply exp(ℎ𝑀)
            if it == save_indices[nsave] # if needs saving during this step, then apply exp(ℎ𝐿/2), save, and apply exp(ℎ𝐿/2) again to start the next step
                apply_exp_p²!(u, uₚ, xh.ft, exp_L_2)
                us[nsave] = copy(u)
                nsave += 1
                # apply exp(ℎ𝐿/2) to start the next iteration, but not needed in the very end
                if it < n_t
                    apply_exp_p²!(u, uₚ, xh.ft, exp_L_2)
                end
            else # if don't need to save during this step, then apply exp(ℎ𝐿)
                apply_exp_p²!(u, uₚ, xh.ft, exp_L)
            end
        end
    end
    return GPESolution(ts[save_indices], us)
end

"""
Propagate `u` using 4th order Yoshida splitting. Save intermediate solutions to `us` and return a `GPESolution` object.
Applies exp(ℎ𝑎₁𝐿) exp(ℎ𝑏₁𝑀)exp(ℎ𝑎₂𝐿) exp(ℎ𝑏₂𝑀)exp(ℎ𝑎₂𝐿) exp(ℎ𝑏₁𝑀)exp(ℎ𝑎₁𝐿) but combines adjacent 𝐿-exponents in real-time propagation and not saving during the present step.
"""
function yoshida4!(xh, u, uₚ, us, g; ts, save_indices, itime, dt)
    exp_cis = itime ? exp : cis
    n_t = length(ts)
    nsave = 2

    # scheme coefficients, including the minuses from the exponents
    c = 1/(2-2^(1/3))
    a1 = -c/2 * dt;  a2 = -(1-c)/2 * dt
    b1 = -c   * dt;  b2 = -(1-2c)  * dt
    # prepare exponentials that apply the Laplacian
    exp_a₁L = @. exp_cis(a1 * xh.∇²)
    exp_a₂L = @. exp_cis(a2 * xh.∇²)
    exp_2a₁L = @. exp_cis(2a1 * xh.∇²)

    if itime
        for it in 2:n_t
            apply_exp_p²!(u, uₚ, xh.ft, exp_a₁L)
            @. u *= exp_cis(b1 * (xh.U[1] + g[1] * abs2(u)))
            apply_exp_p²!(u, uₚ, xh.ft, exp_a₂L)
            @. u *= exp_cis(b2 * (xh.U[1] + g[1] * abs2(u)))
            apply_exp_p²!(u, uₚ, xh.ft, exp_a₂L)
            @. u *= exp_cis(b1 * (xh.U[1] + g[1] * abs2(u)))
            apply_exp_p²!(u, uₚ, xh.ft, exp_a₁L)
            normalize!(u, xh)
            if it == save_indices[nsave]
                us[nsave] = copy(u)
                nsave += 1
            end
        end
    else # real-time propagation
        apply_exp_p²!(u, uₚ, xh.ft, exp_a₁L)
        for it in 2:n_t
            @. u *= exp_cis(b1 * (xh.U[1] + g[1] * abs2(u)))
            apply_exp_p²!(u, uₚ, xh.ft, exp_a₂L)
            @. u *= exp_cis(b2 * (xh.U[1] + g[1] * abs2(u)))
            apply_exp_p²!(u, uₚ, xh.ft, exp_a₂L)
            @. u *= exp_cis(b1 * (xh.U[1] + g[1] * abs2(u)))
            
            if it == save_indices[nsave] # if needs saving during this step, then apply exp(𝑎₁ℎ𝐿), save, and apply exp(𝑎₁ℎ𝐿) again to start the next step
                apply_exp_p²!(u, uₚ, xh.ft, exp_a₁L)
                us[nsave] = copy(u)
                nsave += 1
                # apply exp(𝑎₁ℎ𝐿) to start the next iteration, but not needed in the very end
                if it < n_t
                    apply_exp_p²!(u, uₚ, xh.ft, exp_a₁L)
                end
            else # if don't need to save during this step, then apply exp(2𝑎₁ℎ𝐿)
                apply_exp_p²!(u, uₚ, xh.ft, exp_2a₁L)
            end
        end
    end
    return GPESolution(ts[save_indices], us)
end

"""
Propagate `u` using 2nd order Strang splitting in the presence of a vector potential.
Save intermediate solutions to `us` and return a `GPESolution` object.
"""
function strang2_magnetic(xh::XSpaceHamiltonian{R}, u, uₚ, us, g, 𝛬::AbstractVector{Function}; ts, save_indices, itime, dt) where R
    exp_cis = itime ? exp : cis
    
    # resample the potential `U` because it contains unneeded (𝐀²)
    U = similar(xh.U[1, 1])
    if !isnothing(xh.𝑈[1, 1])
        sample!(U, xh.𝑈[1, 1], xh.ft.xs)
    else
        U .= 0
    end
    if xh.Γ[1] != 0
        @. U -= im*xh.Γ[1]/2
    end

    nx = size(xh.ft.xs, 1)
    dims = (nx, nx)

    # make views of `u` and `uₚ` reshaped into 2D to apply FFT (normaly reshaping would be done in `transform!`)
    u_grid = reshape(u, dims) 
    uₚ_grid = reshape(uₚ, dims)
    # plans for 1D transforms of 2D data. TODO consider incorporating in FourierTransformerX
    plan_forward_x = FFTW.plan_fft(u_grid, 1)
    plan_backward_x = FFTW.plan_ifft(u_grid, 1)
    plan_forward_y = FFTW.plan_fft(u_grid, 2)
    plan_backward_y = FFTW.plan_ifft(u_grid, 2)
    
    # sample the functions 𝛬ⁱ and store cis(𝛬ⁱ/𝛿) in flat vectors
    # TODO handle the case when `𝛬[i] == zero`
    exp_iΛˣ = Vector{Complex{R}}(undef, xh.B)
    exp_iΛʸ = similar(exp_iΛˣ)
    for j in 1:nx, i in 1:nx
        flat = i + (j - 1) * nx
        x, y = xh.ft.xs[i, 1], xh.ft.xs[j, 2]
        exp_iΛˣ[flat] = cis(𝛬[1](x, y) / xh.δ) # each call to 𝛬[i] gives (4 allocations: 80 bytes). If `𝛬` is a tuple, then allocations go away, but anyway we will generalise to matrix 𝛬 for the multicomponent case
        exp_iΛʸ[flat] = cis(𝛬[2](x, y) / xh.δ)
    end

    exp_pˣ² = exp_cis.(-(dt / 2) .* xh.∇[1].^2)
    exp_pʸ² = exp_cis.(-dt .* xh.∇[2].^2)

    n_t = length(ts)
    nsave = 2
    b = -dt/2

    for it in 2:n_t
        @. u *= exp_cis(b * (U + g[1] * abs2(u)))
        apply_gauge_kinetic!(u, u_grid, uₚ, uₚ_grid, exp_iΛˣ, exp_pˣ², plan_forward_x, plan_backward_x)
        apply_gauge_kinetic!(u, u_grid, uₚ, uₚ_grid, exp_iΛʸ, exp_pʸ², plan_forward_y, plan_backward_y)
        apply_gauge_kinetic!(u, u_grid, uₚ, uₚ_grid, exp_iΛˣ, exp_pˣ², plan_forward_x, plan_backward_x)
        @. u *= exp_cis(b * (U + g[1] * abs2(u)))
        
        itime && normalize!(u, xh)
        if it == save_indices[nsave]
            us[nsave] = copy(u)
            nsave += 1
        end
    end
    return GPESolution(ts[save_indices], us)
end

"""
Apply the Strang-split version of exp((𝐩 - 𝐀)²) to the x-space state `u`, by transforming to p-space (and using uₚ as a buffer).
Used during magnetic split-step integration.
"""
function apply_gauge_kinetic!(u, u_grid, uₚ, uₚ_grid, exp_iΛⁱ, exp_pⁱ², plan_forward, plan_backward)
    @. u *= conj(exp_iΛⁱ)
    mul!(uₚ_grid, plan_forward, u_grid)
    @. uₚ *= exp_pⁱ²
    mul!(u_grid, plan_backward, uₚ_grid)
    @. u *= exp_iΛⁱ
    return
end

"""
Apply the exponential exp(𝐩²) to the x-space state `u`, by transforming to p-space (and using uₚ as a buffer).
The actual exponential factor in p-space given by `E`.
Used during non-magnetic split-step integration.
"""
function apply_exp_p²!(u, uₚ, ft, E)
    transform!(uₚ, ft, u; direction=:forward)
    uₚ .*= E
    transform!(u, ft, uₚ; direction=:backward, normalise=true)
    return
end