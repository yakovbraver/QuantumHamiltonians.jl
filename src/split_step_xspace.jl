"A type for storing the solution of the time-dependent GPE at a number of time points."
struct GPESolution{R}
    t::Vector{R}
    u::Vector{Vector{Complex{R}}}
end

"""
Propagate the time-dependent GPE using split-step method, with `solver` being `:strang2` or `:yoshida4`.
Return a solution object containing the funtion in the `u` field, and time moments in the `t` field.
The equation is 𝜕ₜ𝜓 = 𝑟𝐻𝜓 + 𝑟𝑔|𝜓|²𝜓 with `rhs_factor` 𝑟 = -1 for imaginary-time solving and 𝑟 = -i for real-time.
We split as 𝜕ₜ𝜓 = 𝐿𝜓 + 𝑀𝜓 where 𝐿 = 𝑟(𝐩 - 𝐀)² while 𝑀 = 𝑟(𝑈 + 𝑔|𝜓|²𝜓).
Intermediate solutions are saved `nsaves` times in addition to intial and final.
The less you save the faster it goes because not saving allows for combining the exponents between successive steps.
`g` must not include the rhs_factor.
For nonzero vector potential, `:strang2` requires `𝛬=[𝛬ˣ, 𝛬ʸ]` analytic functions satisfying `𝜕ⁱ𝛬ⁱ = 𝐴ⁱ`.
"""
function split_step(xh::XSpaceHamiltonian{R, T}, u₀::AbstractVector{<:Number}, g::AbstractMatrix{R};
                    T_max::R, dt::R, rhs_factor::Union{R, Complex{R}}=-1.0im, nsaves::Integer=0, solver::Symbol, 𝛬::Union{Nothing, AbstractVector{Function}}=nothing) where {R, T}
    # prepare time moments
    ts = zero(R):dt:T_max
    n_t = length(ts)
    nsaves += 2 # add first and last moments
    nsaves = min(nsaves, n_t) # safeguard in case `nsaves > n_t`
    save_indices = round.(Int, range(1, n_t, length=nsaves)) # indices of time moments at which to save the solution
    (save_indices[end] == save_indices[end-1]) && pop!(save_indices) # last and pre-last might coincide, then remove one
    ts_saved = ts[save_indices] # time moments at which the solution will be saved
    # prepare solution saving
    us = [eltype(u₀)[] for _ in 1:length(ts_saved)]
    us[1] = copy(u₀)
    nsave = 2

    # buffers
    uₚ = xh.buff_complex
    u = xh.buff_complex2
    copy!(u, u₀)

    if any(!isempty, xh.A) # if vector potential is present
        xh.basis == :cis || throw(ArgumentError("magnetic split-step propagation requires `basis=:cis`"))
        xh.nc == 1 || throw(ArgumentError("magnetic split-step propagation currently supports one component"))
        solver == :strang2 || throw(ArgumentError("magnetic split-step propagation currently supports only `solver=:strang2`"))
        isnothing(𝛬) && throw(ArgumentError("magnetic split-step propagation requires passing the gauge phases array `𝛬`"))
        return split_step_magnetic!(xh, u, uₚ, us, g; ts, save_indices, 𝛬, rhs_factor, dt)
    end
    
    if solver == :strang2 # applies exp(ℎ𝐿/2)exp(ℎ𝑀)exp(ℎ𝐿/2) but combines adjacent 𝐿-exponents if not saving during the present step
        b = rhs_factor * dt;
        exp_L = @. exp(rhs_factor*dt*xh.∇²)
        exp_L_2 = @. exp(rhs_factor*dt*xh.∇²/2)

        apply_exp_p²!(u, uₚ, xh.ft, exp_L_2)
        for it in 2:n_t
            # apply exp(ℎ𝑀)
            @. u *= exp(b * (xh.U[1] + g[1] * abs2(u)))
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
    else # :yoshida4
        # scheme coefficients
        c = 1/(2-2^(1/3))
        a1 = c/2; a2 = (1-c)/2;
        b1, b2, b3 = [c, 1-2c, c] .* rhs_factor .* dt
        # prepare exponentials that apply the Laplacian
        exp_a₁L = @. exp(a1*rhs_factor*dt*xh.∇²)
        exp_a₂L = @. exp(a2*rhs_factor*dt*xh.∇²)
        exp_2a₁L = @. exp(2a1*rhs_factor*dt*xh.∇²)

        apply_exp_p²!(u, uₚ, xh.ft, exp_a₁L)
        for it in 2:n_t
            @. u *= exp(b1 * (xh.U[1] + g[1] * abs2(u)))
            apply_exp_p²!(u, uₚ, xh.ft, exp_a₂L)
            @. u *= exp(b2 * (xh.U[1] + g[1] * abs2(u)))
            apply_exp_p²!(u, uₚ, xh.ft, exp_a₂L)
            @. u *= exp(b3 * (xh.U[1] + g[1] * abs2(u)))
            
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
    return GPESolution(ts_saved, us) 
end

"Propagate state `u` using split-step methods in the presence of a vector potential."
function split_step_magnetic!(xh::XSpaceHamiltonian{R}, u, uₚ, us, g; ts, save_indices, 𝛬, rhs_factor, dt) where R
    exp_cis = rhs_factor == -1 ? exp : cis
    
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
        
        rhs_factor == -1 && normalize!(u, xh)
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
The actual exponnetial factor in p-space given by `E`.
Used during non-magnetic split-step integration.
"""
function apply_exp_p²!(u, uₚ, ft, E)
    transform!(uₚ, ft, u; direction=:forward)
    uₚ .*= E
    transform!(u, ft, uₚ; direction=:backward, normalise=true)
    return
end