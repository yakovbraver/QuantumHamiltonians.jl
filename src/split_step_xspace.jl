"A type for storing the solution of the time-dependent GPE at a number of time points."
struct GPESolution{R}
    t::Vector{R}
    u::Vector{Vector{Complex{R}}}
end

"""
Propagate the time-dependent GPE using split-step method, with `solver` being `:strang2` or `:yoshida4`.
Return a solution object containing the funtion in the `u` field, and time moments in the `t` field.
The equation is ∂ₜ𝜓 = 𝑟𝐻𝜓 + 𝑟𝑔|𝜓|²𝜓 with `rhs_factor` 𝑟 = -1 for imaginary-time solving and 𝑟 = -i for real-time.
We split as ∂ₜ𝜓 = 𝐿𝜓 + 𝑀𝜓 where 𝐿 = 𝑟(𝐩 - 𝐀)² while 𝑀 = 𝑟(𝑈 + 𝑔|𝜓|²𝜓).
Intermediate solutions are saved `nsaves` times in addition to intial and final.
The less you save the faster it goes because not saving allows for combining the exponents between successive steps.
`g` must not include the rhs_factor.
"""
function split_step(xh::XSpaceHamiltonian{R, T}, u::AbstractVector{<:Number}, g::AbstractMatrix{R};
                    T_max::R, dt::R, rhs_factor::Union{R, Complex{R}}=-1.0im, nsaves::Integer=0, solver::Symbol) where {R, T}
    # prepare time moments
    ts = zero(R):dt:T_max
    n_t = length(ts)
    nsaves += 2 # add first and last moments
    nsaves = min(nsaves, n_t)
    save_indices = round.(Int, range(1, n_t, length=nsaves)) # indices of time moments at which to save the solution
    (save_indices[end] == save_indices[end-1]) && pop!(save_indices) # last and pre-last might coincide, then remove one
    ts_saved = ts[save_indices] # time moments at which the solution will be saved
    # prepare solution saving
    us = [eltype(u)[] for _ in 1:length(ts_saved)]
    us[1] = copy(u)
    nsave = 2

    uₚ = xh.buff_complex
    
    if solver == :strang2 # applies exp(ℎ𝐿/2)exp(ℎ𝑀)exp(ℎ𝐿/2) but combines adjacent 𝐿-exponents if not saving during the present step
        b = rhs_factor * dt;
        exp_L = @. exp(rhs_factor*dt*xh.∇²)
        exp_L_2 = @. exp(rhs_factor*dt*xh.∇²/2)

        apply_exp!(u, uₚ, xh.ft, exp_L_2)
        for it in 1:n_t
            # apply exp(ℎ𝑀)
            @. u *= exp(b * (xh.U[1] + g[1] * abs2(u)))
            if it == save_indices[nsave] # if needs saving during this step, then apply exp(ℎ𝐿/2), save, and apply exp(ℎ𝐿/2) again to start the next step
                apply_exp!(u, uₚ, xh.ft, exp_L_2)
                us[nsave] = copy(u)
                nsave += 1
                # apply exp(ℎ𝐿/2) to start the next iteration, but not needed in the very end
                if it < n_t
                    apply_exp!(u, uₚ, xh.ft, exp_L_2)
                end
            else # if don't need to save during this step, then apply exp(ℎ𝐿)
                apply_exp!(u, uₚ, xh.ft, exp_L)
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

        apply_exp!(u, uₚ, xh.ft, exp_a₁L)
        for it in 1:n_t
            @. u *= exp(b1 * (xh.U[1] + g[1] * abs2(u)))
            apply_exp!(u, uₚ, xh.ft, exp_a₂L)
            @. u *= exp(b2 * (xh.U[1] + g[1] * abs2(u)))
            apply_exp!(u, uₚ, xh.ft, exp_a₂L)
            @. u *= exp(b3 * (xh.U[1] + g[1] * abs2(u)))
            
            if it == save_indices[nsave] # if needs saving during this step, then apply exp(𝑎₁ℎ𝐿), save, and apply exp(𝑎₁ℎ𝐿) again to start the next step
                apply_exp!(u, uₚ, xh.ft, exp_a₁L)
                us[nsave] = copy(u)
                nsave += 1
                # apply exp(𝑎₁ℎ𝐿) to start the next iteration, but not needed in the very end
                if it < n_t
                    apply_exp!(u, uₚ, xh.ft, exp_a₁L)
                end
            else # if don't need to save during this step, then apply exp(2𝑎₁ℎ𝐿)
                apply_exp!(u, uₚ, xh.ft, exp_2a₁L)
            end
        end
    end     
    return GPESolution(ts_saved, us) 
end

"Helper function used during split-step integration: apply the exponential factor `E` to `u` in p-space."
function apply_exp!(u, uₚ, ft, E)
    transform!(uₚ, ft, u; direction=:forward)
    uₚ .*= E
    transform!(u, ft, uₚ; direction=:backward, normalise=true)
    return
end