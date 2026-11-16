"A type for storing the solution of the time-dependent GPE at a number of time points."
struct GPESolution{R}
    t::Vector{R}
    u::Vector{Vector{Complex{R}}}
end

"""
Propagate the time-dependent GPE using split-step method, with `solver` being `strang2` or `yoshida4`.
Return a solution object containing the funtion in the `u` field, and time moments in the `t` field.
`nsaves`: how many points in addition to the initial and final to save.
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
    
    # The equation i∂ₜ𝜓 = 𝐻𝜓 + 𝑔|𝜓|²𝜓 is coded as ∂ₜ𝜓 = -i𝐻𝜓 -i𝑔|𝜓|²𝜓.
    # Denote the `rhs_factor` -i = 𝑟. For 𝐻 = -Δ + 𝑈 we have
    # ∂ₜ𝜓 = 𝑟Δ𝜓 + 𝑟(𝑈 + 𝑔|𝜓|²)𝜓.
    # The first equation is solved in momentum space by (ℎ is the time step)
    # 𝜓ₚ = exp(𝑎𝑟ℎ𝑝²)𝜓ₚ [𝑝² already contains required signs. Meanwhile, 𝑟 comes from the equation; not present when 𝐻 is acting]
    # The second equation is solved in coordinate space by
    # 𝜓 = exp(𝑏𝑟ℎ(𝑈+𝑔|𝜓|²))𝜓

    # scheme coefficients

    if solver == :strang2
        a = 1/2
        b = rhs_factor * dt;
        ex = @. exp(rhs_factor*a*dt*xh.∇²)

        for it in 1:n_t
            transform!(uₚ, xh.ft, u; direction=:forward)
            uₚ .*= ex
            transform!(u, xh.ft, uₚ; direction=:backward, normalise=true)
            @. u *= exp(b * (xh.U[1] + g[1] * abs2(u)))
            transform!(uₚ, xh.ft, u; direction=:forward)
            uₚ .*= ex
            transform!(u, xh.ft, uₚ; direction=:backward, normalise=true)
            if it == save_indices[nsave]
                us[nsave] = copy(u)
                nsave += 1
            end
        end
    else
        c = 1/(2-2^(1/3))
        a = [c/2, (1-c)/2]    .* rhs_factor .* dt
        b = [  c,    1-2c, c] .* rhs_factor .* dt
        # prepare exponentials that apply the Laplacian
        E = [typeof(rhs_factor)[] for _ in 1:4]
        E[1] = @. exp(a[1]*xh.∇²)
        E[2] = @. exp(a[2]*xh.∇²)
        E[3] = E[2]
        E[4] = E[1]

        for it in 1:n_t
            for i in 1:4
                transform!(uₚ, xh.ft, u; direction=:forward)
                uₚ .*= E[i]
                transform!(u, xh.ft, uₚ; direction=:backward, normalise=true)
                if i != 4
                    @. u *= exp(b[i] * (xh.U[1] + g[1] * abs2(u)))
                end
            end
            if it == save_indices[nsave]
                us[nsave] = copy(u)
                nsave += 1
            end
        end
    end     
    return GPESolution(ts_saved, us) 
end