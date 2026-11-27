"A type for storing the solution of the time-dependent GPE at a number of time points."
struct GPESolution{R, T}
    t::Vector{R}
    u::Vector{Vector{T}}
end

"""
Propagate the time-dependent GPE using split-step method, with `solver` being `:strang2` or `:yoshida4` for real-time; `:lie1` or `:_lie1` for imaginary time.
The initial state `u` will be mutated during solving.
The equation is 𝜕ₜ𝜓 = 𝑟𝐻𝜓 + 𝑟𝑔|𝜓|²𝜓 with 𝑟 = -1 for imaginary-time solving and 𝑟 = -i for real-time.
We split as 𝜕ₜ𝜓 = 𝐿𝜓 + 𝑀𝜓 where 𝐿 = 𝑟(𝐩 - 𝐀)² while 𝑀 = 𝑟(𝑈 + 𝑔|𝜓|²𝜓).
Return a solution object `sol` with the intermediate solutions stored in `sol.u` and time moments in `sol.t`.
Intermediate solutions are saved `nsaves` times in addition to intial and final.
The less you save the faster it goes because not saving allows for combining the exponents between successive steps.
For nonzero vector potential, `:strang2` requires passing a matrix `𝛬` of analytic functions satisfying `𝜕ⁱ𝛬ⁱ = 𝐴ⁱ`, stored in the same shape as `xh.A`.
"""
function split_step!(xh::XSpaceHamiltonian{R}, u::AbstractVector{T}, g::AbstractMatrix{R}, 𝛬::AbstractMatrix{<:Function}=fill(zero, size(xh.A));
                     T_max::R, dt::R, itime::Bool=false, nsaves::Integer=0, solver::Symbol) where {R, T}
    # prepare time moments
    ts = zero(R):dt:T_max
    n_t = length(ts)
    nsaves += 2 # add first and last moments
    nsaves = min(nsaves, n_t) # safeguard in case `nsaves > n_t`
    save_indices = round.(Int, range(1, n_t, length=nsaves)) # indices of time moments at which to save the solution
    (save_indices[end] == save_indices[end-1]) && (pop!(save_indices); nsaves -= 1) # last and pre-last might coincide, then remove one
    # prepare solution saving. `T` is the type of `u`, and it reflects the type (real/complex) of equation since this was determined in `propagate!`
    us = [T[] for _ in 1:nsaves]
    us[1] = copy(u)

    # prepare the p-space buffer
    uₚ = T <: Real && xh.basis != :cis ? xh.buff_real : xh.buff_complex
    # `u` is a copy of the input state, so it will be directly used for propagation

    if any(!isempty, xh.A) # if at least one projection of 𝐀 is nonzero
        if itime
            return lie1_itime_nonzeroA!(xh, u, uₚ, us, g, 𝛬, ts, save_indices, dt)
        else
            return strang2_nonzeroA!(xh, u, uₚ, us, g, 𝛬, ts, save_indices, dt)
            # return (xh, u, uₚ, us, g, 𝛬, ts, save_indices, dt)
        end
    else # 𝐀 = 0
        if itime
            return lie1_itime!(xh, u, uₚ, us, g, ts, save_indices, dt, solver)
        elseif solver == :strang2
            return strang2!(xh, u, uₚ, us, g, ts, save_indices, dt)
        else
            return yoshida4!(xh, u, uₚ, us, g, ts, save_indices, dt)
        end 
    end    
end

"""
Propagate `u` in imaginary time using 1st order Lie splitting exp(ℎ𝑀)exp(ℎ𝐿). Save intermediate solutions to `us` and return a `GPESolution` object.
Used when vector potential 𝐀 is not present.
For `solver=:lie1` (recommended), exp(ℎ𝑀) is the solution of 𝜕ₜ = -(𝑈 + 𝑔|𝜓|²)𝜓 with |𝜓|² frozen at the value in the beginning of this substep.
For `solver=:_lie1`, exp(ℎ𝑀) means using the exact solution (with |𝜓|² variable), which is 𝜓(ℎ) = 𝜓ₒexp(-𝑈ℎ) / √(1 + 𝑔|𝜓₀|²(1 - exp(-2𝑈ℎ))/𝑈); See https://doi.org/10.1137/S1064827503422956.
In our tests, 1st order splitting is more accurate than 2nd (or 4th), and `:lie1` always wins over `:_lie1`. Tested by comparing with Newton-Raphson results converged to machine precision.
"""
function lie1_itime!(xh, u, uₚ, us, g, ts, save_indices, dt, solver)
    n_t = length(ts)
    nsave = 2

    exp_L = @. exp(-dt*xh.∇²)

    if solver == :lie1 # splitting with frozen |𝜓|²
        for it in 2:n_t
            apply_exp_p²!(u, uₚ, xh.ft, exp_L)
            @. u *= exp(-dt * (xh.U[1] + g[1] * abs2(u))) # apply exp(ℎ𝑀)
            normalize!(u, xh)
            if it == save_indices[nsave]
                us[nsave] = copy(u)
                nsave += 1
            end
        end
    else # splitting with variable |𝜓|²
        # pre-compute exp(-𝑈ℎ) and 𝑔(1 - exp(-2𝑈ℎ))/𝑈)
        e1 = @. exp(-dt*xh.U[1])
        e2 = map(xh.U[1]) do U # compute the term 𝑔(1 - exp(-2𝑈ℎ))/𝑈; for 𝑈 = 0 exactly, enforce the limit 2ℎ𝑔; when 𝑈 is not exactly 0, expm1 / U converges to this limit by itself
            iszero(U) ? 2dt*g[1] : g[1] * -expm1(-2dt*U) / U
        end
        for it in 2:n_t
            apply_exp_p²!(u, uₚ, xh.ft, exp_L)
            @. u *= e1 / √( 1 + e2*abs2(u) ) # compute 𝜓(ℎ) = 𝜓ₒexp(-𝑈ℎ) / √(1 + 𝑔|𝜓₀|²(1 - exp(-2𝑈ℎ))/𝑈)
            normalize!(u, xh)
            if it == save_indices[nsave]
                us[nsave] = copy(u)
                nsave += 1
            end
        end
    end
    return GPESolution(ts[save_indices], us)
end

"""
Propagate `u` in imaginary time using 1st order Lie splitting exp(ℎ𝑀)exp(ℎ𝐿). Save intermediate solutions to `us` and return a `GPESolution` object.
Used when vector potential 𝐀 is present.
In our tests, 1st order splitting is more accurate than 2nd. Tested by comparing with Newton-Raphson results converged to machine precision.
"""
function lie1_itime_nonzeroA!(xh::XSpaceHamiltonian{R}, u, uₚ, us, g, 𝛬::AbstractMatrix{Function}, ts, save_indices, dt) where R
    # make views of `u` and `uₚ` reshaped into 2D to apply FFT (normaly reshaping would be done in `transform!`)
    u_grid = reshape(u, size(xh.ft.plan_forward)) 
    uₚ_grid = reshape(uₚ, size(xh.ft.plan_forward))
    # plans for 1D transforms of 2D data
    plan_forward_x = FFTW.plan_fft(u_grid, 1)
    plan_backward_x = FFTW.plan_ifft(u_grid, 1)
    plan_forward_y = FFTW.plan_fft(u_grid, 2)
    plan_backward_y = FFTW.plan_ifft(u_grid, 2)
    
    # The first term of the sequence (for real-time) is cis(-ℎ𝑀/2) = cis(-ℎ(𝑈 + 𝑔|𝜓|²)/2), then goes cis(-𝛬ˣ). One could combine them into a single exponential.
    # However, we find a ~20% overall speed-up by pre-computing the exponential cis(-𝛬ˣ - ℎ𝑈/2), and keeping cis(-ℎ𝑔|𝜓|²/2) in the main loop.
    # This way the argument of the latter `cis` is usually less than π/4, resulting in a ~1.5x faster evaluation of `cis`.
    # For real-time, we precompute the terms `cis_Λˣ_U` = cis(𝛬ˣ - ℎ𝑈/2) and `cis_Λˣ⁺_U` = cis(𝛬ˣ⁺ - ℎ𝑈/2)
    # For itime, we precompute `cis_Λˣ_U` = cis(𝛬ˣ) and `cis_Λˣ⁺_U` = cis(𝛬ˣ⁺)exp(-ℎ𝑈)
    Λˣ_iszero = 𝛬[1] == zero
    Λˣ_U_bothzero = Λˣ_iszero && isnothing(xh.𝑈[1, 1]) && iszero(xh.Γ[1])
    cis_Λˣ = Vector{Complex{R}}(undef, (Λˣ_iszero ? 0 : xh.B))
    if !Λˣ_iszero
        cis_𝛬ˣ(x, y) = cis(𝛬[1](x, y) / xh.δ)
        sample!(cis_Λˣ, cis_𝛬ˣ, xh.ft.xs)
    end
    cis_Λˣ⁺_exp_U = Vector{Complex{R}}(undef, (Λˣ_U_bothzero ? 0 : xh.B))
    if !Λˣ_U_bothzero
        cis_𝛬ˣ⁺_exp_𝑈(x, y) = cis(-𝛬[1](x, y) / xh.δ) * exp((xh.𝑈[1, 1](x, y) - im*xh.Γ[1]/2) * -dt)
        sample!(cis_Λˣ⁺_exp_U, cis_𝛬ˣ⁺_exp_𝑈, xh.ft.xs)
    end
    cis_Λˣ_Λʸ = Vector{Complex{R}}(undef, xh.B) # at least one of 𝛬ˣ, 𝛬ʸ is not zero, so cis((𝛬ˣ-𝛬ʸ)/𝛿) will always be needed.
    _𝛬 = replace(𝛬, zero => Returns(zero(R))) # replace so that _𝛬 can be safely called with two arguments
    cis_𝛬ˣ_𝛬ʸ(x, y) = cis((_𝛬[1](x, y) - _𝛬[2](x, y)) / xh.δ)
    sample!(cis_Λˣ_Λʸ, cis_𝛬ˣ_𝛬ʸ, xh.ft.xs)

    exp_pˣ² = @. exp(-dt/2 * xh.∇[1]^2)
    exp_pʸ² = @. exp(-dt * xh.∇[2]^2)

    n_t = length(ts)
    nsave = 2
    b = -dt * g[1]

    for it in 2:n_t
        if Λˣ_U_bothzero
            @. u *= exp(b * abs2(u))
        else
            @. u *= exp(b * abs2(u)) * cis_Λˣ⁺_exp_U
        end
        apply_exp_p²!(u_grid, uₚ, uₚ_grid, exp_pˣ², plan_forward_x, plan_backward_x)
        @. u *= cis_Λˣ_Λʸ
        apply_exp_p²!(u_grid, uₚ, uₚ_grid, exp_pʸ², plan_forward_y, plan_backward_y)
        @. u *= conj(cis_Λˣ_Λʸ)
        apply_exp_p²!(u_grid, uₚ, uₚ_grid, exp_pˣ², plan_forward_x, plan_backward_x)
        !Λˣ_iszero && (@. u *= cis_Λˣ)
        normalize!(u, xh)
        if it == save_indices[nsave]
            us[nsave] = copy(u)
            nsave += 1
        end
    end
    return GPESolution(ts[save_indices], us)
end

"""
Propagate `u` using 2nd order Strang splitting. Save intermediate solutions to `us` and return a `GPESolution` object.
Applies exp(ℎ𝐿/2)exp(ℎ𝑀)exp(ℎ𝐿/2) but combines adjacent 𝐿-exponents in real-time propagation when not saving during the present step.
"""
function strang2!(xh, u, uₚ, us, g, ts, save_indices, dt)
    n_t = length(ts)
    nsave = 2

    exp_L = @. cis(-dt*xh.∇²)
    exp_L_2 = @. cis(-dt*xh.∇²/2)

    apply_exp_p²!(u, uₚ, xh.ft, exp_L_2)
    for it in 2:n_t
        @. u *= cis(-dt * (xh.U[1] + g[1] * abs2(u))) # apply exp(ℎ𝑀)
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
    return GPESolution(ts[save_indices], us)
end

"""
Propagate `u` using 4th order Yoshida splitting. Save intermediate solutions to `us` and return a `GPESolution` object.
Applies exp(ℎ𝑎₁𝐿) exp(ℎ𝑏₁𝑀)exp(ℎ𝑎₂𝐿) exp(ℎ𝑏₂𝑀)exp(ℎ𝑎₂𝐿) exp(ℎ𝑏₁𝑀)exp(ℎ𝑎₁𝐿) but combines adjacent 𝐿-exponents in real-time propagation and not saving during the present step.
"""
function yoshida4!(xh, u, uₚ, us, g, ts, save_indices, dt)
    n_t = length(ts)
    nsave = 2

    # scheme coefficients, including the minuses from the exponents
    c = 1/(2-2^(1/3))
    a1 = -c/2 * dt;  a2 = -(1-c)/2 * dt
    b1 = -c   * dt;  b2 = -(1-2c)  * dt
    # prepare exponentials that apply the Laplacian
    exp_a₁L = @. cis(a1 * xh.∇²)
    exp_a₂L = @. cis(a2 * xh.∇²)
    exp_2a₁L = @. cis(2a1 * xh.∇²)

    apply_exp_p²!(u, uₚ, xh.ft, exp_a₁L)
    for it in 2:n_t
        @. u *= cis(b1 * (xh.U[1] + g[1] * abs2(u)))
        apply_exp_p²!(u, uₚ, xh.ft, exp_a₂L)
        @. u *= cis(b2 * (xh.U[1] + g[1] * abs2(u)))
        apply_exp_p²!(u, uₚ, xh.ft, exp_a₂L)
        @. u *= cis(b1 * (xh.U[1] + g[1] * abs2(u)))
        
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
    return GPESolution(ts[save_indices], us)
end

"""
Propagate `u` in real time using 2nd order Strang splitting
    exp(ℎ𝑀/2)exp(ℎ𝐿/2)exp(ℎ𝑀/2)
for nonzero vector potential.
The middle exponent with the linear operator 𝐿 = 𝑟(𝐩 - 𝐀)² is also Strang-split into
    exp(ℎ𝑟(𝑝ˣ - 𝐴ˣ)²/2)exp(ℎ𝑟(𝑝ʸ - 𝐴ʸ)²)exp(ℎ𝑟(𝑝ˣ - 𝐴ˣ)²/2).
Each exponent is then applied as
    exp(ℎ𝑟(𝑝ʸ - 𝐴ʸ)²) = exp(i𝛬ʸ)exp(ℎ𝑟𝑝ʸ²)exp(-i𝛬ʸ),
which is an exact identity. For real-time propagation, 𝑟 = -i.
Save intermediate solutions to `us` and return a `GPESolution` object.
If 𝑈 is zero and the gauge is such that either 𝐴ˣ or 𝐴ʸ is zero (e.g. Landau gauge), it is better to choose 𝐴ˣ = 0 (computation will run slightly faster since multiplication by exp(±i𝛬ˣ - 𝑟ℎ𝑈/2) will be omitted).
If decay Γ nonzero, then the norm decays during the steps, and we get the lie-vs-_lie situation as in imaginary-time propagation. We use the freezing approach, but solving with decay has not been tested.
"""
function strang2_nonzeroA!(xh::XSpaceHamiltonian{R}, u, uₚ, us, g, 𝛬::AbstractMatrix{Function}, ts, save_indices, dt) where R
    # make views of `u` and `uₚ` reshaped into 2D to apply FFT (normaly reshaping would be done in `transform!`)
    u_grid = reshape(u, size(xh.ft.plan_forward)) 
    uₚ_grid = reshape(uₚ, size(xh.ft.plan_forward))
    # plans for 1D transforms of 2D data
    plan_forward_x = FFTW.plan_fft(u_grid, 1)
    plan_backward_x = FFTW.plan_ifft(u_grid, 1)
    plan_forward_y = FFTW.plan_fft(u_grid, 2)
    plan_backward_y = FFTW.plan_ifft(u_grid, 2)
    
    exp_pˣ² = @. cis(-dt/2 * xh.∇[1]^2)
    exp_pʸ² = @. cis(-dt * xh.∇[2]^2)

    # The first term of the sequence is cis(-ℎ𝑀/2) = cis(-ℎ(𝑈 + 𝑔|𝜓|²)/2), then goes cis(-𝛬ˣ). One could combine them into a single exponential.
    # However, we find a ~20% overall speed-up by pre-computing the exponential cis(-𝛬ˣ - ℎ𝑈/2), and keeping cis(-ℎ𝑔|𝜓|²/2) separately in the main loop.
    # This way the argument of the latter `cis` is usually less than π/4, resulting in a ~1.5x faster evaluation of `cis`.
    # Thus, we precompute the terms `cis_Λˣ_U` = cis(𝛬ˣ - ℎ𝑈/2) and `cis_Λˣ⁺_U` = cis(-𝛬ˣ - ℎ𝑈/2).
    # If not saving during the present step, then cis(-ℎ(𝑈 + 𝑔|𝜓|²)/2)cis(-𝛬ˣ) can be combined with cis(-ℎ(𝑈 + 𝑔|𝜓|²)/2)cis(+𝛬ˣ) yielding cis(-ℎ(𝑈 + 𝑔|𝜓|²)). Again we find that having cis(-ℎ𝑈) precomputed is faster
    _𝛬 = replace(𝛬, zero => Returns(zero(R))) # replace so that _𝛬 can be safely called with two arguments
    U_iszero = isnothing(xh.𝑈[1, 1]) && iszero(xh.Γ[1])
    Λˣ_U_bothzero = 𝛬[1] == zero && U_iszero
    cis_Λˣ_U = Vector{Complex{R}}(undef, (Λˣ_U_bothzero ? 0 : xh.B))
    cis_Λˣ⁺_U = similar(cis_Λˣ_U)
    if !Λˣ_U_bothzero
        cis_𝛬ˣ_𝑈(x, y) = cis(_𝛬[1](x, y) / xh.δ + (xh.𝑈[1, 1](x, y) - im*xh.Γ[1]/2) * -dt/2)
        sample!(cis_Λˣ_U, cis_𝛬ˣ_𝑈, xh.ft.xs)
        cis_𝛬ˣ⁺_𝑈(x, y) = cis(-_𝛬[1](x, y) / xh.δ + (xh.𝑈[1, 1](x, y) - im*xh.Γ[1]/2) * -dt/2)
        sample!(cis_Λˣ⁺_U, cis_𝛬ˣ⁺_𝑈, xh.ft.xs)
    end
    cis_Λˣ_Λʸ = Vector{Complex{R}}(undef, xh.B) # at least one of 𝛬ˣ, 𝛬ʸ is not zero, so cis((𝛬ˣ-𝛬ʸ)/𝛿) will always be needed.
    cis_𝛬ˣ_𝛬ʸ(x, y) = cis((_𝛬[1](x, y) - _𝛬[2](x, y)) / xh.δ)
    sample!(cis_Λˣ_Λʸ, cis_𝛬ˣ_𝛬ʸ, xh.ft.xs)

    cis_U = Vector{Complex{R}}(undef, (U_iszero ? 0 : xh.B))
    if !U_iszero
        cis_𝑈(x, y) = cis(-dt * (xh.𝑈[1, 1](x, y) - im*xh.Γ[1]/2))
        sample!(cis_U, cis_𝑈, xh.ft.xs)
    end
    
    n_t = length(ts)
    nsave = 2
    gdt_2 = -g[1] * dt / 2
    gdt = -g[1] * dt

    if Λˣ_U_bothzero
        @. u *= cis(gdt_2 * abs2(u))
    else
        @. u *= cis(gdt_2 * abs2(u)) * cis_Λˣ⁺_U
    end
    
    for it in 2:n_t
        apply_exp_p²!(u_grid, uₚ, uₚ_grid, exp_pˣ², plan_forward_x, plan_backward_x)
        @. u *= cis_Λˣ_Λʸ
        apply_exp_p²!(u_grid, uₚ, uₚ_grid, exp_pʸ², plan_forward_y, plan_backward_y)
        @. u *= conj(cis_Λˣ_Λʸ) # compiler optimises `conj` to no extra cost
        apply_exp_p²!(u_grid, uₚ, uₚ_grid, exp_pˣ², plan_forward_x, plan_backward_x)
        if it == save_indices[nsave]
            if Λˣ_U_bothzero
                @. u *= cis(gdt_2 * abs2(u))
            else
                @. u *= cis(gdt_2 * abs2(u)) * cis_Λˣ_U
            end
            us[nsave] = copy(u)
            nsave += 1
            if it < n_t
                if Λˣ_U_bothzero
                    @. u *= cis(gdt_2 * abs2(u))
                else
                    @. u *= cis(gdt_2 * abs2(u)) * cis_Λˣ⁺_U
                end
            end
        else
            if U_iszero
                @. u *= cis(gdt * abs2(u))
            else
                @. u *= cis(gdt * abs2(u)) * cis_U
            end
        end
    end
    return GPESolution(ts[save_indices], us)
end

"""
Apply the exponential exp(𝑝ⁱ²) to the x-space state `u`, by transforming to p-space (and using uₚ as a buffer).
Used during split-step integration with nonzero vector potential.
"""
function apply_exp_p²!(u_grid, uₚ, uₚ_grid, exp_pⁱ², plan_forward, plan_backward)
    mul!(uₚ_grid, plan_forward, u_grid)
    @. uₚ *= exp_pⁱ²
    mul!(u_grid, plan_backward, uₚ_grid)
    return
end

"""
Apply the exponential exp(𝐩²) to the x-space state `u`, by transforming to p-space (and using uₚ as a buffer).
Used during split-step integration with zero vector potential.
"""
function apply_exp_p²!(u, uₚ, ft::FourierTransformerX, exp_p²)
    transform!(uₚ, ft, u; direction=:forward)
    uₚ .*= exp_p²
    transform!(u, ft, uₚ; direction=:backward, normalise=true)
    return
end