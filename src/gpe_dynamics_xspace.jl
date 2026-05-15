"Convenience caller for the 1-component case, where `ψ₀` is an analytic function or an array representing the discretised function, `g` is a number, and `ψ₀_iseven` is a Bool."
function propagate(xh::XSpaceHamiltonian{R}, ψ₀::Union{Function, AbstractArray{<:Number}}, g::R;
                   T_max::R, dt::R, itime::Bool=false,
                   solver=(itime ? ODE.LawsonEuler(;krylov=true) : ODE.ETDRK4(;krylov=true)), nsaves::Integer=0) where {R}
    propagate(xh, [ψ₀], [g;;]; T_max, dt, itime, solver, nsaves)
end

"""
Propagate the time-dependent Schrödinger or Gross-Pitaevskii (with nonlinearity matrix `g`) equation (SE and GPE, respectively) for the initial wave function `ψ₀`.
`ψ₀` can be:
    1. A vector of x-space analytic functions (one for each component);
    2. A vector of etiher D-dimensional arrays or flattened vectors, one for each component, representing discretised x-space functions.
    3. An x-space flattened vector, such as one obtained during diagonalisation. This is the format using throughout the solving; inputs of types 1 and 2 are converted to this type.
Set `itime=true` for imaginary time propagation.
`solver` is a DifferentialEquations.jl Semilinear Split ODE Solvers from https://docs.sciml.ai/DiffEqDocs/stable/solvers/split_ode_solve/, with Krylov iteration necessarily enabled.
For imaginary-time GPE, the default is `LawsonEuler(;krylov=true)`, which is first order (and hence fast), but is sufficient when the time step is small.
For real-time GPE, the default is `ETDRK4(;krylov=true)`; lower order variants can also be used for quick results. `HochOst4` seems to conserve the norm even better, but is a bit slower.
Return the DifferentialEquations solution object. 
"""
function propagate(xh::XSpaceHamiltonian{R, T}, ψ₀::Union{AbstractVector{<:Function}, AbstractVector{<:Number}, AbstractVector{<:AbstractArray{<:Number}}}, g::AbstractMatrix{R};
                   T_max::R, dt::R, itime::Bool=false,
                   solver=(itime ? ODE.LawsonEuler(;krylov=true) : ODE.ETDRK4(;krylov=true)), nsaves::Integer=0) where {R, T}
    (;xlims, B, nc, ft) = xh

    # determine if equation is real (in x-space)
    eq_isreal = itime && T <: Real # below `eq_isreal` might change if initial state is complex

    # prepare the input wf
    if ψ₀ isa AbstractVector{<:Number} # `ψ₀` is already a flattened vector, but we might need to copy it into a complex array
        ψ₀_isreal = eltype(ψ₀) <: Real
        eq_isreal &= ψ₀_isreal
        ψ_input = ψ₀_isreal && !eq_isreal ? complex(ψ₀) : ψ₀  # if the passed initial is real but equation is not, then convert; otherwise take as-is
    elseif ψ₀ isa AbstractVector{<:AbstractArray{<:Number}} # then need to flatten each and put into a contiguous array
        ψ₀_arereal = all(eltype(ψ) <: Real for ψ in ψ₀)
        eq_isreal &= ψ₀_arereal
        ψ_input = Vector{eq_isreal ? R : Complex{R}}(undef, nc*B)
        for c in 1:nc
            copyto!(ψ_input, (c-1)B+1, ψ₀[c], 1, B) # copy `B` (= all) elements of `ψ₀[c]`, starting at 1st, to `ψ_input`, starting at element (c-1)B+1. `ψ₀[c]` might be D-dimensional, but `copyto!` automatically flattens it (i.e. treats it as a contiguous vector)
        end
    else # `ψ₀` a vector of analytic functions
        ψ₀_arereal = [ ψ([xlims[i][1] for i in eachindex(xlims)]...) isa Real for ψ in ψ₀ ]
        eq_isreal &= all(ψ₀_arereal)
        ψ_input = Vector{eq_isreal ? R : Complex{R}}(undef, nc*B)
        for c in 1:nc
            @views sample!(ψ_input[(c-1)B+1:c*B], ψ₀[c], ft.xs)
        end
    end

    # Initialise the coupling matrix with the appropriate sign and `im` factor.
    # The equation i∂ₜ𝜓 = 𝐻𝜓 + g|𝜓|²𝜓 is coded as ∂ₜ𝜓 = -i𝐻𝜓 -i𝑔|𝜓|²𝜓.
    # In imaginary time, the equation is ∂ₜ𝜓 = -𝐻𝜓 -𝑔|𝜓|²𝜓. The factor multiplying 𝐻 is `H_factor`
    if itime
        H_factor = -1
        g_input = -g
    else
        H_factor = -im
        g_input = -im * g
    end

    # initialise the problem
    tspan = (zero(R), T_max)
    # the split problem passes one shared parameter tuple to both the linear operator (`FunctionOperator`) and nonlinear part (encoded in `prob`)
    params = (g_input, xh, H_factor)
    xh_op = SciMLOperators.FunctionOperator(XSpaceHamiltonianGPE!, similar(ψ_input); u=ψ_input, p=params, isconstant=true)

    prob = ODE.SplitODEProblem(xh_op, gpe_1comp!, ψ_input, tspan, params) # use sepcialisation `ODEProblem{true, SciMLBase.FullSpecialize}` for production!

    if itime
        # prepare the callback that remormalises wf at every step
        condition = Returns(true) # condition is checked at the end of each time step; we want this to be always true
        cb = ODE.DiscreteCallback(condition, affect_xspace!) # will save every step before and after the callback (`save_positions=(true, true)`); docs say this is mandatory when change of `u` is discontinuous
        sol = ODE.solve(prob, solver; callback=cb, save_everystep=false, save_start=true, dt)
        for c in 1:xh.nc
            @views normalize!(sol.u[end][(c-1)xh.B+1:c*xh.B], xh)
        end
        return sol
    else
        # when `saveat` is set, saving happens at points `tspan[1]:saveat:tspan[2]`
        saveat = nsaves == 0 ? T_max+1 : (tspan[2] - tspan[1]) / nsaves
        return ODE.solve(prob, solver; save_everystep=false, save_start=true, dt, saveat)
    end
end

"""
Helper function for wrapping `XSpaceHamiltonian` as a SciMLOperator, representing the action |𝑤⟩ = 𝑎𝐻|𝑣⟩ during GPE solving.
`params[1]` containg GPE `g` matrix and is not used (but passed to ODE problem).
`params[2]` contains a `XSpaceHamiltonian` object.
`params[3]` contains a the number 𝑎, which is -1 for imaginary time problem, and `-im` for usual real-time propagation.
"""
function XSpaceHamiltonianGPE!(w, v, u, params, t)
    xh = params[2]
    H_factor = params[3]
    mul!(w, xh, v)
    w .*= H_factor
    return
end

function affect_xspace!(integrator)
    xh = integrator.p[2]
    for c in 1:xh.nc
        @views normalize!(integrator.u[(c-1)xh.B+1:c*xh.B], xh)
    end
    return
end

"""
Update the 𝑢′ vector of the nonlinear part of the 1-component GPE
    𝑢′ = 𝑔|𝑢|²𝑢
The 𝑔 must contain `im` (for real-time propagation) and the proper sign.
"""
function gpe_1comp!(du, u, params, t)
    g = params[1]
    @. du = g * abs2(u) * u
    return
end
