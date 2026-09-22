#=
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Calculations for https://doi.org/10.1103/qlbm-9sps (https://arxiv.org/abs/2506.08683), originally done using GPELab. ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
=#

using QuantumHamiltonians, AppleAccelerate

using Plots, LaTeXStrings
plotlyjs()
theme(:dark, size=(600, 250))
CMAP = cgrad(:Spectral, rev=true);
include("helpers.jl")

Float = Float64 # operating type

# cylindrical trap
𝑉(x::Real, y::Real) = V₀ / (1 + V₀*exp(-b*(hypot(x, y) - ρ₀)))

################ Dark state analysis ################

function 𝑈(x::Real, y::Real)
    ρ² = x^2 + y^2
    2(ν/a)^2 * (ρ²/a^2)^(ν-1) / (1 + (ρ²/a^2)^ν)^2 + 𝑉(x, y)
end

function 𝐴ˣ(x::Real, y::Real)
    ρ = hypot(x, y)
    ν/a * (ρ/a)^(2ν-1) / (1 + (ρ/a)^2ν) * sin(atan(y, x))
end

function 𝐴ʸ(x::Real, y::Real)
    ρ = hypot(x, y)
    -ν/a * (ρ/a)^(2ν-1) / (1 + (ρ/a)^2ν) * cos(atan(y, x))
end

R::Float = 1.2
xlimits = (-R, R) 

V₀::Float = 1000
b::Float = 17
ρ₀::Float = 0.7

ν::Int = 1
a::Float = 0.5

######## Linear analysis ########

M = 64
xh_ds = XSpaceHamiltonian([xlimits, xlimits], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M)
@time diagonalize!(xh_ds; nev=3, verbose=true)
xh_ds.ε
stateno = 1
xs, ys, ψ₀_ds = make_eigenfunction(xh_ds, stateno)
plot_comps_2D(xs, ys, ψ₀_ds)

######## Non-linear analysis (GPE) ########

g_ds::Float = 443

# imaginary-time propagation
T_max = 1e-1 |> Float
dt = 1e-4 |> Float
# using exact diagonalisation result as a guess
@time sol = propagate(xh_ds, ψ₀_ds, [g_ds;;]; T_max, dt, itime=true, solver=QuantumHamiltonians.ODE_EXP.LawsonEuler(;krylov=true, m=5))
E, μ₀ = get_EμN(xh_ds, sol.u[end], [g_ds;;])
xs, ys, ψ_ds = make_wavefunction(xh_ds, sol.u[end])
plot_comps_2D(xs, ys, ψ_ds)

# 𝜇 evolution; take every second because energy is saved before normalisation and after; we only want after
μs = map(sol.u[1:2:end]) do u
    get_EμN(xh_ds, u, [g_ds;;])[2][1]
end
plot(μs)

# Newton-Raphson using the result of imaginary time as the starting guess
natoms::Float = 1
@time xs, ys, ψ_nr, μ_nr = find_stationary(xh_ds, ψ_ds, [g_ds;;], μ₀[1], natoms; abstol=1e-6, show_trace=Val(true))
E, μ₀ = get_EμN(xh_ds, ψ_nr, [g_ds;;])
xs, ys, ψ_ds = make_wavefunction(xh_ds, ψ_nr)
plot_comps_2D(xs, ys, ψ_ds)

################ 3-component analysis ################

function 𝛺₁(x::Real, y::Real)
    ρ = hypot(x, y)
    Ω₀ * (ρ/a)^ν * exp(-(ρ/w₀)^2) * cis(ν*atan(y, x))
end

function 𝛺₂(x::Real, y::Real)
    ρ = hypot(x, y)
    Ω₀ * exp(-(ρ/w₀)^2)
end

𝜁(x, y) = 𝛺₁(x, y)/𝛺₂(x, y)

Ω₀::Float64 = 1.23e7
w₀::Float64 = 2
𝛥::Float64 = 10Ω₀/a^ν
Γ₃::Float64 = 2.35e7
𝑈₃₃(x, y) = 𝑉(x, y) - 𝛥

######## Linear analysis ########

𝑈_3comp = [      𝑉  nothing  conj∘𝛺₁
           nothing        𝑉       𝛺₂
           nothing  nothing      𝑈₃₃]
# construct XSpaceHamiltonian, which is better than PSpaceHamiltonian for GPE analysis. And we can use larger `M`
xh_3comp = XSpaceHamiltonian([xlimits, xlimits], 𝑈_3comp; basis=:cis, M, Γ=[0, 0, Γ₃])
@time diagonalize!(xh_3comp, nev=3, verbose=true, ls_prec=:block_jacobi, ls_abstol=1e-9, ls_reltol=1e-9, tol=1e-5); # M=64: 39 s (BICGSTAB: 47 s)
xh_3comp.ε
xs, ys, ψ_3comp_linear = make_eigenfunction(xh_3comp, 1)
plot_comps_2D(xs, ys, ψ_3comp_linear)

# Alternatively, a dense matrix can be diagonalised for small M=16. (Meanwhile sparse doesn't make sense because matrix density is ~0.5.)
# ph_3comp = PSpaceHamiltonian{:dense}([xlimits, xlimits], 𝑈_3comp; basis=:cis, M=16, Γ=[0, 0, Γ₃])
# @time diagonalize!(ph_3comp; nev=3, verbose=true)

######## Nonlinear analysis (GPE) ########

g_3comp = Float[443 434 0
                434 423 0
                  0   0 0]

# Prepare initial guess for 3-component GPE: convert `ψ_ds` to components 1 and 2; fill component 3 with zeros
ψ₀_3comp = [ComplexF64[;;] for _ in 1:3]
ψ₀_3comp[1] = @. ψ_ds[1] / √(1+abs2(𝜁(xs, ys')))
ψ₀_3comp[2] = @. -𝜁(xs, ys') * ψ₀_3comp[1]
ψ₀_3comp[3] = zeros(ComplexF64, size(ψ₀_3comp[1]))
plot_comps_2D(xs, ys, ψ₀_3comp) # check
get_EμN(xh_3comp, vcat(vec.(ψ₀_3comp)...), g_3comp)

# Newton-Raphson. Linear solving is slow because it does 1-2K iterations -- a preconditioner is needed.
@time xs, ys, ψ_3comp_nr, μ_3comp_nr = find_stationary(xh_3comp, ψ₀_3comp, g_3comp, μ_nr[1], natoms; abstol=1e-3, show_trace=Val(true)) # abstol=1e-3: 300 s
E_3comp, μ_3comp, η_3comp = get_EμN(xh_3comp, ψ_3comp_nr, g_3comp, makereal=false)
xs, ys, ψ_3comp = make_wavefunction(xh_3comp, ψ_3comp_nr)
plot_comps_2D(xs, ys, ψ_3comp)
