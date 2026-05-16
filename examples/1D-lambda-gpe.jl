#=
╔═══════════════════════════════════════════════════╗
║ Calculations for https://arxiv.org/abs/2603.28876 ║
╚═══════════════════════════════════════════════════╝
=#
using QuantumHamiltonians, AppleAccelerate

using Plots, LaTeXStrings
plotlyjs()
theme(:dark, size=(600, 500))
CMAP = cgrad(:Spectral, rev=true);
include("helpers.jl")

Float = Float64 # operating type

m = 1.443e-25 # [kg]
h = 6.62607015e-34 # [J⋅s]
ħ = h/2π # [J⋅s]
λ = 780e-9 # [m], 5 ²P3/2 -> 5 ²S1/2 transition
k = 2π/λ # [1/m]
Er = ħ^2 * k^2 / 2m # recoil energy [J]
τ = ħ/Er * 1e3 # time unit [ms]
ωr = Er/ħ # frequency unit [Hz]
Γ₃ = 38.1e6 / ωr # decay rate of 38.1 MHz converted to dimensionless

function 𝑈(x::Real)
    (ϵ*cos(x) / (ϵ^2 + sin(x)^2))^2
end

ϵ::Float = 0.1

### Basis and limits, choose as needed

# single
basis = :cos
M = get_M(basis, 7)
xlimits = (-π/2, π/2) .|> Float

# small
basis = :cis
M = get_M(basis, 8)
xlimits = (-π, π) .|> Float

# full
basis = :sin # use cis for periodic, sin for hard walls
M = get_M(basis, 8)
xlimits = (-3π, 3π) .|> Float

################ Dark-state system ################

@time phD = PSpaceHamiltonian{:dense}([xlimits], 𝑈; basis, 𝑈_iseven=true, M)

g = 100 |> Float # nonlinearity

# Ground state using Newton-Raphson, under the constraint of total number of atoms
μ₀ = 36 |> Float # initial guess
natoms = 1 |> Float
@time xs, ψD, μD = find_stationary(phD, [one], [g;;], μ₀, natoms; searchreal=true, maxiters=100, abstol=1e-10, show_trace=Val(true))
E, μs, η = get_EμN(phD, ψD, [g;;], state_is_pspace=false)
plot(xs, ψD)

################ 3-component system ################

𝛺₁(x) = Ω₁₀/ϵ*sin(x)
𝛺₂(x) = Ω₁₀

Ω₁₀::Float = 2000

𝑉 = [nothing nothing 𝛺₁
     nothing nothing 𝛺₂
     nothing nothing nothing]

# @time ph = PSpaceHamiltonian{:dense}([xlimits], 𝑉; basis, M, Γ=[0, 0, Γ₃])
@time ph = PSpaceHamiltonian{:dense}([xlimits], 𝑉; basis, M)

# Rb-87 case
g = [100   98 0
      98 95.4 0
       0    0 0] .|> Float
# Manakov case close to Rb-87
g = [100 100 0
     100 100 0
       0   0 0] .|> Float

# make trial wf from the dark-state wf
ψ1 = @.( ψD / √(1 + 1/ϵ^2 * sin(xs)^2) ) |> real |> vec 
ψ2 = @.( -ψ1 * 1/ϵ * sin(xs) ) |> real |> vec

Ψ₀ = [ψ1, ψ2, zeros(Float, length(ψ1))]

# Newton-Raphson under the constraint of the total number of particles equal to 1; using `μD` from above as the initial guess
@time xs, ψ_3comp, μ_3comp = find_stationary(ph, Ψ₀, g, μD, natoms; searchreal=true, abstol=1e-10, show_trace=Val(true))
E, μs, η_3comp = get_EμN(ph, ψ_3comp, g, state_is_pspace=false)
plot_comps(xs, ψ_3comp)

################ Fields off ################

ph_2comp = PSpaceHamiltonian{:dense}([xlimits], fill(nothing, 2, 2); basis, 𝑈_iseven=trues(2, 2), M);
B = size(ph_2comp.H, 1) ÷ 2 # how many points belong to each component; needed to pass only the first two

# quench dynamics
nsaves = 1000
T_max = 1 / τ |> Float # divide desired time in ms by τ. The paper uses 1000
dt = 2e-4 |> Float
@time sol_dynamics = propagate(ph_2comp, [ψ_3comp[1:B], ψ_3comp[B+1:2B]], g; T_max, dt, itime=false, nsaves)
get_EμN(ph_2comp, sol_dynamics.u[end], g)[1] / get_EμN(ph_2comp, sol_dynamics.u[1], g)[1] # relative change in energy (final/initial)

xs, U = make_map_comps(ph_2comp, sol_dynamics)
ts = sol_dynamics.t * τ
figs = [heatmap(xs./π, ts, abs2.(U[:, :, c])', c=CMAP, xlabel=L"x/\pi", ylabel=L"$t$ (ms)") for c in 1:ph_2comp.nc]
plot(figs..., layout=(1, ph_2comp.nc))

######## Stationary state analysis (not relevant for hard-walls case) ########

# Find stationary state with the fields off, keeping the number of atoms `η` in the two commponents the same as in the fields-on 3-component case, and use the guess for μs from above
@time xs, ψ_2comp, μ_2comp_stationary = find_stationary(ph_2comp, ψ_3comp[1:2B], g, μ_3comp[1:2], η_3comp[1:2]; searchreal=true, show_trace=Val(true), abstol=1e-11)
plot_comps(xs, ψ_2comp)

#### BdG ####

@time vals, _ = bdg_spectrum(ph_2comp, ψ_2comp, g, μ_2comp_stationary) # for a single DB-soliton (cos case), which is stable, this usually yields a real array, even in the non-Manakov case
# vals, _ = PSpaceHamiltonians.bdg_spectrum_pspace(ph_half, [ψ_2comp[1:B] ψ_2comp[B+1:2B]], g, μ_2comp_stationary)
scatter(vals, legend=false, markersize=1, markerstrokewidth=0)

smallindx = findall(x -> abs(x) < 1e-2, vals) # as a check, find indices of values close to zero; should be a total of 6 (=3 pairs) due to symmetries. However, in the cos case, one of the pairs (related to translational symmetry) is not small enough due to the x-domain being too small
vals[smallindx]

#### Floquet BdG (use this in the single-period cis case) ####

L = xlimits[2] - xlimits[1]
qs = range(-π/L, π/L, 101)
nev = 0
@time vals, _ = bdg_spectrum(ph_2comp, ψ_2comp, g, μ_2comp_stationary, [qs]; nev=0) # partial diagonalisation yields noise; must use full (nev = 0). [100 seconds for 101 qs]
scatter(vals[:], legend=false, markersize=1, markerstrokewidth=0, xlabel="Re", ylabel="Im")

# Cross-check with p-space BdG if you want. Partial diagonalisation is also noisy and not reliable
using ProgressMeter
ph_half = PSpaceHamiltonian{:dense}([xlimits], fill(nothing, 2, 2); basis, 𝑈_iseven=trues(2, 2), M=M÷2);
data = Vector{Complex{Float}}(undef, 2size(ph_half.H, 1)*length(qs))
@showprogress for (iq, q) in enumerate(qs)
    vals, _ = QuantumHamiltonians.bdg_spectrum_pspace(ph_half, [ψ_2comp[1:B] ψ_2comp[B+1:2B]], g, μ_2comp_stationary, [q]; nev=0)
    data[(iq-1)length(vals)+1:iq*length(vals)] = vals
end
scatter(data[:], legend=false, markersize=1, markerstrokewidth=0, xlabel="Re", ylabel="Im")

#### Dynamics of the fields-free stationary state ####

# quench dynamics
nsaves = 1000
T_max = 1 / τ |> Float # divide desired time in ms by τ
dt = 2e-4 |> Float
@time sol_dynamics = propagate(ph_2comp, [ψ_2comp[1:B], ψ_2comp[B+1:2B]], g; T_max, dt, itime=false, nsaves)
get_EμN(ph_2comp, sol_dynamics.u[end], g)[1] / get_EμN(ph_2comp, sol_dynamics.u[1], g)[1] # relative change in energy (final/initial)

xs, U = make_map_comps(ph_2comp, sol_dynamics)
ts = sol_dynamics.t * τ
figs = [heatmap(xs./π, ts, abs2.(U[:, :, c])', c=CMAP, xlabel=L"x/\pi", ylabel=L"$t$ (ms)") for c in 1:ph_2comp.nc]
plot(figs..., layout=(1, ph_2comp.nc))
