#=
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║ Calculations for https://doi.org/10.1103/xq58-614t (https://arxiv.org/abs/2603.28876) ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
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
dt = 5e-5

# small
basis = :cis
M = get_M(basis, 8)
xlimits = (-π, π) .|> Float
dt = 2e-4

# full
basis = :cis # use cis for periodic, sin for hard walls
M = get_M(basis, 8)
xlimits = (-3π, 3π) .|> Float
dt = 2e-4

################ Dark-state system ################

@time phD = PSpaceHamiltonian{:dense}([xlimits], 𝑈; basis, 𝑈_iseven=true, M)

######## Linear analysis ########

@time diagonalize!(phD; nev=3)
phD.ε
Xs, ψD = make_eigenfunction(phD, 1)
plot(Xs, real.(ψD[1]))

######## Non-Linear analysis (GPE) ########

g = 100 |> Float # nonlinearity

# Ground state using Newton-Raphson, under the constraint of total number of atoms
μ₀ = 36 |> Float # initial guess
natoms = 1 |> Float # wave function is normalised to unity; actual number of atoms is incorporated into `g`
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

ph = PSpaceHamiltonian{:dense}([xlimits], 𝑉; basis, M)

######## Linear analysis ########

@time diagonalize!(ph; nev=3)
ph.ε # should agree with `phD.ε`
Xs, ψ = make_eigenfunction(ph, 1)
plot_comps(Xs, ψ)
plot_comps_complex(Xs, ψ) # in the :cis case, plot complex

######## Non-Linear analysis ########

# Rb-87 case
g = [100.4   98.006 0
      98.006 95.44  0
       0      0     0] .|> Float
# Manakov case close to Rb-87
g = [100 100 0
     100 100 0
       0   0 0] .|> Float

# make trial wf from the dark-state wf
ψ1 = @.( ψD / √(1 + 1/ϵ^2 * sin(xs)^2) ) |> vec
ψ2 = @.( -ψ1 * 1/ϵ * sin(xs) ) |> vec

Ψ₀ = [ψ1, ψ2, zeros(Float, length(ψ1))]
# Newton-Raphson under the constraint that the total number of particles shoud be equal to 1; using `μD` from above as the initial guess
@time xs, ψ_3comp, μ_3comp = find_stationary(ph, Ψ₀, g, μD, natoms; searchreal=true, abstol=1e-10, show_trace=Val(true))
E, μs, η_3comp = get_EμN(ph, ψ_3comp, g, state_is_pspace=false)
plot_comps(xs, ψ_3comp)

######## With decay ########

phΓ = PSpaceHamiltonian{:dense}([xlimits], 𝑉; basis, M, Γ=[0, 0, Γ₃])
diagonalize!(phΓ; nev=3)
phΓ.ε # real parts should agree with `phD.ε`. The imaginary part for the ground state is ~1e-7

# Run Newton-Raphson for 50 iterations. The solver works with real 𝜇; since actual is complex, it cannot reach abstol=1e-10. Will stop at abstol≈1e-5; this is L-inf norm, meaning the imaginary part of 𝜇 does not exceed 1e-3
xs, ψΓ, μΓ = find_stationary(phΓ, ψ_3comp, g, μD, natoms; maxiters=50, abstol=1e-10, show_trace=Val(true))
E, μs, ηΓ = get_EμN(phΓ, ψΓ, g, state_is_pspace=false, makereal=false)
plot_comps_complex(xs, ψΓ)

################ Fields off ################

ph_2comp = PSpaceHamiltonian{:dense}([xlimits], fill(nothing, 2, 2); basis, 𝑈_iseven=trues(2, 2), M);
B = ph_2comp.B # how many points belong to each component; needed to pass only the first two

# quench dynamics
nsaves = 1000
T_max = 1 / τ |> Float # divide desired time in ms by τ. The paper uses 1000
@time sol_dynamics = propagate(ph_2comp, [ψ_3comp[1:B], ψ_3comp[B+1:2B]], g; T_max, dt, itime=false, nsaves)
get_EμN(ph_2comp, sol_dynamics.u[end], g)[1] / get_EμN(ph_2comp, sol_dynamics.u[1], g)[1] # relative change in energy (final/initial)

xs, U = make_map_comps(ph_2comp, sol_dynamics)
ts = sol_dynamics.t * τ
figs = [heatmap(xs./π, ts, abs2.(U[:, :, c])', c=CMAP, xlabel=L"x/\pi", ylabel=L"$t$ (ms)") for c in 1:ph_2comp.nc]
plot(figs..., layout=(1, ph_2comp.nc), link=:y)

######## Stationary state analysis (not relevant for hard-walls case) ########

# Find stationary state with the fields off, keeping the number of atoms `η` in the two commponents the same as in the fields-on 3-component case, and use the guess for μs from above
@time xs, ψ_2comp, μ_2comp_stationary = find_stationary(ph_2comp, ψ_3comp[1:2B], g, μ_3comp[1:2], η_3comp[1:2]; searchreal=true, show_trace=Val(true), abstol=1e-11)
plot_comps(xs, ψ_2comp)
get_overlap(ψ_3comp[1:2B], ψ_2comp, ph_2comp)

#### BdG ####

@time vals, vecs = bdg_spectrum(ph_2comp, ψ_2comp, g, μ_2comp_stationary) # for a single DB-soliton (cos case), which is stable, this usually yields a real array, even in the non-Manakov case
scatter(vals, legend=false, markersize=1, markerstrokewidth=0)

smallindx = findall(x -> abs(x) < 1e-2, vals) # as a check, find indices of values close to zero; should be a total of 6 (=3 pairs) due to symmetries. However, in the cos case, one of the pairs (related to translational symmetry) is not small enough due to the x-domain being too small
vals[smallindx]

# calculate overlap of the stationary state and the unstable mode
i = findmax(imag, vals)[2] # index of the unstable mode (the one with the largest imaginary part)
ψB = [vecs[1:B, i] .+ conj.(vecs[2B+1:3B, i]); vecs[B+1:2B, i] .+ conj.(vecs[3B+1:4B, i])] # construct a flattened state vector using `i`th column of `vecs`
ψB ./= √get_overlap(ψB, ψB, ph_2comp) # normalise
abs(get_overlap(ψ_2comp, ψB, ph_2comp))

#### Floquet BdG (use this in the single-period cis case) ####

L = xlimits[2] - xlimits[1]
qs = range(-π/L, π/L, 101)
nev = 0
@time vals, _ = bdg_spectrum(ph_2comp, ψ_2comp, g, μ_2comp_stationary, [qs]; nev=0) # partial diagonalisation yields noise; must use full (nev = 0). [100 seconds for 101 qs]
scatter(vals[:], legend=false, markersize=1, markerstrokewidth=0, xlabel="Re", ylabel="Im")

# Cross-check with p-space BdG if you want. Partial diagonalisation is also noisy and not reliable, so using full
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
T_max = 50 / τ |> Float # divide desired time in ms by τ
@time sol_dynamics = propagate(ph_2comp, [ψ_2comp[1:B], ψ_2comp[B+1:2B]], g; T_max, dt, itime=false, nsaves)
get_EμN(ph_2comp, sol_dynamics.u[end], g)[1] / get_EμN(ph_2comp, sol_dynamics.u[1], g)[1] # relative change in energy (final/initial)

xs, U = make_map_comps(ph_2comp, sol_dynamics)
ts = sol_dynamics.t * τ
figs = [heatmap(xs./π, ts, abs2.(U[:, :, c])', c=CMAP, xlabel=L"x/\pi", ylabel=L"$t$ (ms)") for c in 1:ph_2comp.nc]
plot(figs..., layout=(1, ph_2comp.nc))

#### Study the growth of the error between the evolved state and the initial (used in Manakov case for `xlimits = (-2π, 2π)`) ####

dx = xs[2] - xs[1]
Δψ = map(2:size(U, 2)) do it
    Δ = @. abs2(U[:, it]) - abs2(U[:, 1])
    sum(abs, Δ) * dx
end
plot(ts[2:end], Δψ, yaxis=:log, xlabel="t")

# do a linear fit
using LinearAlgebra: \
window = 100:600 # a window where the growth is approximately exponential (linear in log plot); these are the numbers of points, not the t-values
X = ts[window]
Y = Δψ[window] .|> log
O = [X ones(length(X))]
K = O \ Y # (K[1]*τ) should coincide with the unstable BdG eigenvalue
plot!(X, @. exp(K[1]*X + K[2]))