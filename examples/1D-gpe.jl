using QuantumHamiltonians, AppleAccelerate

using Plots, LaTeXStrings
plotlyjs()
theme(:dark, size=(600, 500))
CMAP = cgrad(:Spectral, rev=true);
include("helpers.jl")

Float = Float64 # operating type

################ Linear ################

function 𝑈(x::Real)
    (ϵ*cos(x) / (ϵ^2 + sin(x)^2))^2
end

function 𝑈₁(x::Real)
    (ϵ / (ϵ^2 + x^2))^2
end

ϵ::Float = 0.1

# plot potential
basis = :cis
M = get_M(basis)
xlimits = (-π, π) .|> Float
xs = range(xlimits..., 100)
plot(xs, 𝑈)

# diagonalise to get exact eigenstates
@time ph = PSpaceHamiltonian{:dense}([xlimits], 𝑈; basis=:cis, 𝑈_iseven=true, M);
@time diagonalize!(ph, nev=5);
ph.ε

stateno = 1
xs, ψ = make_eigenfunctions(ph; statenos=[stateno], nx=100)
plot(xs, -real(ψ[:, 1, 1]))
plot(xs, imag(ψ[:, 1, 1]))
plot(xs, abs2.(ψ[:, 1, 1]))

######## Use imaginary time to get eigenstates

# will converge to the first 3 lowest states in 𝑈, respectively:
guesses = [one, sin, cos]
guesses_iseven = [true, false, true]
# we use DE.LinearExponential, which is an exact solver (equivalent to diagonalisation) so a few large steps is enough to converge to full precision
T_max = 2 |> Float
dt = 1 |> Float
gs = 1 # guess number
@time sol = propagate(ph, guesses[gs]; ψ₀_iseven=guesses_iseven[gs], T_max, dt, itime=true)
v = sol.u[end]
get_Eμη(ph, v)

xs, ψ = make_wavefunction(ph, v)
plot(xs, real(ψ[1]))

################ Nonlinear ################

######## GPE ground state in dark-state potential ########

g = 100 |> Float # nonlinearity

T_max = 5 |> Float
dt = 1e-4 |> Float
@time sol = propagate(ph, one, g; T_max, dt, itime=true)
V = sol.u[end]
get_Eμη(ph, V, [g;;])

xs, ψD = make_wavefunction(ph, V)
plot(xs, real(ψD[1]), ylims=(-0.5, 0.5))
plot!(xs, imag(ψD[1]), ylims=(-0.5, 0.5))

######## Free system ########

δ = √0.5 |> Float
R = 5
xlimits = (-R, R) .|> Float

basis = :cis
M = get_M(basis)

@time ph = PSpaceHamiltonian{:dense}([xlimits], nothing; basis, M, δ)
@time diagonalize!(ph, nev=0);
ph.ε

#### Noninteracting free particle dispersion ####

L = xlimits[2] - xlimits[1]

# cis
dp = 2π / L # as in the Laplacian and the exponents of the basis functions
ps = (-M:M) .* dp # alternatively, we can calculate `dx = L / 2M` and then `ps = range(-π/dx, π/dx, 2M+1)` 
scatter(ps, (δ.*ps).^2, xlabel="p", label="exact")
scatter!(ps[end÷2+1:end], ph.ε[1:2:end], label="numerics")

# cos
dp = π / L # this is the ground truth because this features in exponents of the basis functions and the Laplacian
ps = (0:M) .* dp
scatter(ps, (δ.*ps).^2, xlabel="p", label="exact")
scatter!(ps, ph.ε, label="numerics")

#### Interacting particle dispersion (Bogoliubov dispersion) ####

g = 500 |> Float # nonlinearity

μ = get_Eμη(ph, ph.V[:, 1], [g;;])[2]
xs, ψ = make_wavefunction(ph, ph.V[:, 1])
vals, vecs = bdg_spectrum(ph, real(ψ[1]), g, μ[1])

ω = -real(vals) |> sort

# plotting for the cis case
scatter(ps, ω[2:2:end], xlabel="p", label="numerics") # take every second ω to ignore degeneracy
n₀ = real(ψ[1][1])^2 # ground state density (wave function density at an arbitrary point)
ϵ_p = @. sqrt(2n₀*g*ps^2/2 + ps^4/4) # Pethick & Smith, (7.31)
scatter!(ps, ϵ_p.*sign.(ps), label="exact")

# plotting for the cos case
scatter([-reverse(ps); ps], ω, xlabel="q", label="numerics")
ϵ_p = @. sqrt(2n₀*g*ps^2/2 + ps^4/4) # Pethick & Smith, (7.31)
scatter!(ps, ϵ_p, label="exact")

#=
╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Soliton oscillations in a harmonic potential (https://doi.org/10.1103/PhysRevLett.84.2298, https://arxiv.org/abs/cond-mat/0001360) ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
=#

# Na-23 parameters similar to https://doi.org/10.1103/PhysRevLett.87.130402 (https://arxiv.org/abs/cond-mat/0104549)

m = 3.8165e-26
aₛ = 2.5e-9
h = 6.62607015e-34
ħ = h / 2π
ω = 3.5 * 2π # 1D trap frequency
a₀ = √(ħ / (m*ω)) # [1/m] -- unit of length
α = 100 # ω⟂ / ω ratio
τ = 1/ω # [s] unit of time

n_atoms = 1e4
g = 2 * α * (aₛ/a₀) * n_atoms |> Float # coefficient of nonlinearity
R = 11 # trap half-length, in units of a₀

δ = √0.5 |> Float # coefficient of the momentum term

𝑈(x::Real) = x^2 / 2

#### Imaginary time

basis = :cis
M = get_M(basis, 8)

xlimits = (-R, R) .|> Float
xs = range(xlimits..., 100)
plot(xs, 𝑈)

@time ph = PSpaceHamiltonian{:dense}([xlimits], 𝑈; basis, 𝑈_iseven=true, M, δ)
diagonalize!(ph, nev=5)
ph.ε

p = 1/√(2R) |> Float # value of wf in the bulk (= ground state solution for the free case)
ξ = √(1/(p^2 * g)) |> Float # healing length
𝜓₀(x) = p * tanh(x/ξ) # soliton trial

T_max = 1 |> Float
dt = 1e-4 |> Float
@time sol = propagate(ph, 𝜓₀, g; ψ₀_iseven=false, T_max, dt, itime=true)
V = sol.u[end]
E, μ₀ = get_Eμη(ph, V, [g;;])
xs, ψ = make_wavefunction(ph, V)
plot(xs, real(ψ[1]))
plot!(xs, imag(ψ[1]))

# using Newton-Raphson (undef fixed total number of particles)
natoms = 1.0
@time xs, sol = find_stationary(ph, [𝜓₀], [g;;], μ₀, natoms; show_trace=Val(true))
ψ = sol.u[1:end-1] # last element is the chemical potential
E, μ = get_Eμη(ph, ψ, [g;;], v_is_pspace=false)
plot!(xs, ψ)

#### Real-time propagation of a displaced soliton

# get ground state
natoms = 1.0
@time xs, sol = find_stationary(ph, [one], [g;;], μ₀, natoms; show_trace=Val(true))
ψ = sol.u[1:end-1] # last element is the chemical potential

# create a displaced soliton
ψ₀ = real(ψ) .* tanh.(9 .* (xs .- 5)) |> vec
plot(xs, ψ₀)
plot(xs, abs2.(ψ₀))

T_max = 10 |> Float
dt = 1e-3 |> Float
nsaves = 500

@time sol = propagate(ph, ψ₀, g; T_max, dt, itime=false, nsaves)

v = sol.u[end]
get_Eμη(ph, v, [g;;])
xs, ψ = make_wavefunction(ph, v)
plot(xs, abs2.(ψ[1]))

xs, U = make_map(ph, sol)
ts = range(0, T_max, nsaves+1)
heatmap(xs, 0:T_max/nsaves:T_max, abs2.(U)', c=CMAP, xlabel="x", ylabel="t")

#=
╔═════════════════════════════════════════════════════════════════════════════════════════╗
║ Soliton on a hill from https://doi.org/10.1093/oso/9780192843234.001.0001, Section 22.5 ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
=#

function 𝑈(x::Real)
    (Ω*x)^2 / 2 + B*sech(β*x)^2
end

Ω::Float = 0.075
B::Float = 0.3
β::Float = 0.5
δ = √0.5 |> Float # coefficient of the momentum term

basis = :cis
M = get_M(basis, 8)
R = 15
xlimits = (-R, R) .|> Float
xs = range(xlimits..., 100)
plot(xs, 𝑈)

@time ph = PSpaceHamiltonian{:dense}([xlimits], 𝑈; basis, 𝑈_iseven=true, M, δ)

# check out noninteracting eigenstates if you want
@time diagonalize!(ph, nev=5);
ph.ε
stateno = 1
xs, ψ = make_eigenfunctions(ph; statenos=[stateno], nx=100)
plot!(xs, real(ψ[:, 1, 1]) ./ 5 .+ ph.ε[stateno])
plot!(xs, imag(ψ[:, 1, 1]) ./ 5 .+ ph.ε[stateno])
plot(xs, abs2.(ψ[:, 1, 1]))

#### Get stationary state ####

𝜓₀(x) = sech(x)
g = -1 |> Float # nonlinearity
μ₀ = -1 |> Float
@time xs, sol = find_stationary(ph, [𝜓₀], [g;;], μ₀, show_trace=Val(true))
ψ_nln = sol.u
E, μ = get_Eμη(ph, sol.u, [g;;], v_is_pspace=false)
plot(xs, ψ_nln)

#### Calculate real-time dynamics ####

T_max = 200 |> Float
dt = 1e-3 |> Float
nsaves = 500

ψ_rand = ψ_nln .+ 1e-5 .* rand(length(ψ_nln))
@time sol = propagate(ph, [ψ_rand], [g;;]; T_max, dt, itime=false, nsaves, solver=QuantumHamiltonians.ODE.ETDRK4())

xs, U = make_map(ph, sol)
ts = range(0, T_max, nsaves+1)
heatmap(xs, 0:T_max/nsaves:T_max, abs2.(U)', c=CMAP, xlabel="x", ylabel="t")

#### Calculate BdG and compare with dynamics ####

@time vals, vecs = bdg_spectrum(ph, ψ_nln, g, μ₀);
scatter(vals, legend=false, markersize=2, markerstrokewidth=0)
maximum(imag, vals)

# Plot the growth of the error between the evolved state and the initial
dx = xs[2] - xs[1]
Δψ = map(2:size(U, 2)) do it
    Δ = @. abs2(U[:, it]) - abs2(U[:, 1])
    sum(abs, Δ) * dx
end
plot(ts[2:end], Δψ, yaxis=:log, xlabel="t")

# do a linear fit
using LinearAlgebra: \
window = 50:100 # a window where the growth is approximately exponential (linear in log plot); this are the numbers of points, not the t-values
X = ts[window]
Y = Δψ[window] .|> log
O = [X ones(length(X))]
K = O \ Y # K[1] should coincide with the unstable eigenvalue
plot!(X, @. exp(K[1]*X + K[2]))