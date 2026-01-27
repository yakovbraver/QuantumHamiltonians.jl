using XSpaceHamiltonians, AppleAccelerate

using Plots
plotlyjs()
theme(:dark, size=(600, 500))
CMAP = cgrad(:Spectral, rev=true);

Float = Float32 # operating type

function 𝑈(x::Real)
    (ϵ*cos(x) / (ϵ^2 + sin(x)^2))^2
end

function 𝑈₁(x::Real)
    (ϵ / (ϵ^2 + x^2))^2
end

ϵ::Float = 0.1

# plot potential
M = 50
N = 2M + 1
xlimits = (-π, π) .|> Float
xs = range(xlimits..., N)
plot(xs, 𝑈)

# diagonalise to get exact eigenstates
@time xh = XSpaceHamiltonian{:dense}([xlimits], 𝑈; basis=:cis, 𝑈_iseven=true, M);
@time diagonalize!(xh, nev=5);
xh.ε

stateno = 3
xs, ψ = make_eigenfunctions(xh; statenos=[stateno], nx=N)
plot(xs, real(ψ[:, 1, 1]))
plot(xs, imag(ψ[:, 1, 1]))
plot(xs, abs2.(ψ[:, 1, 1]))

### Use imaginary time to get eigenstates

# will converge to the first 3 lowest states in 𝑈, respectively:
guesses = [Returns(0.1), sin, cos]
guesses_iseven = [true, false, true]
# we use DE.LinearExponential, which is an exact solver (equivalent to diagonalisation) so a few large steps is enough to converge to full precision
T_max = 2 |> Float
dt = 1 |> Float
gs = 3 # guess number
@time sol = propagate(xh, guesses[gs]; 𝜓₀_iseven=guesses_iseven[gs], T_max, dt, itime=true)
v = sol.u[end]
get_ε_μ(xh, v)

xs, ψ = make_wavefunction(xh, v; nx=101)
plot(xs, real(ψ[:, 1]))


################ nonlinear ################

######## Free system ########

M = 50
N = 2M + 1
R = 5
xlimits = (-R, R) .|> Float

# diagonalise to get exact eigenstates
δ = √0.5 |> Float
@time xh = XSpaceHamiltonian{:dense}([xlimits], nothing; basis=:cis, M, δ)
@time diagonalize!(xh, nev=0);
xh.ε

stateno = 1
xs, ψ = make_eigenfunctions(xh; statenos=[stateno], nx=N)
plot(xs, real(ψ[:, 1, 1]))
plot(xs, imag(ψ[:, 1, 1]))
plot(xs, abs2.(ψ[:, 1, 1]))

### Use imaginary time to get eigenstates

g = 500 |> Float # nonlinearity
get_ε_μ(xh, xh.V[:, 1], [g;;])

𝜓₀(x) = one(Float)
p = 1/√(2R) |> Float # value of wf in the bulk (= ground state solution for the free case)
ξ = √(1/(p^2 * g)) |> Float # healing length
𝜓₀(x) = p * tanh(x/ξ) # soliton trial

T_max = 1e-1 |> Float
dt = 1e-4 |> Float
@time sol = propagate(xh, 𝜓₀, g; 𝜓₀_iseven=false, T_max, dt, itime=true)
v = sol.u[end]
get_ε_μ(xh, v, [g;;])

xs, ψ = make_wavefunction(xh, v; nx=N)
plot!(xs, real(ψ[:, 1]), ylims=(-0.5, 0.5))

######## Na23 soliton in harmonic potential (https://doi.org/10.1103/PhysRevLett.87.130402, https://arxiv.org/abs/cond-mat/0104549)

#### Prepare parameters

m = 3.8165e-26
aₛ = 2.5e-9
h = 6.62607015e-34
ħ = h / 2π
ω = 3.5 * 2π # 1D trap frequency
a₀ = √(ħ / (m*ω)) # [1/m] -- unit of length
α = 100 # omega_perp / ω ratio
τ = 1/ω # [s] unit of time

n_atoms = 1e4
g = 2 * α * (aₛ/a₀) * n_atoms |> Float # coefficient of nonlinearity
R = 11 # trap half-length, in units of a₀

δ = √0.5 |> Float # coefficient of the momentum term

𝑈(x::Real) = x^2 / 2

#### Imaginary time

N = 125 # number of points, must be odd for cis; 125 is very good
M = (N - 1) ÷ 2 # maximum harmonic number
xlimits = (-R, R) .|> Float
xs = range(xlimits..., N)
plot(xs, 𝑈)

@time xh = XSpaceHamiltonian{:dense}([xlimits], 𝑈; basis=:cis, 𝑈_iseven=true, M, δ)
diagonalize!(xh, nev=5)


𝜓₀(x) = one(Float)
p = 1/√(2R) |> Float # value of wf in the bulk (= ground state solution for the free case)
ξ = √(1/(p^2 * g)) |> Float # healing length
𝜓₀(x) = p * tanh(x/ξ) # soliton trial

T_max = 0.2 |> Float
dt = 1e-2 |> Float
@time sol = propagate(xh, 𝜓₀, g; ψ₀_iseven=false, T_max, dt, itime=true)
V = sol.u[end]
get_ε_μ(xh, V, [g;;])

xs, ψ = make_wavefunction(xh, V; nx=N)
plot(xs, real(ψ[:, 1]))
plot(xs, imag(ψ[:, 1]))

# real time

# ground state times soliton
L = xlimits[2] - xlimits[1]
dx = L / (M+1)
xs = range(xlimits[1]+dx, xlimits[2]-dx, M)

ψ₀ = real(ψ) .* tanh.(9 .* (xs .- 5)) |> vec
plot(xs, ψ₀)

𝜓₀(x) = p * tanh(x/ξ) # soliton trial
ψ₀ = vec(ψ)

T_max = 10 |> Float
dt = 1e-3 |> Float
nsaves = 500

# starting from x-space functions
@time sol = propagate(xh, real(ψ₀), g; T_max, dt, itime=false, nsaves)
@time sol = propagate(xh, real(ψ[:, 1]), g; T_max, dt, itime=false, nsaves)

# starting from p-space functions
@time sol = propagate(xh, V, [g;;]; T_max, dt, itime=false, nsaves, solver=nothing)

v = sol.u[end]
get_ε_μ(xh, v, [g;;])
xs, ψ = make_wavefunction(xh, v; nx=N)
plot(xs, abs2.(ψ[:, 1]))

U = make_map(sol)
ts = range(0, T_max, nsaves+1)
heatmap(xs, 0:T_max/nsaves:T_max, abs2.(U)', c=CMAP, xlabel="x", ylabel="t")

"return a 2D evolution map using the solution"
function make_map(sol)
    U = Matrix{eltype(sol.u[1])}(undef, N, length(sol.u))
    for i in axes(U, 2)
        _, U[:, i] = make_wavefunction(xh, sol.u[i]; nx=N)
    end
    return U
end