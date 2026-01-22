using XSpaceHamiltonians
using LinearAlgebra: dot

using Plots
plotlyjs()
theme(:dark, size=(600, 500))

function 𝑈(x::Real)
    (ϵ*cos(x) / (ϵ^2 + sin(x)^2))^2
end

Float = Float32 # operating type

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

stateno = 2
xs, ψ = make_eigenfunctions(xh; statenos=[stateno], nx=N)
plot(xs, real(ψ[:, 1, 1]))
plot(xs, imag(ψ[:, 1, 1]))
plot(xs, abs2.(ψ[:, 1, 1]))

### Use imaginary time to get eigenstates

# will converge to the first 3 lowest states respectively:
guesses = [Returns(0.1), sin, cos]
guesses_iseven = [true, false, true]
# we use DE.LinearExponential, which is an exact solver, so we only need to calculate the wf at some large time moment (i.e. make one large step)
T_max = 1 # this is not really large compared to eigenenergy, but is in fact enough to "converge" to full precision
dt = 1
g = 1 # guess number
@time sol = XSpaceHamiltonians.evolve_imag(xh, guesses[g]; 𝜓₀_iseven=guesses_iseven[g], T_max, dt)
v = sol.u[end]
dot(v, xh.H, v) # energy

xs, ψ = make_wavefunction(xh, v; nx=101)
plot(xs, real(ψ[:, 1]))