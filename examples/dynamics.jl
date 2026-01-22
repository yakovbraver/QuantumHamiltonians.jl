using XSpaceHamiltonians

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

# diagonalise
@time xh = XSpaceHamiltonian{:dense}([xlimits], 𝑈; basis=:cis, 𝑈_iseven=true, M);
@time diagonalize!(xh, nev=5);

xh.ε

stateno = 2
xs, ψ = make_eigenfunctions(xh; statenos=[stateno], nx=N)
plot(xs, real(ψ[:, 1, 1]))
plot(xs, imag(ψ[:, 1, 1]))
plot(xs, abs2.(ψ[:, 1, 1]))

### 

𝜓₀(x) = cos(x)
T_max = 1
dt = 1e-2
@time sol = XSpaceHamiltonians.evolve_imag(xh, 𝜓₀; 𝜓₀_iseven=true, T_max, dt, solver=XSpaceHamiltonians.DE.MagnusGauss4())
v = sol.u[end]
dot(v, xh.H, v)

𝜓₀(x) = 0.1
T_max = 100
dt = 1
@time sol = XSpaceHamiltonians.evolve_imag(xh, 𝜓₀; 𝜓₀_iseven=true, T_max, dt);

v = sol.u[end]
dot(v, xh.H, v)

xs, ψ = make_wavefunction(xh, v; nx=101)
plot(xs, real(ψ[:, 1]))
plot!(xs, imag(ψ[:, 1]))
plot(xs, abs2.(ψ[:, 1]))
