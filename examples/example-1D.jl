using XSpaceHamiltonians

using Plots, DelimitedFiles
plotlyjs()
theme(:dark, size=(600, 500))

### Harmonic potential

function 𝑈(x::Real)
    x^2
end

Float = Float32 # operating type

xlimits = (-2π, 2π) .|> Float
M = 100

dh = XSpaceHamiltonian{:dense}(𝑈, xlimits; isperiodic=true, 𝑈_iseven=true, M)
@time diagonalize!(dh, nev=5);
dh.ε

# For real Hamiltonian matrix (in momentum space), the eigenstates can be chosen real. But since we are using a complex cis basis, the coordinate-space functions are complex.
# However, if potential is even, the eigenstates have definite parity. E.g. ground state is even, and so can be expressed in terms of cos, i.e. the real part of cis.
# First excited state is odd, and so can be expressed in terms of sin, i.e. the imaginary part of cis.

stateno = 4
part = iseven(stateno-1) ? real : imag
xs, ψ_sin = make_eigenfunctions(dh; statenos=[stateno], nx=200)
plot(xs, 𝑈)
plot!(xs, part(ψ[:, 1, 1]) .+ dh.ε[stateno])

# Or we can solve using sine basis, yielding real coordinate-space eigenfunctions

dh = XSpaceHamiltonian{:dense}(𝑈, xlimits; isperiodic=false, M)
@time diagonalize!(dh, nev=5);
dh.ε

stateno = 4
xs, ψ = make_eigenfunctions(dh; statenos=[stateno], nx=200)
plot(xs, 𝑈)
plot!(xs, ψ[:, 1, 1] .+ dh.ε[stateno])


### Lattice potential

function 𝑈(x::Real)
    U₀*sin(x)^2 + (ϵ*cos(x) / (ϵ^2 + sin(x)^2))^2
end

Float = Float32 # operating type

U₀::Float = 50
ϵ::Float = 0.15

xlimits = (-2π, 2π) .|> Float
M = 50

dh = XSpaceHamiltonian{:dense}(𝑈, xlimits; isperiodic=true, 𝑈_iseven=true, M)
@time diagonalize!(dh, nev=10);
foreach(println, dh.ε)

# plot eigenfunctions
statenos = 1:4
xs, ψ = make_eigenfunctions(dh; statenos, nx=500)
plot(xs, 𝑈, label=false)
plot!(xs, abs2.(ψ[:, 1, :]) .+ dh.ε[statenos]', label=statenos) # in this system we can have (two-fold) degeneracies, so extracting real wf's is not so straightforward. Therefore, plotting densities

# study wanniers
@time compute_wanniers!(dh, targetlevels=1:4);
# dh.wanniers.pos
xs, ψ, w = make_wannierfunctions(dh; nx=500);
plot(xs, 𝑈, legend=false, ylims=(0, 1.05U₀));
# plot!(xs, abs2.(w) .+ dh.wanniers.E')
w_real = make_wanniers_real(w)
plot!(xs, w_real .+ dh.wanniers.E')

plot(xs, imag.(w[:, 1]))

# study one cell using quasimomentum

xlimits = (-π/2, π/2) .|> Float
dh = XSpaceHamiltonian{:dense}(𝑈, xlimits; isperiodic=true, 𝑈_iseven=true, M)
ncells = 4
qs = range(0, 1, ncells)
diagonalize!(dh, qs; nev=10)
dh.ε_q