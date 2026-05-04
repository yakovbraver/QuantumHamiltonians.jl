using XSpaceHamiltonians, AppleAccelerate

using Plots
plotlyjs()
theme(:dark, size=(600, 500))

#=
╔════════════════════╗
║ Harmonic potential ║
╚════════════════════╝
=#

function 𝑈(x::Real)
    x^2 / 2
end

Float = Float64 # operating type

xlimits = (-10, 10) .|> Float
M = 64

@time ph = PSpaceHamiltonian{:dense}([xlimits], 𝑈; basis=:cis, 𝑈_iseven=true, M, δ=Float(√0.5));
@time diagonalize!(ph, nev=5);
ph.ε

# For real Hamiltonian matrix (in momentum space), the eigenstates can be chosen real. But since we are using a complex cis basis, the coordinate-space functions are complex.
# However, if potential is even, the eigenstates have definite parity. E.g. ground state is even, and so can be expressed in terms of cos, i.e. the real part of cis.
# First excited state is odd, and so can be expressed in terms of sin, i.e. the imaginary part of cis.

stateno = 5
part = iseven(stateno-1) ? real : imag
xs, ψ = make_eigenfunctions(ph; statenos=[stateno], nx=2M+1)
plot(xs, 𝑈)
plot!(xs, part(ψ[:, 1, 1]) .+ ph.ε[stateno])

# Or we can solve using the sine basis, yielding real coordinate-space eigenfunctions
ph = PSpaceHamiltonian{:dense}([xlimits], 𝑈; basis=:sin, M, δ=Float(√0.5))
@time diagonalize!(ph, nev=5, verbose=true);
ph.ε

stateno = 1
@time xs, ψ = make_eigenfunctions(ph; statenos=[stateno], nx=200);
plot(xs, 𝑈)
plot!(xs, ψ[:, 1, 1] .+ ph.ε[stateno])

### Diagonalisation in x-space
xh = XSpaceHamiltonian([xlimits], 𝑈; basis=:cis, M=128, δ=Float(√0.5))
xh = XSpaceHamiltonian([xlimits], 𝑈; basis=:cos, M=128, δ=Float(√0.5))
xh = XSpaceHamiltonian([xlimits], 𝑈; basis=:sin, M=127, δ=Float(√0.5))
@time vals, vecs, info = diagonalize(xh; nev=5); # for large M you may need to tweak solver parameters. E.g. for M = 256, to converge to the default Float64 tolerance of 1e-12, set `krylovdim=35` (default is 30)
vals
info

stateno = 5
plot(xh.ft.xs, 𝑈)
plot!(xh.ft.xs, -vecs[stateno][1] .+ vals[stateno])

#=
╔═════════════════════════════════════════════════════════════════╗
║ Sine-lattice potential (repeating TTSC.jl/SineModel/spatial.jl) ║
╚═════════════════════════════════════════════════════════════════╝
=#

function 𝑈(x::Real)
    gₗ*cos(2x)^2 + Vₗ*cos(x)^2
end

Float = Float64 # operating type

gₗ::Float = -7640
Vₗ::Float = -2

ncells = 4
xlimits = (-π*ncells/2, π*ncells/2) .|> Float
xlimits = (0, π*ncells) .|> Float
M = 32ncells * 2

ph = PSpaceHamiltonian{:dense}([xlimits], 𝑈; basis=:cis, 𝑈_iseven=true, M)

@time diagonalize!(ph, nev=0);
scatter(ph.ε[1:M])

# energies of a certain band
targetband = 27
nsubbands = 2
targetlevels = (targetband-1)*2ncells+1:targetband*2ncells
scatter(ph.ε[targetlevels])

# plot eigenfunctions
xs, ψ = make_eigenfunctions(ph; statenos=targetlevels, nx=ncells*1000)
plot(xs, 𝑈, label=false)
plot!(xs, abs2.(ψ[:, 1, :]) .+ ph.ε[targetlevels]') 

# study wanniers
@time compute_wanniers!(ph; targetlevels);
ph.wanniers.pos
xs, ψ, w = make_wannierfunctions(ph; nx=500ncells);
plot(xs, 𝑈);
w_real = make_wanniers_real(w)
plot!(xs, w_real .+ ph.wanniers.E')

# visually compare real and imaginary parts
plot(xs, real.(w[:, 3]))
plot!(xs, imag.(w[:, 3]))

H_TB = compute_tb_hamiltonian(ph)

# study one cell using quasimomentum

xlimits = (0, π) .|> Float
xh_q = PSpaceHamiltonian{:dense}([xlimits], 𝑈; basis=:cis, 𝑈_iseven=true, M)
qlimits = (0, 2)
dq = qlimits[2]/ncells
qs = range(qlimits[1], qlimits[2]-dq, ncells)
@time diagonalize!(xh_q, [qs]; nev=0)
fig = plot();
for iq in 1:ncells
    scatter!(xh_q.ε_q[1:M÷ncells, iq], c=iq, legend=false)
end
fig

# energies of a certain band
targetband = 27
targetlevels = (2targetband-1):2targetband
xh_q.ε_q[targetlevels, :]

# plot eigenfunctions
iqxs = 1:ncells
xs, ψ = make_eigenfunctions(xh_q; statenos=[2targetband], nx=ncells*1000, iqxs)
plot(xs, 𝑈, label=false)
plot!(xs, abs2.(ψ[:, 1, :]) .+ xh_q.ε_q[2targetband, iqxs]')