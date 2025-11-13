# Harmonic oscillator with a sharp needle at the centre
using XSpaceHamiltonians

using Plots, DelimitedFiles
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_phase = cgrad(:RdBu_9);
theme(:dark, size=(600, 500))

function 𝑈(x::Real, y::Real)
    U₀*((k*x)^2 + (k*y)^2)/2 + 2(k*ρ₀)^2 / (((k*x)^2 + (k*y)^2)/ρ₀ + (k*ρ₀)^2)^2
end

Float = Float32 # operating type

k::Float = 2π
U₀::Float = 5
ρ₀::Float = 0.05

δ::Float = 1

xlimits = (-0.75, 0.75) .|> Float
ylimits = (-0.75, 0.75) .|> Float

# plot 𝑈

M = 60
N = 2M + 1
xs = range(xlimits[1], xlimits[2], N)
ys = range(ylimits[1], ylimits[2], N)
heatmap(xs, ys, 𝑈, c=:viridis)
surface(xs, ys, 𝑈, c=:viridis)

# Diagonalise periodic

@time dh = DenseHamiltonian(xlimits, ylimits; isperiodic=true, M, 𝑈);
@time dh = SparseHamiltonian(xlimits, ylimits; M, 𝑈);
@time diagonalize!(dh, nev=5);

scatter(dh.ε, ylims=(0, 6), yticks=0:2.5:7)

stateno = 1
xs, ys, ψ = make_wavefunction(dh, stateno, 100, 100)
heatmap(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_phase)