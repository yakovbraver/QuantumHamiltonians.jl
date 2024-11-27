includet("../src/XSpaceHamiltonians.jl")
using .XSpaceHamiltonians

using Plots, LaTeXStrings
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_cyclic = cgrad(:cyclic_mrybm_35_75_c68_n256);
theme(:dark, size=(600, 500))

# For the functions below, it is assumed that the arguments x, y are in [0, L].
# In the functions, we start by shifting x, y to [-L/2, L/2], which is a shift by l = L/2
# Then, we shift by l/2 so that the quadrant of interest is centered at (0, 0)

function 𝑈(x::Real, y::Real)
    x -= l
    x == 0 && return x
    y -= l
    y == 0 && return y
    x -= sign(x) * o
    y -= sign(y) * o
    return 2(ϵ / (1 + ϵ^2*(x^2+y^2)))^2
end

function 𝐴_x(x::Real, y::Real)
    x -= l
    x == 0 && return x
    y -= l
    y == 0 && return y
    x -= sign(x) * o
    y -= sign(y) * o
    return ϵ^2 * √(x^2+y^2) / (1 + ϵ^2*(x^2+y^2)) * sin(atan(y, x))
end

function 𝐴_y(x::Real, y::Real)
    x -= l
    x == 0 && return x
    y -= l
    y == 0 && return y
    x -= sign(x) * o
    y -= sign(y) * o
    return -ϵ^2 * √(x^2+y^2) / (1 + ϵ^2*(x^2+y^2)) * cos(atan(y, x))
end

const o = 2f0 # nodes of the lattice will be at (±o, ±o)
const ϵ = 10f0

xlimits = (-4, 4) .|> Float32
ylimits = (-4, 4) .|> Float32
const l = (xlimits[2] - xlimits[1]) / 2 # size of each quadrant

# shift so that limits go from 0 to L
xlimits = xlimits .+ l
ylimits = ylimits .+ l

# plot potential
N = 2^6
M = 2N + 2
xs = range(xlimits[1], xlimits[2], M)
ys = range(ylimits[1], ylimits[2], M)
heatmap(xs, ys, 𝑈, c=cmap_rainbow)
heatmap(ys, xs, 𝐴_x, c=:coolwarm)
heatmap(ys, xs, 𝐴_y, c=:coolwarm)
surface(xs, ys, (x, y) -> 𝐴_x(x, y)^2 + 𝐴_y(x, y)^2)

# Calculate
dh = DirichletHamiltonian(xlimits, ylimits; 𝑈, 𝐴_x, 𝐴_y, N)

@time diagonalize!(dh, nev=1);

stateno = 1
xs, ys, ψ = make_wavefunction(dh, stateno, 100, 100)
surface(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_cyclic)

using DelimitedFiles
writedlm("phi0_lg_lat.txt", transpose(ψ))

dh.ε

import CairoMakie

n = 3 # plot every nth arrow
window = CartesianIndices( (1:n:length(xs), 1:n:length(ys)) )
A_x = [𝐴_x(x, y) for x in xs, y in ys]
A_y = [𝐴_y(x, y) for x in xs, y in ys]
fig = CairoMakie.Figure(size=(800, 800), fontsize=30);
ax = CairoMakie.Axis(fig[1, 1], aspect=1, title=L"\vec{A}", xlabel=L"xs/w_0", ylabel=L"ys/w_0", limits=(xs[1], xs[end], ys[1], ys[end]))
CairoMakie.arrows!(xs[1:n:end], ys[1:n:end], A_x[window], A_y[window], arrowsize=5, lengthscale=0.5)
fig