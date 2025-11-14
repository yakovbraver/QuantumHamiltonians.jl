using XSpaceHamiltonians

using Plots, LaTeXStrings
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_phase = cgrad(:RdBu_9);
theme(:dark, size=(600, 500))

function 𝑈(x::Real, y::Real)
    x == 0 && return x # return zero because shift direction is undefined
    y == 0 && return y
    # shift so that the quadrant of interest is centered at (0, 0)
    x -= sign(x) * o
    y -= sign(y) * o
    return 2(ϵ / (1 + ϵ^2*(x^2+y^2)))^2
end

function 𝐴_x(x::Real, y::Real)
    x == 0 && return x
    y == 0 && return y
    x -= sign(x) * o
    y -= sign(y) * o
    return ϵ^2 * √(x^2+y^2) / (1 + ϵ^2*(x^2+y^2)) * sin(atan(y, x))
end

function 𝐴_y(x::Real, y::Real)
    x == 0 && return x
    y == 0 && return y
    x -= sign(x) * o
    y -= sign(y) * o
    return -ϵ^2 * √(x^2+y^2) / (1 + ϵ^2*(x^2+y^2)) * cos(atan(y, x))
end

Float = Float32 # operating type

o::Float = 2 # nodes of the lattice will be at (±o, ±o)
ϵ::Float = 10

xlimits = (-4, 4) .|> Float
ylimits = (-4, 4) .|> Float

# plot potential
M = 50
N = 2M + 1
xs = range(xlimits[1], xlimits[2], N)
ys = range(ylimits[1], ylimits[2], N)
heatmap(xs, ys, 𝑈, c=cmap_rainbow)
heatmap(ys, xs, 𝐴_x, c=:coolwarm)
heatmap(ys, xs, 𝐴_y, c=:coolwarm)
surface(xs, ys, (x, y) -> 𝐴_x(x, y)^2 + 𝐴_y(x, y)^2)

# Plot 𝐴 vector field
import CairoMakie

n = 3 # plot every nth arrow
window = CartesianIndices( (1:n:length(xs), 1:n:length(ys)) )
A_x = [𝐴_x(x, y) for x in xs, y in ys]
A_y = [𝐴_y(x, y) for x in xs, y in ys]
fig = CairoMakie.Figure(size=(800, 800), fontsize=30);
ax = CairoMakie.Axis(fig[1, 1], aspect=1, title=L"\vec{A}", xlabel=L"xs/w_0", ylabel=L"ys/w_0", limits=(xs[1], xs[end], ys[1], ys[end]))
CairoMakie.arrows!(xs[1:n:end], ys[1:n:end], A_x[window], A_y[window], arrowsize=5, lengthscale=0.5)
fig

# Calculate
dh = DenseHamiltonian(xlimits, ylimits; isperiodic=false, M, 𝑈, 𝐴_x, 𝐴_y)
dh = DenseHamiltonian(xlimits, ylimits; isperiodic=false, M, 𝑈)

@time diagonalize!(dh, nev=5);

stateno = 1
xs, ys, ψ = make_eigenfunction(dh, stateno, 100, 100)
surface(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_phase)

dh.ε