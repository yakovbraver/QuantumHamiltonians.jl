includet("../src/XSpaceHamiltonians.jl")
using .XSpaceHamiltonians

using Plots
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_cyclic = cgrad(:cyclic_mrybm_35_75_c68_n256);
theme(:dark, size=(600, 500))

function 𝑈(x::Real, y::Real)
    x -= 2
    y -= 2
    2(ϵ / (1 + ϵ^2*(x^2+y^2)))^2
end

function 𝐴_x(x::Real, y::Real)
    x -= 2
    y -= 2
    ϵ^2 * √(x^2+y^2) / (1 + ϵ^2*(x^2+y^2)) * sin(atan(y, x))
end

function 𝐴_y(x::Real, y::Real)
    x -= 2
    y -= 2
    -ϵ^2 * √(x^2+y^2) / (1 + ϵ^2*(x^2+y^2)) * cos(atan(y, x))
end

const ϵ = 10f0

xlimits = (0, 4) .|> Float32
ylimits = (0, 4) .|> Float32

# plot potential
N = 2^7-1
M = 2N + 1
xs = range(xlimits[1], xlimits[2], M)
ys = range(ylimits[1], ylimits[2], M)
surface(xs, ys, 𝑈)
surface(xs, ys, (x, y) -> 𝐴_x(x, y)^2 + 𝐴_y(x, y)^2)

# Calculate
dh = DirichletHamiltonian(xlimits, ylimits; 𝑈, 𝐴_x, 𝐴_y, N)

@time diagonalize!(dh, nev=5);

stateno = 1
xs, ys, ψ = make_wavefunction(dh, stateno, M, M)
surface(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_cyclic)

dh.ε[1]