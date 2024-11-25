includet("../src/XSpaceHamiltonians.jl")
using .XSpaceHamiltonians

using Plots
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_cyclic = cgrad(:cyclic_mrybm_35_75_c68_n256);
theme(:dark, size=(600, 500))

function 𝑈(x, y; ϵ::Real)
    x -= 2
    y -= 2
    2(ϵ / (1 + ϵ^2*(x^2+y^2)))^2
end

function 𝐴_x(x, y; ϵ::Real)
    x -= 2
    y -= 2
    ϵ^2 * √(x^2+y^2) / (1 + ϵ^2*(x^2+y^2)) * sin(atan(y, x))
end

function 𝐴_y(x, y; ϵ::Real)
    x -= 2
    y -= 2
    -ϵ^2 * √(x^2+y^2) / (1 + ϵ^2*(x^2+y^2)) * cos(atan(y, x))
end

ϵ = 10f0

xlimits = (0, 4) .|> Float32
ylimits = (0, 4) .|> Float32

# plot potential
N = 2^6-1
M = 2N + 1
xs = range(xlimits[1], xlimits[2], M)
ys = range(ylimits[1], ylimits[2], M)
surface(xs, ys, (x, y) -> 𝑈(x, y; ϵ))
surface(xs, ys, (x, y) -> 𝐴_x(x, y; ϵ)^2 + 𝐴_y(x, y; ϵ)^2)

# Calculate
dh = DirichletHamiltonian(xlimits, ylimits; 𝑈=(x, y) -> 𝑈(x, y; ϵ), 𝐴_x=(x, y) -> 𝐴_x(x, y; ϵ), 𝐴_y=(x, y) -> 𝐴_y(x, y; ϵ), N)
dh.H[diagind(dh.H)] .= 0
heatmap(abs.(dh.H), yaxis=:flip)

using LinearAlgebra
@time f = eigen(Hermitian(dh.H));

xs, ys, ψ = make_wavefunction(dh, f.vectors[:, 1], M, M)
surface(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_cyclic)