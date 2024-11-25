includet("../src/XSpaceHamiltonians.jl")
using .XSpaceHamiltonians

using Plots, DelimitedFiles
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
theme(:dark, size=(600, 500))

function 𝑈(x::Real, y::Real; ϵ::Real, ϵc::Real, χ::Real)
    (sin(x+y)^2 + (ϵc*sin(x-y))^2) / 𝛼(x, y; ϵ, ϵc, χ)^2 * 2ϵ^2 * (1+ϵc^2)
end

function 𝐴_x(x::Real, y::Real; ϵ::Real, ϵc::Real, χ::Real)
    sin(2y) .* ϵc .* sin(χ) ./ 𝛼(x, y; ϵ, ϵc, χ)
end

function 𝐴_y(x::Real, y::Real; ϵ::Real, ϵc::Real, χ::Real)
    sin(2x) .* ϵc .* sin(χ) ./ 𝛼(x, y; ϵ, ϵc, χ)
end

function ∇𝐴(x::Real, y::Real; ϵ::Real, ϵc::Real, χ::Real)
    ((-2ϵc*cos(χ)sin(2x) + ϵc^2 * sin(2(x-y)) + sin(2(x+y))) * ϵc * sin(2y) * sin(χ) +
     (-2ϵc*cos(χ)sin(2x) - ϵc^2 * sin(2(x-y)) + sin(2(x+y))) * ϵc * sin(2x) * sin(χ)) /
    𝛼(x, y; ϵ, ϵc, χ)^2
end

function 𝛼(x::Real, y::Real; ϵ::Real, ϵc::Real, χ::Real)
    η₋ = cos(x-y); η₊ = cos(x+y)
    return ϵ^2 * (1 + ϵc^2) + η₊^2 + (ϵc*η₋)^2 - 2ϵc*η₊*η₋*cos(χ)
end

ϵ = 0.1f0
ϵc = 1f0

########## χ = 0

χ = 0f0

xlimits = (0, π) .|> Float32
ylimits = (0, π) .|> Float32
N = 2^5-1
dh = DirichletHamiltonian(xlimits, ylimits; 𝑈=(x, y) -> 𝑈(x, y; ϵ, ϵc, χ), N)

heatmap(dh.H, yaxis=:flip)

using LinearAlgebra
f = eigen(dh.H)

xs, ys, ψ = XSpaceSSE.make_wavefunction(dh, f.vectors[:, 1], 50, 50)
surface(xs, ys, abs2.(ψ)')

########## χ = π/2, x from -pi/2 to pi/2

χ = π/2 |> Float32

# define U with a shift so that the are of interest is bounded by [0, Lx] x [0, Ly]
# this is because the basis is such that it is zero at 0, Lx, 0, and Ly.
function 𝑈(x::Real, y::Real; ϵ::Real, ϵc::Real, χ::Real)
    x -= pi/2
    (sin(x+y)^2 + (ϵc*sin(x-y))^2) / 𝛼(x, y; ϵ, ϵc, χ)^2 * 2ϵ^2 * (1+ϵc^2)
end

function 𝐴_x(x::Real, y::Real; ϵ::Real, ϵc::Real, χ::Real)
    x -= pi/2
    sin(2y) .* ϵc .* sin(χ) ./ 𝛼(x, y; ϵ, ϵc, χ)
end

function 𝐴_y(x::Real, y::Real; ϵ::Real, ϵc::Real, χ::Real)
    x -= pi/2
    sin(2x) .* ϵc .* sin(χ) ./ 𝛼(x, y; ϵ, ϵc, χ)
end

# the region of interest in the shifted potential
xlimits = (0, π) .|> Float32
ylimits = (0, π) .|> Float32

# plot potential
N = 2^6 - 1
M = 2N + 1
xs = range(xlimits[1], xlimits[2], M)
ys = range(ylimits[1], ylimits[2], M)
surface(xs, ys, (x, y) -> 𝑈(x, y; ϵ, ϵc, χ))
surface(xs, ys, (x, y) -> 𝐴_x(x, y; ϵ, ϵc, χ)^2 + 𝐴_y(x, y; ϵ, ϵc, χ)^2)

########## χ = π/2, FULL

χ = π/2 |> Float32

function 𝑈(x::Real, y::Real; ϵ::Real, ϵc::Real, χ::Real)
    (sin(x+y)^2 + (ϵc*sin(x-y))^2) / 𝛼(x, y; ϵ, ϵc, χ)^2 * 2ϵ^2 * (1+ϵc^2)
end

function 𝐴_x(x::Real, y::Real; ϵ::Real, ϵc::Real, χ::Real)
    sin(2y) .* ϵc .* sin(χ) ./ 𝛼(x, y; ϵ, ϵc, χ)
end

function 𝐴_y(x::Real, y::Real; ϵ::Real, ϵc::Real, χ::Real)
    sin(2x) .* ϵc .* sin(χ) ./ 𝛼(x, y; ϵ, ϵc, χ)
end

function ∇𝐴(x::Real, y::Real; ϵ::Real, ϵc::Real, χ::Real)
    ((-2ϵc*cos(χ)sin(2x) + ϵc^2 * sin(2(x-y)) + sin(2(x+y))) * ϵc * sin(2y) * sin(χ) +
     (-2ϵc*cos(χ)sin(2x) - ϵc^2 * sin(2(x-y)) + sin(2(x+y))) * ϵc * sin(2x) * sin(χ)) /
    𝛼(x, y; ϵ, ϵc, χ)^2
end

# the region of interest in the shifted potential
xlimits = (0, 2π) .|> Float32
ylimits = (0, 2π) .|> Float32

# plot potential
N = 2^6 - 1
M = 2N + 1
xs = range(xlimits[1], xlimits[2], M)
ys = range(ylimits[1], ylimits[2], M)
surface(xs, ys, (x, y) -> 𝑈(x, y; ϵ, ϵc, χ))
surface(xs, ys, (x, y) -> 𝐴_x(x, y; ϵ, ϵc, χ)^2 + 𝐴_y(x, y; ϵ, ϵc, χ)^2)

########## Calculate

dh = DirichletHamiltonian(xlimits, ylimits; 𝑈=(x, y) -> 𝑈(x, y; ϵ, ϵc, χ), 𝐴_x=(x, y) -> 𝐴_x(x, y; ϵ, ϵc, χ), 𝐴_y=(x, y) -> 𝐴_y(x, y; ϵ, ϵc, χ), N)
heatmap(dh.H, yaxis=:flip)

using LinearAlgebra
@time f = eigen(Hermitian(dh.H));

xs, ys, ψ = make_wavefunction(dh, f.vectors[:, 1], M, M)
heatmap(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_cyclic)
writedlm("exact_lambda_no1_256.txt", ψ, ',')
f.values[1]