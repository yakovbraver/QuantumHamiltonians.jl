# Analysing the system in https://doi.org/10.1103/dhkv-zvwg (https://arxiv.org/abs/2506.17096)
using XSpaceHamiltonians

using Plots, DelimitedFiles
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_phase = cgrad(:RdBu_9);
theme(:dark, size=(600, 500))

function 𝛺₁(x::Real, y::Real)
    Ω₀ * ( sin(k*(x+y)) - im*sin(k*(x-y)) )
end

function ∂ₓ𝛺₁(x::Real, y::Real)
    Ω₀ * ( k * cos(k*(x+y)) - im*k*cos(k*(x-y)) )
end

function ∂y𝛺₁(x::Real, y::Real)
    Ω₀ * ( k * cos(k*(x+y)) + im*k*cos(k*(x-y)) )
end

######## Using squared 𝛺₁
function 𝛺₁(x::Real, y::Real)
    Ω₀ * ( sin(k*(x+y)) - im*sin(k*(x-y)) )^2
end

function ∂ₓ𝛺₁(x::Real, y::Real)
    Ω₀ * 2( sin(k*(x+y)) - im*sin(k*(x-y)) ) * ( k * cos(k*(x+y)) - im*k*cos(k*(x-y)) )
end

function ∂y𝛺₁(x::Real, y::Real)
    Ω₀ * 2( sin(k*(x+y)) - im*sin(k*(x-y)) ) * ( k * cos(k*(x+y)) + im*k*cos(k*(x-y)) )
end
#########

function 𝛺₂(x::Real, y::Real)
    ϵ * Ω₀ * (1 + ν/2 * cos(2k*x) + ν/2 * cos(2k*y))
end

function ∂ₓ𝛺₂(x::Real, y::Real)
    -k * ν * sin(2k*x)
end

function ∂y𝛺₂(x::Real, y::Real)
    -k * ν * sin(2k*y)
end

function 𝜁(x::Real, y::Real)
    𝛺₁(x, y) / 𝛺₂(x, y)
end

function ∂ₓ𝜁(x::Real, y::Real)
    (∂ₓ𝛺₁(x, y) * 𝛺₂(x, y) - 𝛺₁(x, y) * ∂ₓ𝛺₂(x, y) ) / 𝛺₂(x, y)^2
end

function ∂y𝜁(x::Real, y::Real)
    (∂y𝛺₁(x, y) * 𝛺₂(x, y) - 𝛺₁(x, y) * ∂y𝛺₂(x, y) ) / 𝛺₂(x, y)^2
end

function 𝜙(x::Float, y::Float) where Float <: AbstractFloat
    (abs2(∂ₓ𝜁(x, y)) + abs2(∂y𝜁(x, y))) / (1+abs2(𝜁(x, y)))^2 / Float(2π)^2
end

function 𝐴ₓ(x::Float, y::Float) where Float <: AbstractFloat
    -imag(𝜁(x, y)' * ∂ₓ𝜁(x, y)) / (1+abs2(𝜁(x, y))) / Float(2π)
end

function 𝐴y(x::Float, y::Float) where Float <: AbstractFloat
    -imag(𝜁(x, y)' * ∂y𝜁(x, y)) / (1+abs2(𝜁(x, y))) / Float(2π)
end

Float = Float32 # operating type

Ω₀::Float = 1
ϵ::Float = 1
k::Float = 2π
ν::Float = 0.9
δ::Float = 1/2π # gradient coefficient

xlimits = (-0.5, 0.5) .|> Float
ylimits = (-0.5, 0.5) .|> Float

# plot beams

M = 50
N = 2M + 1
xs = range(xlimits[1], xlimits[2], N)
ys = range(ylimits[1], ylimits[2], N)
heatmap(xs, ys, abs∘𝛺₁, c=:viridis)
heatmap(xs, ys, 𝛺₂, c=:viridis)

# plot 𝜙

M = 30
N = 2M + 1
xs = range(xlimits[1], xlimits[2], N)
ys = range(ylimits[1], ylimits[2], N)

surface(xs, ys, 𝜙, zlims=(-2, 2), clims=(-2, 2))
ν::Float = 0.95
heatmap(xs, ys, 𝜙, c=cmap_rainbow)

# plot 𝐴

import CairoMakie

Aₓ = [𝐴ₓ(x, y) for x in xs, y in ys]
Ay = [𝐴y(x, y) for x in xs, y in ys]

x_window = 1:5:length(xs)
window = CartesianIndices((x_window, x_window))
fig = CairoMakie.Figure(size=(500, 500));
ax = CairoMakie.Axis(fig[1, 1], xlabel="x", ylabel="y", limits=(xs[1], xs[end], ys[1], ys[end]))
CairoMakie.arrows2d!(xs[x_window], ys[x_window], Aₓ[window], Ay[window], tiplength=5, lengthscale=0.05, shaftwidth=1, tipwidth=5, normalize=true)
fig

# Diagonalise periodic with 𝜙 and 𝐴

@time dh = DenseHamiltonian(xlimits, ylimits; isperiodic=true, M, δ, 𝑈=𝜙, 𝐴_x=𝐴ₓ, 𝐴_y=𝐴y);
@time dh = SparseHamiltonian(xlimits, ylimits; δ=1/2π, 𝑈=𝜙, 𝐴_x=𝐴ₓ, 𝐴_y=𝐴y, M=2M);
@time diagonalize!(dh, nev=20);

scatter(dh.ε, ylims=(0, 6), yticks=0:2.5:7)

stateno = 1
xs, ys, ψ = make_eigenfunction(dh, stateno, 100, 100)
heatmap(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_phase)

"Return wave functions of the two components"
function get_components(ψ, xs, ys)
    [ψ[I] / √(1 + abs2(𝜁(xs[I[1]], ys[I[2]]))) for I in CartesianIndices(ψ)],
    [ψ[I] * 𝜁(xs[I[1]], ys[I[2]]) / √(1 + abs2(𝜁(xs[I[1]], ys[I[2]]))) for I in CartesianIndices(ψ)]
end

ψ₁, ψ₂ = get_components(ψ, xs, ys)
surface(xs, ys, abs2.(ψ₁)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow, title="|ψ₁|²")
heatmap(xs, ys, abs2.(ψ₁)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow, title="|ψ₁|²")
heatmap(xs, ys, angle.(ψ₁)', xlabel="x/a", ylabel="y/a", c=cmap_phase, title="arg ψ₁")

surface(xs, ys, abs2.(ψ₂)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow, title="|ψ₂|²")
heatmap(xs, ys, abs2.(ψ₂)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow, title="|ψ₂|²")
heatmap(xs, ys, angle.(ψ₂)', xlabel="x/a", ylabel="y/a", c=cmap_phase, title="arg ψ₂")

# Diagonalise periodic with for various quasimomenta

M = 30
a = 0.5
xlimits = (-a/2, a/2) .|> Float
ylimits = (-a/2, a/2) .|> Float

nqx = 3
nqy = 3
qxs = range(0, 4π/a, nqx) .|> Float
qys = range(0, 4π/a, nqy) .|> Float
@time dh = DenseHamiltonian(xlimits, ylimits; isperiodic=true, M, δ, 𝑈=𝜙);
@time dh = DenseHamiltonian(xlimits, ylimits; isperiodic=true, M, δ, 𝑈=𝜙, 𝐴_x=𝐴ₓ, 𝐴_y=𝐴y);
@time diagonalize!(dh, qxs, qys, nev=5);

scatter(sort(vec(dh.ε_q)), ylims=(0, 6), yticks=0:2.5:7, xlims=(0, 200))

stateno = 1
iqx = 1; iqy = 1
xs, ys, ψ = make_eigenfunction(dh, stateno, 100, 100, iqx, iqy)
heatmap(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_phase)
surface(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)

ψ₁, ψ₂ = get_components(ψ, xs, ys)
surface(xs, ys, abs2.(ψ₁)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow, title="|ψ₁|²")
heatmap(xs, ys, abs2.(ψ₁)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow, title="|ψ₁|²")
heatmap(xs, ys, angle.(ψ₁)', xlabel="x/a", ylabel="y/a", c=cmap_phase, title="arg ψ₁")

surface(xs, ys, abs2.(ψ₂)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow, title="|ψ₂|²")
heatmap(xs, ys, abs2.(ψ₂)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow, title="|ψ₂|²")
heatmap(xs, ys, angle.(ψ₂)', xlabel="x/a", ylabel="y/a", c=cmap_phase, title="arg ψ₂")