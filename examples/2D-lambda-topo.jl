#=
╔══════════════════════════════════════════════════════════════════════════════════╗
║ Analysis of https://doi.org/10.1103/dhkv-zvwg (https://arxiv.org/abs/2506.17096) ║
╚══════════════════════════════════════════════════════════════════════════════════╝
=#
using XSpaceHamiltonians

using Plots, LaTeXStrings
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_phase = cgrad(:RdBu_9);
theme(:dark, size=(600, 500))
Plots.default(colorbar_tickfontcolor=:white)

function 𝛺₁(x::Real, y::Real)
    Ω₀ * ( sin(k*(x+y)) - im*sin(k*(x-y)) )
end

function ∂ˣ𝛺₁(x::Real, y::Real)
    Ω₀ * ( k * cos(k*(x+y)) - im*k*cos(k*(x-y)) )
end

function ∂ʸ𝛺₁(x::Real, y::Real)
    Ω₀ * ( k * cos(k*(x+y)) + im*k*cos(k*(x-y)) )
end

######## Using squared 𝛺₁
# function 𝛺₁(x::Real, y::Real)
#     Ω₀ * ( sin(k*(x+y)) - im*sin(k*(x-y)) )^2
# end

# function ∂ˣ𝛺₁(x::Real, y::Real)
#     Ω₀ * 2( sin(k*(x+y)) - im*sin(k*(x-y)) ) * ( k * cos(k*(x+y)) - im*k*cos(k*(x-y)) )
# end

# function ∂ʸ𝛺₁(x::Real, y::Real)
#     Ω₀ * 2( sin(k*(x+y)) - im*sin(k*(x-y)) ) * ( k * cos(k*(x+y)) + im*k*cos(k*(x-y)) )
# end
#########

function 𝛺₂(x::Real, y::Real)
    ϵ * Ω₀ * (1 + ν/2 * cos(2k*x) + ν/2 * cos(2k*y))
end

function ∂ˣ𝛺₂(x::Real, y::Real)
    -ϵ * Ω₀ * k * ν * sin(2k*x)
end

function ∂ʸ𝛺₂(x::Real, y::Real)
    -ϵ * Ω₀ * k * ν * sin(2k*y)
end

function 𝜁(x::Real, y::Real)
    𝛺₁(x, y) / 𝛺₂(x, y)
end

function ∂ˣ𝜁(x::Real, y::Real)
    (∂ˣ𝛺₁(x, y) * 𝛺₂(x, y) - 𝛺₁(x, y) * ∂ˣ𝛺₂(x, y) ) / 𝛺₂(x, y)^2
end

function ∂ʸ𝜁(x::Real, y::Real)
    (∂ʸ𝛺₁(x, y) * 𝛺₂(x, y) - 𝛺₁(x, y) * ∂ʸ𝛺₂(x, y) ) / 𝛺₂(x, y)^2
end

function 𝜙(x::Float, y::Float) where Float <: AbstractFloat
    (abs2(∂ˣ𝜁(x, y)) + abs2(∂ʸ𝜁(x, y))) / (1+abs2(𝜁(x, y)))^2 / Float(2π)^2
end

function 𝐴ˣ(x::Float, y::Float) where Float <: AbstractFloat
    -imag(𝜁(x, y)' * ∂ˣ𝜁(x, y)) / (1+abs2(𝜁(x, y))) / Float(2π)
end

function 𝐴ʸ(x::Float, y::Float) where Float <: AbstractFloat
    -imag(𝜁(x, y)' * ∂ʸ𝜁(x, y)) / (1+abs2(𝜁(x, y))) / Float(2π)
end

Float = Float64 # operating type

Ω₀::Float = 1 # the value plays no role in the dark-state approach
ϵ::Float = 1
k::Float = 2π
ν::Float = 0.98
δ::Float = 1/2π # gradient coefficient

xlimits = (-0.5, 0.5) .|> Float
ylimits = (-0.5, 0.5) .|> Float

# plot beams

M = 50
N = 2M + 1
xs = range(xlimits[1], xlimits[2], N)
ys = range(ylimits[1], ylimits[2], N)
heatmap(xs, ys, real∘𝛺₁, c=:viridis)
heatmap(xs, ys, imag∘𝛺₁, c=:viridis)
heatmap(xs, ys, abs∘𝛺₁, c=:viridis)
heatmap(xs, ys, 𝛺₂, c=:viridis)

# plot 𝜙

M = 50
N = 2M + 1
xs = range(xlimits[1], xlimits[2], N)
ys = range(ylimits[1], ylimits[2], N)

ν::Float = 0.95
surface(xs, ys, 𝜙, zlims=(-2, 2), clims=(-2, 2))
heatmap(xs, ys, 𝜙, c=cmap_rainbow, zlims=(-2, 2), clims=(-2, 2))

### plot 𝐴

import CairoMakie

Aˣ = [𝐴ˣ(x, y) for x in xs, y in ys]
Ay = [𝐴ʸ(x, y) for x in xs, y in ys]

x_window = 1:5:length(xs)
window = CartesianIndices((x_window, x_window))
fig = CairoMakie.Figure(size=(500, 500));
ax = CairoMakie.Axis(fig[1, 1], xlabel="x", ylabel="y", limits=(xs[1], xs[end], ys[1], ys[end]))
CairoMakie.arrows2d!(xs[x_window], ys[x_window], Aˣ[window], Ay[window], tiplength=5, lengthscale=0.05, shaftwidth=1, tipwidth=5, normalize=true)
fig

# Absolute value of 𝐴

plotlyjs()
A_abs = [sqrt(𝐴ˣ(x, y)^2 + 𝐴ʸ(x, y)^2) for x in xs, y in ys]
surface(xs, ys, A_abs, zlims=(0, 3))

A_abs_cut = [sqrt(𝐴ˣ(x-0.25, 0.25)^2 + 𝐴ʸ(x-0.25, 0.25)^2) for x in xs] # a cut, shifted by 0.25
plot(xs, A_abs_cut)

### Dark-state diagonalisation

M = 32
ν = 0.95
@time ph = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝜙, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M, δ, 𝑈_iseven=true); # 243s for M=64

@time diagonalize!(ph, nev=5); # 33s for M=64
ph.ε

stateno = 2
xs, ψ = make_eigenfunction(ph, stateno)
heatmap(xs[:, 1], xs[:, 2], abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs[:, 1], xs[:, 2], abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs[:, 1], xs[:, 2], angle.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_phase)

# Diagonalisation in x-space: faster than p-space
M = 128
@time xh = XSpaceHamiltonian([xlimits, ylimits], 𝜙, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M, δ);
# Setting `ishermitian=false` because solver claims that map is nonhermitian. TODO: investigate
@time vals, vecs, info = diagonalize(xh, nev=2, ishermitian=false, krylovdim=50); # M=64, krylovdim=50: 4.4s. M=128, krylovdim=50: 22s. [with AppleAccelerate]
vals
heatmap(xh.ft.xs[:, 1], xh.ft.xs[:, 2], abs2.(vecs[stateno].data[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xh.ft.xs[:, 1], xh.ft.xs[:, 2], angle.(vecs[stateno].data[1])', xlabel="x/a", ylabel="y/a", c=cmap_phase)

### Full 3-component diagonalisation

Ω₀::Float = 2000
Γ₃::Float = 1e3
Δ::Float = 2000

𝛥(x, y) = -Δ

M = 50
𝑈 = [nothing nothing 𝛺₁
     nothing nothing 𝛺₂
     nothing nothing 𝛥] # only upper triangle is needed
𝑈_iseven = trues(3, 3)

@time ph = PSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M, δ, 𝑈_iseven, Γ=[0, 0, Γ₃], fft_threshold=1e-3);
#@time ph = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cis, M, δ, 𝑈_iseven);
matrix_density(ph)

@time diagonalize!(ph, nev=5);
ph.ε

stateno = 1
@time xs, ψ = make_eigenfunction(ph, stateno);

plot_comps(xs, ψ)
# total density
heatmap(xs[:, 1], xs[:, 2], (abs2.(ψ[1])+abs2.(ψ[2]))', xlabel=L"x/w_0", ylabel=L"y/w_0", c=cmap_rainbow, title=L"|\psi_{1}|^2+|\psi_{2}|^2")

# Plot all components
function plot_comps(xs, ψ)
    gr()
    theme(:dark, size=(600, 550*1.45))
    figs = [plot() for _ in 1:6]
    for i in 1:2:6
        c = (i+1) ÷ 2 # component number
        figs[i]   = heatmap(xs[:, 1], xs[:, 2], abs2.(ψ[c])', xlabel=L"x/w_0", ylabel=L"y/w_0", c=cmap_rainbow, title=L"|\psi_{%$c}|^2");
        figs[i+1] = heatmap(xs[:, 1], xs[:, 2], angle.(ψ[c])' ./ π, c=:viridis, xlabel=L"x/w_0", ylabel=L"y/w_0", title=L"\arg(\psi_{%$c})", cbar_title="phase ("*L"\pi"*" rad)", clims=(-1, 1));
    end
    plot(figs..., layout=(3, 2))
end

# Experimental: diagonalisation in x-space. For M = 16, linear solving struggles to converge to sufficient accuracy, so eigenvalues cannot converge correctly. Tweaking krylovdim and maxiter does not help.
# Perhaps a preconditioner is needed. However, it is still enough to yield the lowest eigenvalue with at least 3 digits accuracy, and the eigenfunction looks correct.
# Still, this is too slow: 20s, while dense Arnoldi is 0.8s. The slowness comes from linear solving: solving for :SR with no inversion (invert=false) is fast but is not what we need.
@time ph = XSpaceHamiltonian([xlimits, ylimits], 𝑈; basis=:cis, M=16, δ);
@time vals, vecs, info = diagonalize(ph; nev=1);
info
vals
plot_comps(ph.ft.xs[:, 1], ph.ft.xs[:, 2], vecs[1].data)