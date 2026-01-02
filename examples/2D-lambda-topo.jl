# Analysing the system in https://doi.org/10.1103/dhkv-zvwg (https://arxiv.org/abs/2506.17096)
using XSpaceHamiltonians

using Plots, DelimitedFiles, LaTeXStrings
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_phase = cgrad(:RdBu_9);
theme(:dark, size=(600, 500))
Plots.default(colorbar_tickfontcolor=:white)

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
# function 𝛺₁(x::Real, y::Real)
#     Ω₀ * ( sin(k*(x+y)) - im*sin(k*(x-y)) )^2
# end

# function ∂ₓ𝛺₁(x::Real, y::Real)
#     Ω₀ * 2( sin(k*(x+y)) - im*sin(k*(x-y)) ) * ( k * cos(k*(x+y)) - im*k*cos(k*(x-y)) )
# end

# function ∂y𝛺₁(x::Real, y::Real)
#     Ω₀ * 2( sin(k*(x+y)) - im*sin(k*(x-y)) ) * ( k * cos(k*(x+y)) + im*k*cos(k*(x-y)) )
# end
#########

function 𝛺₂(x::Real, y::Real)
    ϵ * Ω₀ * (1 + ν/2 * cos(2k*x) + ν/2 * cos(2k*y))
end

function ∂ₓ𝛺₂(x::Real, y::Real)
    -ϵ * Ω₀ * k * ν * sin(2k*x)
end

function ∂y𝛺₂(x::Real, y::Real)
    -ϵ * Ω₀ * k * ν * sin(2k*y)
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

Aₓ = [𝐴ₓ(x, y) for x in xs, y in ys]
Ay = [𝐴y(x, y) for x in xs, y in ys]

x_window = 1:5:length(xs)
window = CartesianIndices((x_window, x_window))
fig = CairoMakie.Figure(size=(500, 500));
ax = CairoMakie.Axis(fig[1, 1], xlabel="x", ylabel="y", limits=(xs[1], xs[end], ys[1], ys[end]))
CairoMakie.arrows2d!(xs[x_window], ys[x_window], Aₓ[window], Ay[window], tiplength=5, lengthscale=0.05, shaftwidth=1, tipwidth=5, normalize=true)
fig

# Absolute value of 𝐴

plotlyjs()
A_abs = [sqrt(𝐴ₓ(x, y)^2 + 𝐴y(x, y)^2) for x in xs, y in ys]
surface(xs, ys, A_abs, zlims=(0, 3))

A_abs_cut = [sqrt(𝐴ₓ(x-0.25, 0.25)^2 + 𝐴y(x-0.25, 0.25)^2) for x in xs] # a cut, shifted by 0.25
plot(xs, A_abs_cut)

### Dark state diagonalisation

M = 30
ν = 0.95
@time xh = XSpaceHamiltonian{:dense}([𝜙;;], xlimits, ylimits; isperiodic=true, M, δ, 𝑈_iseven=[true;;], 𝐴_x=𝐴ₓ, 𝐴_y=𝐴y);

@time diagonalize!(xh, nev=5);
xh.ε

stateno = 1
xs, ys, ψ = make_eigenfunction(xh, stateno, 100, 100)
heatmap(xs, ys, abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs, ys, abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_phase)

### Full 3-component diagonalisation

Ω₀::Float = 2000
Γ₃::Float = 1e3
Δ::Float = 2000

𝛥(x, y) = -Δ

M = 50
𝑈 = [nothing nothing 𝛺₁
     nothing nothing 𝛺₂
     nothing nothing 𝛥] # only upper triangle is needed
𝑈_iseven = BitArray([0 0 0; 0 0 1; 0 0 1])

@time xh = XSpaceHamiltonian{:sparse}(𝑈, xlimits, ylimits; isperiodic=true, M, δ, 𝑈_iseven, Γ=[0, 0, Γ₃], fft_threshold=1e-3);

@time diagonalize!(xh, nev=5);
xh.ε

stateno = 5
@time xs, ys, ψ = make_eigenfunction(xh, stateno, 101, 101);

plot_comps(xs, ys, ψ)
# total density
heatmap(xs, ys, (abs2.(ψ[1])+abs2.(ψ[2]))', xlabel=L"x/w_0", ylabel=L"y/w_0", c=cmap_rainbow, title=L"|\psi_{1}|^2+|\psi_{2}|^2")

# Plot all components
function plot_comps(xs, ys, ψ)
    gr()
    theme(:dark, size=(600, 550*1.45))
    figs = [plot() for _ in 1:6]
    for i in 1:2:6
        c = (i+1) ÷ 2 # component number
        figs[i]   = heatmap(xs, ys, abs2.(ψ[c])', xlabel=L"x/w_0", ylabel=L"y/w_0", c=cmap_rainbow, title=L"|\psi_{%$c}|^2");
        figs[i+1] = heatmap(xs, ys, angle.(ψ[c])' ./ π, c=:viridis, xlabel=L"x/w_0", ylabel=L"y/w_0", title=L"\arg(\psi_{%$c})", cbar_title="phase ("*L"\pi"*" rad)", clims=(-1, 1));
    end
    plot(figs..., plot_title="Full solution, state no. $stateno, "*L"\epsilon=%$(ϵ),\ \Omega_{0}=%$(Int(Ω₀)), \Gamma=%$(Γ₃),"*"\n"*L"E="*"$(round(ComplexF64(xh.ε[stateno]), sigdigits=3))",
         plot_titlefontcolor=:white, plot_titlefontsize=12, layout=(3, 2))
end