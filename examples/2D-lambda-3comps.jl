# 3-component analysis of https://doi.org/10.1103/PhysRevA.107.033328 (https://arxiv.org/abs/2304.00302)
using XSpaceHamiltonians

using Plots, LaTeXStrings
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_phase = cgrad(:RdBu_9);
theme(:dark, size=(600, 500))

"Plot all components"
function plot_comps(xs, ys, ψ)
    gr()
    theme(:dark, size=(600, 550*1.5))
    figs = [plot() for _ in 1:6]
    for i in 1:2:6
        c = (i+1) ÷ 2 # component number
        figs[i]   = heatmap(xs, ys, abs2.(ψ[c])', xlabel=L"x/w_0", ylabel=L"y/w_0", c=cmap_rainbow, title=L"|\psi_{%$c}|^2");
        figs[i+1] = heatmap(xs, ys, angle.(ψ[c])' ./ π, c=:viridis, xlabel=L"x/w_0", ylabel=L"y/w_0", title=L"\arg(\psi_{%$c})", cbar_title="phase ("*L"\pi"*" rad)", clims=(-1, 1));
    end
    plot(figs..., plot_title="Full solution, state no. $stateno, "*L"\epsilon=%$(ϵ),\ \Omega_{10}=%$(Int(Ω₁₀)), \Gamma=%$(Γ₃),"*"\n"*L"E="*"$(round(ComplexF64(xh.ε[stateno]), sigdigits=3))",
         plot_titlefontcolor=:white, plot_titlefontsize=12, layout=(3, 2))
end

########## χ = 0 (real 𝛺₂)

function 𝛺₁(x::Real, y::Real)
    Ω₁₀ / 2
end

function 𝛺₂(x::Real, y::Real)
    ( -Ω₋ * cos(x-y) + Ω₊ * cos(x+y) ) / 2
end

Float = Float64 # operating type

ϵ::Float = 0.1
ϵc::Float = 1
Ω₁₀::Float = 2000
Ω₊ = Ω₁₀ / (ϵ*√(1+ϵc^2))
Ω₋ = Ω₊ * ϵc
Γ₃::Float = 1e3

# Use full period of 𝛺₂
xlimits = (-π, π) .|> Float
ylimits = (-π, π) .|> Float

# plot the coupling
M = 50
N = 2M + 1
xs = range(xlimits..., N)
ys = range(ylimits..., N)
heatmap(xs, ys, 𝛺₂, c=:viridis)

# Since 𝛺₂ only has the ±1st harmonic, M can be small.
# For M=15, the (real part of) ground state energy matches M=200 at 5 digits accuracy, wfs also match well, so dense calculation (even with full diagonalisation) is possible
M = 200
𝑈 = [nothing nothing 𝛺₁      
     nothing nothing 𝛺₂
     nothing nothing nothing] # only upper triangle is needed
@time xh = XSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=trues(3, 3), Γ=[0, 0, Γ₃])
@time xh = XSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=trues(3, 3), Γ=[0, 0, Γ₃], fft_threshold=1e-3)

@time diagonalize!(xh, nev=5);
xh.ε

# l = findfirst(x -> real(x) > 0, xh.ε) # find the dark state from full diagonalisation
# xh.ε[l]

stateno = 1
@time xs, ys, ψ = make_eigenfunction(xh, stateno, 101, 101);

plot_comps(xs, ys, ψ)

### Quasimomenta

M = 50 # something like M=30 is needed to get converged lowest band
@time xh = XSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=trues(3, 3), Γ=[0, 0, Γ₃])
# xh = XSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=trues(3, 3), Γ=[0, 0, Γ₃])
ncells = 11
P = xlimits[2] - xlimits[1]
qlimits = (-π/P, π/P)
qxs = range(qlimits..., ncells)
qys = [0.0]
@time diagonalize!(xh, [qxs, qys]; nev=4); # doing a cut for fixed 𝑞𝑦 = 0

fig = plot();
for n in axes(xh.ε_q, 1)
    scatter!(qxs, real.(xh.ε_q[n, :, 1]), c=n)
end
fig

########## χ = 1.4 (complex 𝛺₂)

function 𝛺₂_cis(x::Real, y::Real)
    ( -Ω₋ * cis(χ/2) * cos(x-y) + Ω₊ * cis(-χ/2) * cos(x+y) ) / 2
end

ϵ::Float = 0.1
ϵc::Float = 0.09
Ω₁₀::Float = 2000
Ω₊ = Ω₁₀ / (ϵ*√(1+ϵc^2))
Ω₋ = Ω₊ * ϵc
χ::Float = 1.4
Γ₃::Float = 1e3

# Use full period of 𝛺₂
xlimits = (-π, π) .|> Float
ylimits = (-π, π) .|> Float

# Calculate
M = 100
𝑈 = [nothing nothing 𝛺₁      
     nothing nothing 𝛺₂_cis
     nothing nothing nothing] # only upper triangle is needed
@time xh = XSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=trues(3, 3), Γ=[0, 0, Γ₃], fft_threshold=1e-3)
@time diagonalize!(xh, nev=5);
xh.ε

stateno = 1
@time xs, ys, ψ = make_eigenfunction(xh, stateno, 101, 101);

plot_comps(xs, ys, ψ)

### Quasimomenta

M = 50
xh = XSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=trues(3, 3), Γ=[0, 0, Γ₃])
ncells = 21
P = xlimits[2] - xlimits[1]
qlimits = (-π/P, π/P)
qxs = range(qlimits..., ncells)
qys = Float[0]
@time diagonalize!(xh, [qxs, qys]; nev=5); # doing a cut for fixed 𝑞𝑦 = 0
fig = plot();
for n in axes(xh.ε_q, 1)
    scatter!(qxs, real.(xh.ε_q[n, :, 1]), c=n)
end
fig