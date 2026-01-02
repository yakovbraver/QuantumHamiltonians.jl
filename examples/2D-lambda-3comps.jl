# 3-component analysis of https://doi.org/10.1103/PhysRevA.107.033328 (https://arxiv.org/abs/2304.00302)
using XSpaceHamiltonians

using Plots, LaTeXStrings
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_phase = cgrad(:RdBu_9);
theme(:dark, size=(600, 500))

# Plot all components
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
Ω₊ = Ω₁₀ / (ϵ*√(1+ϵc^2)) # include division by 2 (present in Hamiltonian (1)) here
Ω₋ = Ω₊ * ϵc
Γ₃::Float = 1e3

# Use full period of 𝛺₂
xlimits = (-π, π) .|> Float
ylimits = (-π, π) .|> Float

M = 200 # for M=15, the (real part of) ground state energy matches M=200 at 5 digits accuracy, wfs also match well, so dense calculation (even with full diagonalisation) is possible
𝑈 = [nothing nothing 𝛺₁      
     nothing nothing 𝛺₂
     nothing nothing nothing] # only upper triangle is needed
𝑈_iseven = BitArray([0 0 1; 0 0 1; 0 0 0])
@time xh = XSpaceHamiltonian{:dense}(𝑈, xlimits, ylimits; isperiodic=true, M, 𝑈_iseven, Γ=[0, 0, Γ₃])
@time xh = XSpaceHamiltonian{:sparse}(𝑈, xlimits, ylimits; isperiodic=true, M, 𝑈_iseven, Γ=[0, 0, Γ₃])

@time diagonalize!(xh, nev=5);
xh.ε

l = findfirst(x -> real(x) > 0, xh.ε) # find the dark-state from full diagonalisation
xh.ε[l]

stateno = 5
@time xs, ys, ψ = make_eigenfunction(xh, stateno, 101, 101);

plot_comps(xs, ys, ψ)

### Complex 𝛺₂

function 𝛺₂_cis(x::Real, y::Real)
    ( -Ω₋ * cis(χ/2) * cos(x-y) + Ω₊ * cis(-χ/2) * cos(x+y) ) / 2
end

ϵ::Float = 0.1
ϵc::Float = 0.09
Ω₁₀::Float = 2000
Ω₊ = Ω₁₀ / (ϵ*√(1+ϵc^2)) # include division by 2 (present in Hamiltonian (1)) here
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
𝑈_iseven = BitArray([0 0 1; 0 0 1; 0 0 0])
@time xh = XSpaceHamiltonian{:dense}(𝑈, xlimits, ylimits; isperiodic=true, M, 𝑈_iseven, Γ=[0, 0, Γ₃])
@time xh = XSpaceHamiltonian{:sparse}(𝑈, xlimits, ylimits; isperiodic=true, M, 𝑈_iseven, Γ=[0, 0, Γ₃])

@time diagonalize!(xh, nev=5);
xh.ε
    
l = findfirst(x -> real(x) > 0, xh.ε)
xh.ε[l]

stateno = 5
@time xs, ys, ψ = make_eigenfunction(xh, stateno, 101, 101);

plot_comps(xs, ys, ψ)