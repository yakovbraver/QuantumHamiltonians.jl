#=
╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ 3-component analysis of https://doi.org/10.1103/PhysRevA.107.033328 (https://arxiv.org/abs/2304.00302) ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════╝
=#
using QuantumHamiltonians, AppleAccelerate

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
    plot(figs..., layout=(3, 2))
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
𝑈 = [nothing nothing 𝛺₁      
     nothing nothing 𝛺₂
     nothing nothing nothing] # only upper triangle is needed
@time ph = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cis, M=15, 𝑈_iseven=trues(3, 3), Γ=[0, 0, Γ₃])
@time ph = PSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M=200, 𝑈_iseven=trues(3, 3), Γ=[0, 0, Γ₃], fft_threshold=1e-3) # fft_threshold=1e-3 removes noise; there are no actual elements which are that small
@time ph = PSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M=32, 𝑈_iseven=trues(3, 3), Γ=[0, 0, Γ₃], fft_threshold=1e-3) # fft_threshold=1e-3 removes noise; there are no actual elements which are that small
matrix_density(ph)

@time diagonalize!(ph, nev=5);
ph.ε

# l = findfirst(x -> real(x) > 0, ph.ε) # find the dark state from full diagonalisation
# ph.ε[l]

stateno = 1
xs, ys, ψ = make_eigenfunction(ph, stateno);

plot_comps(xs, ys, ψ)

### Diagonalisation in x-space. Inversion slows down solving, and this becomes much slower than sparse p-space diagonalisation. The matrix is indeed extremely sparse, so sparse method is obviously superior here.
xh = XSpaceHamiltonian([xlimits, ylimits], 𝑈; basis=:cis, M=32, Γ=[0, 0, Γ₃])
# test at lower accuracy for speed. Testing shows linear solve accuracy must be ~4 orders of magnitude higher than diagonalisation accuracy (`tol`).
@time diagonalize!(xh; nev=5, verbose=true, ls_verbose=false, ls_abstol=1e-9, ls_reltol=1e-9, tol=1e-5) # M=16: 130 s. Linear solving does ~2000 iterations for each Arnoldi iteration.
@time diagonalize!(xh, nev=5, verbose=true, ls_prec=:jacobi, ls_verbose=false, ls_abstol=1e-9, ls_reltol=1e-9, tol=1e-5); # M=16: 2.7 s, ~380 linsolve iterations. The preconditioner is singular, but the default shift works well.
@time diagonalize!(xh, nev=5, verbose=true, ls_prec=:block_jacobi, ls_verbose=false, ls_abstol=1e-9, ls_reltol=1e-9, tol=1e-5); # M=16: 0.5 s, ~120 linsolve iterations. M=32: 48 s, ~500 linsolve iterations; agrees with sparse calculation at ~1e-5, but is ~1000 times slower; does beat dense diagonalisation however.
@time diagonalize!(xh, nev=5, verbose=true, ls_prec=:block_jacobi, ls_verbose=false, ls_solver=QuantumHamiltonians.LS.KrylovJL_BICGSTAB, ls_abstol=1e-9, ls_reltol=1e-9, tol=1e-5); # BICGSTAB is faster than GMRES here: M=32: 27 s.
xh.ε

xs, ys, ψ = make_eigenfunction(xh, 1)
plot_comps(xs, ys, ψ)

## Diagonalising via `StateVector`. Linear solving struggles to converge.
# @time xh = XSpaceHamiltonian([xlimits, ylimits], 𝑈; basis=:cis, M=8, Γ=[0, 0, Γ₃])
# @time vals, vecs, info = QuantumHamiltonians.diagonalize_via_statevector(xh; nev=5, krylovdim=30, maxiter=200, tol=1e-3);
# vals
# plot_comps(xh.ft.xs, vecs[1])

### Quasimomenta

M = 50 # something like M=30 is needed to get converged lowest band
@time ph = PSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=trues(3, 3), Γ=[0, 0, Γ₃])
# ph = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=trues(3, 3), Γ=[0, 0, Γ₃])
ncells = 11
P = xlimits[2] - xlimits[1]
qlimits = (-π/P, π/P)
qxs = range(qlimits..., ncells)
qys = [0.0]
@time diagonalize!(ph, [qxs, qys]; nev=4); # doing a cut for fixed 𝑞ʸ = 0

fig = plot();
for n in axes(ph.ε_q, 1)
    scatter!(qxs, real.(ph.ε_q[n, :, 1]), c=n)
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

M = 100
𝑈 = [nothing nothing 𝛺₁      
     nothing nothing 𝛺₂_cis
     nothing nothing nothing] # only upper triangle is needed
@time ph = PSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=trues(3, 3), Γ=[0, 0, Γ₃], fft_threshold=1e-3)
@time diagonalize!(ph, nev=5);
ph.ε

stateno = 1
@time xs, ys, ψ = make_eigenfunction(ph, stateno);

plot_comps(xs, ys, ψ)

### Quasimomenta

M = 50
ph = PSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=trues(3, 3), Γ=[0, 0, Γ₃])
ncells = 21
P = xlimits[2] - xlimits[1]
qlimits = (-π/P, π/P)
qxs = range(qlimits..., ncells)
qys = Float[0]
@time diagonalize!(ph, [qxs, qys]; nev=5); # doing a cut for fixed 𝑞ʸ = 0

fig = plot();
for n in axes(ph.ε_q, 1)
    scatter!(qxs, real.(ph.ε_q[n, :, 1]), c=n)
end
fig
