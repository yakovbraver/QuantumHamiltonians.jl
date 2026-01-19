# Dark state analysis of https://doi.org/10.1103/PhysRevA.107.033328 (https://arxiv.org/abs/2304.00302)
using XSpaceHamiltonians

using Plots
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_phase = cgrad(:RdBu_9);
theme(:dark, size=(600, 500))

function 𝑈(x::Real, y::Real)
    (sin(x+y)^2 + (ϵc*sin(x-y))^2) / 𝛼(x, y)^2 * 2ϵ^2 * (1+ϵc^2)
end

function 𝐴_x(x::Real, y::Real)
    sin(2y) .* ϵc .* sin(χ) ./ 𝛼(x, y)
end

function 𝐴_y(x::Real, y::Real)
    sin(2x) .* ϵc .* sin(χ) ./ 𝛼(x, y)
end

function 𝛼(x::Real, y::Real)
    η₋ = cos(x-y); η₊ = cos(x+y)
    return ϵ^2 * (1 + ϵc^2) + η₊^2 + (ϵc*η₋)^2 - 2ϵc*η₊*η₋*cos(χ)
end

Float = Float32 # real operating type; sparse supports only Float64 :(

ϵ::Float = 0.1
ϵc::Float = 1

########## χ = 0

χ::Float = 0

xlimits = (0, π) .|> Float
ylimits = (0, π) .|> Float

# plot potential
M = 50
N = 2M + 1
xs = range(xlimits..., N)
ys = range(ylimits..., N)
surface(xs, ys, 𝑈)

# Here we have no 𝐴, so Hamiltonian xh.H will be real. Eigenfunctions also since basis is real in nonperiodic case. `𝑈_iseven` is not used in the nonperiodic case
@time xh = XSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:sin, M);
xh = XSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:sin, M)

# Or we can solve periodic case. Potential is even, so we set `𝑈_iseven`, yielding a real Hamiltonian
xh = XSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=true)
xh = SparseHamiltonian{:sparse}(𝑈, xlimits, ylimits; isperiodic=true, M, 𝑈_iseven=true, fft_threshold=1e-3) # for `fft_threshold=1e-3` sparsity is 0.1, so diagonalisation is slower compared to dense

@time diagonalize!(xh, nev=5);
xh.ε

stateno = 1
xs, ys, ψ = make_eigenfunction(xh, stateno, M, M)
surface(xs, ys, abs2.(ψ[1])')

### Quasimomenta

# folded spectrum for fixed 𝑞_𝑦 = 0

xlimits = (0, 2π) .|> Float
ylimits = (0, 2π) .|> Float
M = 30
xh = XSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=true)
xh = XSpaceHamiltonian{:sparse}(𝑈, xlimits, ylimits; isperiodic=true, M, 𝑈_iseven=true, fft_threshold=1e-1)
ncells = 11
P = xlimits[2] - xlimits[1]
qlimits = (-π/P, π/P) .|> Float
qxs = range(qlimits..., ncells)
qys = Float[0] # doing a cut for fixed 𝑞_𝑦 = 0
@time diagonalize!(xh, [qxs, qys]; nev=5);
fig = plot();
for n in axes(xh.ε_q, 1)
    scatter!(qxs, xh.ε_q[n, :, 1], c=n)
end
fig

# unfolded spectrum for 0 ≤ 𝑞ₓ, 𝑞_𝑦 ≤ π (a quater of Fig. 2(b))

xlimits = (0, π) .|> Float # (0, π) for unfolded spectrum (like Fig. 4), (0, 2π) for folded
ylimits = (0, π) .|> Float

M = 35
xh = XSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=true) # M = 30, ncells = 5: 9.6 s.
xh = XSpaceHamiltonian{:sparse}(𝑈, xlimits, ylimits; isperiodic=true, M, 𝑈_iseven=true, fft_threshold=1e-1) # M = 30, ncells = 5: 10.9 s
ncells = 5
P = xlimits[2] - xlimits[1]
qlimits = (0, π/P)
qxs = range(qlimits..., ncells)
@time diagonalize!(xh, qxs, qxs; nev=1); # doing a cut for fixed 𝑞𝑦 = 0
heatmap(qxs, qxs, xh.ε_q[1, :, :], c=:viridis, xlabel="q_x", ylabel="q_y")

########## χ = π/2, x from -π/2 to π/2

χ = π/2

xlimits = (-π/2, π/2) .|> Float
ylimits = (0, π) .|> Float

# plot potential
M = 15
N = 2M + 1
xs = range(xlimits..., N)
ys = range(ylimits..., N)
surface(xs, ys, 𝑈)
surface(xs, ys, (x, y) -> 𝐴_x(x, y)^2 + 𝐴_y(x, y)^2)

@time xh = XSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈, [𝐴_x, 𝐴_y]; basis=:sin, M);

@time diagonalize!(xh, nev=5);
xh.ε

stateno = 1
xs, ys, ψ = make_eigenfunction(xh, stateno, 100, 100)
heatmap(xs, ys, abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs, ys, abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_phase)

########## χ = π/2, full period

χ = π/2

xlimits = (0, 2π) .|> Float
ylimits = (0, 2π) .|> Float

# plot potential
M = 30
xs = range(xlimits..., 2M)
ys = range(ylimits..., 2M)
surface(xs, ys, 𝑈)
surface(xs, ys, (x, y) -> 𝐴_x(x, y)^2 + 𝐴_y(x, y)^2)

# The matrix is not really sparse because of 𝐴, so dense method is more efficient
# We pass 𝑈_iseven=true so that the imaginary part of the Fourier transform of 𝑈 is dropped, although the Hamiltonian is complex anyway because of 𝐴
@time xh = XSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈, [𝐴_x, 𝐴_y]; basis=:cis, M, 𝑈_iseven=true);
@time xh = XSpaceHamiltonian{:sparse}(𝑈, Float64.(xlimits), Float64.(ylimits); isperiodic=true, M, 𝑈_iseven=true, 𝐴_x, 𝐴_y);

@time diagonalize!(xh, nev=1);
xh.ε

stateno = 1
@time xs, ys, ψ = make_eigenfunction(xh, stateno, 100, 100);
heatmap(xs, ys, abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs, ys, abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_phase)

########## χ = 1.4, full period

χ = 1.4

ϵ::Float = 0.1
ϵc::Float = 0.09

xlimits = (-π, π) .|> Float
ylimits = (-π, π) .|> Float

# plot potential
M = 15
xs = range(xlimits..., 2M)
ys = range(ylimits..., 2M)
surface(xs, ys, 𝑈)
surface(xs, ys, (x, y) -> 𝐴_x(x, y)^2 + 𝐴_y(x, y)^2)

# For this configuration, sparse is more efficient
@time xh = XSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈, [𝐴_x, 𝐴_y]; basis=:cis, M, 𝑈_iseven=true); # M = 50, Float64: 56 s construct + 10 s diagonalise
@time xh = XSpaceHamiltonian{:sparse}(𝑈, Float64.(xlimits), Float64.(ylimits); isperiodic=true, M, 𝑈_iseven=true, 𝐴_x, 𝐴_y, fft_threshold=1e-3); # M = 50, Float64: 1.7 s construct + 0.6 s diagonalise, matches dense result at 4 digits accuracy for `fft_threshold=1e-3`

@time diagonalize!(xh, nev=5);
xh.ε

stateno = 1
@time xs, ys, ψ = make_eigenfunction(xh, stateno, 100, 100);
heatmap(xs, ys, abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs, ys, abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_phase)

### Quasimomenta

@time xh = XSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈, [𝐴_x, 𝐴_y]; basis=:cis, M, 𝑈_iseven=true);
xh = XSpaceHamiltonian{:sparse}(𝑈, xlimits, ylimits; isperiodic=true, M, 𝑈_iseven=false, 𝐴_x, 𝐴_y, fft_threshold=1e-3)
ncells = 21
P = xlimits[2] - xlimits[1]
qlimits = (-π/P, π/P) .|> Float
qxs = range(qlimits..., ncells)
qys = Float[0] # doing a cut for fixed 𝑞_𝑦 = 0
@time diagonalize!(xh, [qxs, qys]; nev=5);

fig = plot();
for n in axes(xh.ε_q, 1)
    scatter!(qxs, xh.ε_q[n, :, 1], c=n)
end
fig