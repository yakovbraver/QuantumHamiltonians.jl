#=
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Dark state analysis of https://doi.org/10.1103/PhysRevA.107.033328 (https://arxiv.org/abs/2304.00302) ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════╝
=# 
using QuantumHamiltonians

using Plots
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_phase = cgrad(:RdBu_9);
theme(:dark, size=(600, 500))

function 𝑈(x::Real, y::Real)
    (sin(x+y)^2 + (ϵc*sin(x-y))^2) / 𝛼(x, y)^2 * 2ϵ^2 * (1+ϵc^2)
end

function 𝐴ˣ(x::Real, y::Real)
    sin(2y) .* ϵc .* sin(χ) ./ 𝛼(x, y)
end

function 𝐴ʸ(x::Real, y::Real)
    sin(2x) .* ϵc .* sin(χ) ./ 𝛼(x, y)
end

function 𝛼(x::Real, y::Real)
    η₋ = cos(x-y); η₊ = cos(x+y)
    return ϵ^2 * (1 + ϵc^2) + η₊^2 + (ϵc*η₋)^2 - 2ϵc*η₊*η₋*cos(χ)
end

Float = Float64 # real operating type; sparse supports only Float64 :(

ϵ::Float = 0.1
ϵc::Float = 1

########## χ = 0

χ::Float = 0.0

xlimits = (0, π) .|> Float
ylimits = (0, π) .|> Float

# plot potential
M = 32
N = 2M + 1
xs = range(xlimits..., N)
ys = range(ylimits..., N)
surface(xs, ys, 𝑈)

# Here we have no 𝐴, so Hamiltonian ph.H will be real. Eigenfunctions also since basis is real in nonperiodic case. `𝑈_iseven` is not used in the nonperiodic case
ph = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cos, M)

# Or we can solve periodic case. Potential is even, so we set `𝑈_iseven`, yielding a real Hamiltonian
ph = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=true)
@time ph = PSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=true, fft_threshold=Float(1e-1));
matrix_density(ph)

# basis=:cis, M=64, nev=1: Dense: 9.7s (w/ AA: 17s). Sparse: 6.6s (w/ AA: 9.3s) at threshold=1e-1 (less accurate). [AppleAccelerate slows down. Tested with --check-bounds=no]
@time diagonalize!(ph, nev=5);
ph.ε

stateno = 1
xs, ψ = make_eigenfunction(ph, stateno)
surface(xs[:, 1], xs[:, 2], abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)

# Diagonalisation in x-space -- faster than p-space for higher M.
# basis=:cis, M=64, nev=1: 0.76s (w/o AA: 2.4s).  [AppleAccelerate speeds up. Tested with --check-bounds=no]
@time xh = XSpaceHamiltonian([xlimits, ylimits], 𝑈; basis=:cis, M=64);
@time xh = XSpaceHamiltonian([xlimits, ylimits], 𝑈; basis=:cos, M=64);
@time xh = XSpaceHamiltonian([xlimits, ylimits], 𝑈; basis=:sin, M=63);
@time vals, vecs, info = diagonalize(xh, nev=1); # eigenvalues 2 and 3 are degenerate; usually only one is obtained. But works well for nev=1. Increase to krylovdim=40 to converge to 1e-12
vals
surface(xh.ft.xs[:, 1], xh.ft.xs[:, 2], abs2.(vecs[1].data[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)

### Quasimomenta

# folded spectrum for fixed 𝑞ʸ = 0

xlimits = (0, 2π) .|> Float
ylimits = (0, 2π) .|> Float
M = 30
@time ph = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=true); # M=30, ncells=21: 0.025 s construct + 8.4 s diagonalise (w/o AA)
@time ph = PSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=true, fft_threshold=1e-2); # M=30, thresh=1e-2, ncells=21: 0.05 s + 16.8 s diagonalise
matrix_density(ph)

ncells = 21
P = xlimits[2] - xlimits[1]
qlimits = (-π/P, π/P) .|> Float
qxs = range(qlimits..., ncells) .|> Float
qys = Float[0] # doing a cut for fixed 𝑞ʸ = 0
@time diagonalize!(ph, [qxs, qys]; nev=5);

fig = plot();
for n in axes(ph.ε_q, 1)
    scatter!(qxs, ph.ε_q[n, :, 1], c=n)
end
fig

stateno = 5
iqs = [15, 1]
xs, ψ = make_eigenfunction(ph, stateno, iqs)
surface(xs[:, 1], xs[:, 2], abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs[:, 1], xs[:, 2], angle.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)

# unfolded spectrum for 0 ≤ 𝑞ˣ, 𝑞ʸ ≤ π (a quater of Fig. 2(b))

xlimits = (0, π) .|> Float # (0, π) for unfolded spectrum (like Fig. 4), (0, 2π) for folded
ylimits = (0, π) .|> Float

M = 30
ph = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=true) # M=30, ncells=5, nev=1: 7.2 s.
ph = PSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈; basis=:cis, M, 𝑈_iseven=true, fft_threshold=Float(1e-1)) # M=30, ncells=5, nev=1: 10.9 s
ncells = 5
P = xlimits[2] - xlimits[1]
qlimits = (0, π/P)
qxs = range(qlimits..., ncells)
@time diagonalize!(ph, [qxs, qxs]; nev=1);
heatmap(qxs, qxs, ph.ε_q[1, :, :], c=:viridis, xlabel="q_x", ylabel="q_y")

########## χ = π/2, x from -π/2 to π/2 -- vortex state for hard-wall BC

χ::Float = π/2

xlimits = (-π/2, π/2) .|> Float
ylimits = (0, π) .|> Float

# plot potential
M = 30
N = 2M + 1
xs = range(xlimits..., N)
ys = range(ylimits..., N)
surface(xs, ys, 𝑈)
surface(xs, ys, (x, y) -> 𝐴ˣ(x, y)^2 + 𝐴ʸ(x, y)^2)

@time ph = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:sin, M);

@time diagonalize!(ph, nev=5);
ph.ε

stateno = 1
xs, ψ = make_eigenfunction(ph, stateno)
heatmap(xs[:, 1], xs[:, 2], abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs[:, 1], xs[:, 2], abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs[:, 1], xs[:, 2], angle.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_phase)

########## χ = π/2, full period

χ::Float = π/2

xlimits = (0, 2π) .|> Float
ylimits = (0, 2π) .|> Float

# plot potential
M = 50
xs = range(xlimits..., 2M)
ys = range(ylimits..., 2M)
surface(xs, ys, 𝑈)
surface(xs, ys, (x, y) -> 𝐴ˣ(x, y)^2 + 𝐴ʸ(x, y)^2)

# Pass 𝑈_iseven=true so that the imaginary part of the Fourier transform of 𝑈 is dropped, although the Hamiltonian is complex anyway because of 𝐴
@time ph = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M, 𝑈_iseven=true); # M=50: 56 s construct + 10 s diagonalise
@time ph = PSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M, 𝑈_iseven=true, fft_threshold=1e-3); # M=50, thresh=1e-3: 4.3 s construct + 42 s diagonalise (3-5 digits accuracy). thresh=1e-2: 0.9 s construct + 6.8 s diagonalise (2-3 digits accuracy); 
matrix_density(ph)

@time diagonalize!(ph, nev=5);
ph.ε

stateno = 1
xs, ψ = make_eigenfunction(ph, stateno);
heatmap(xs[:, 1], xs[:, 2], abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs[:, 1], xs[:, 2], abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs[:, 1], xs[:, 2], angle.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_phase)

# Diagonalisation in x-space: faster than p-space
@time xh = XSpaceHamiltonian([xlimits, ylimits], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M=64);
# Setting `ishermitian=false` because solver claims that map is nonhermitian. TODO: investigate
@time vals, vecs, info = diagonalize(xh, nev=5, ishermitian=false); # M=64: 6.5s. Increase `krylovdim` ot say 40 (or maxiter to say 200) if you want better convergence.
vals
heatmap(xh.ft.xs[:, 1], xh.ft.xs[:, 2], abs2.(vecs[stateno][1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xh.ft.xs[:, 1], xh.ft.xs[:, 2], angle.(vecs[stateno][1])', xlabel="x/a", ylabel="y/a", c=cmap_phase)


########## χ = 1.4, full period

χ::Float = 1.4

ϵ::Float = 0.1
ϵc::Float = 0.09

xlimits = (-π, π) .|> Float
ylimits = (-π, π) .|> Float

# plot potential
M = 32
xs = range(xlimits..., 2M)
ys = range(ylimits..., 2M)
surface(xs, ys, 𝑈)
surface(xs, ys, (x, y) -> 𝐴ˣ(x, y)^2 + 𝐴ʸ(x, y)^2)

@time ph = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M, 𝑈_iseven=true); # M = 50, Float64: 56 s construct + 10 s diagonalise
@time ph = PSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M, 𝑈_iseven=true, fft_threshold=1e-3); # M=50, thresh=1e-2: 1.7 s construct + 0.6 s diagonalise (4+ digits accuracy)
matrix_density(ph)

@time diagonalize!(ph, nev=5);
ph.ε

stateno = 1
xs, ψ = make_eigenfunction(ph, stateno);
heatmap(xs[:, 1], xs[:, 2], abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs[:, 1], xs[:, 2], abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs[:, 1], xs[:, 2], angle.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_phase)

# Diagonalisation in x-space: faster than p-space
@time xh = XSpaceHamiltonian([xlimits, ylimits], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M=64);
# Setting `ishermitian=false` because solver claims that map is nonhermitian. TODO: investigate
@time vals, vecs, info = diagonalize(xh, nev=5, ishermitian=false);
vals
heatmap(xh.ft.xs[:, 1], xh.ft.xs[:, 2], abs2.(vecs[stateno][1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xh.ft.xs[:, 1], xh.ft.xs[:, 2], angle.(vecs[stateno][1])', xlabel="x/a", ylabel="y/a", c=cmap_phase)


### Quasimomenta

M = 30
@time ph = PSpaceHamiltonian{:dense}([xlimits, ylimits], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M, 𝑈_iseven=true); # M=30, ncells=21: 3 s construct + 76 s diagonalise
@time ph = PSpaceHamiltonian{:sparse}([xlimits, ylimits], 𝑈, [𝐴ˣ, 𝐴ʸ]; basis=:cis, M, 𝑈_iseven=true, fft_threshold=1e-2) # M=30, ncells=21, thresh=1e-2: 0.17 s construct + 4.6 diagonalise (3+ digits accuracy)
matrix_density(ph)

ncells = 21
P = xlimits[2] - xlimits[1]
qlimits = (-π/P, π/P) .|> Float
qxs = range(qlimits..., ncells) .|> Float
qys = Float[0] # doing a cut for fixed 𝑞ʸ = 0
@time diagonalize!(ph, [qxs, qys]; nev=5);

fig = plot();
for n in axes(ph.ε_q, 1)
    scatter!(qxs, ph.ε_q[n, :, 1], c=n)
end
fig