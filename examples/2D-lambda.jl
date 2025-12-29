# Dark state analysis of https://doi.org/10.1103/PhysRevA.107.033328 (https://arxiv.org/abs/2304.00302)
using XSpaceHamiltonians

using Plots, DelimitedFiles
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

Float = Float32 # operating type

ϵ::Float = 0.1
ϵc::Float = 1

########## χ = 0

χ::Float = 0

xlimits = (0, π) .|> Float
ylimits = (0, π) .|> Float

# plot potential
M = 30
N = 2M + 1
xs = range(xlimits[1], xlimits[2], N)
ys = range(ylimits[1], ylimits[2], N)
surface(xs, ys, 𝑈)

# Here we have no 𝐴, so Hamiltonian xh.H will be real. Eigenfuntions also since basis is real in nonperiodic case
xh = DenseHamiltonian(xlimits, ylimits; isperiodic=false, M, 𝐻=[𝑈;;], 𝐻_iseven=[true;;])

# Or we can solve periodic case. Potential is even, so we set `𝐻_iseven`, yielding a real Hamiltonian
xh = DenseHamiltonian(xlimits, ylimits; isperiodic=true, M, 𝐻=[𝑈;;], 𝐻_iseven=[true;;])
xh = SparseHamiltonian(xlimits, ylimits; isperiodic=true, M, 𝐻=[𝑈;;], 𝐻_iseven=[true;;]) # sparse works but the matrix is not actually sparse (sparsity ~0.5), so diagonalisation is longer than dense

@time diagonalize!(xh, nev=5);
xh.ε

stateno = 1
xs, ys, ψ = make_eigenfunction(xh, stateno, M, M)
surface(xs, ys, abs2.(ψ[1])')

########## χ = π/2, x from -pi/2 to pi/2

χ = π/2

xlimits = (-π/2, π/2) .|> Float
ylimits = (0, π) .|> Float

# plot potential
M = 50
N = 2M + 1
xs = range(xlimits[1], xlimits[2], N)
ys = range(ylimits[1], ylimits[2], N)
surface(xs, ys, 𝑈)
surface(xs, ys, (x, y) -> 𝐴_x(x, y)^2 + 𝐴_y(x, y)^2)

@time dh = DenseHamiltonian(xlimits, ylimits; isperiodic=false, M, 𝐻=[𝑈;;], 𝐴_x, 𝐴_y);

@time diagonalize!(dh, nev=1);
dh.ε

stateno = 1
xs, ys, ψ = make_eigenfunction(dh, stateno, 100, 100)
heatmap(xs, ys, abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs, ys, abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_phase)

########## χ = π/2, full period

χ = π/2

xlimits = (0, 2π) .|> Float
ylimits = (0, 2π) .|> Float

# plot potential
M = 50
xs = range(xlimits[1], xlimits[2], 2M)
ys = range(ylimits[1], ylimits[2], 2M)
surface(xs, ys, 𝑈)
surface(xs, ys, (x, y) -> 𝐴_x(x, y)^2 + 𝐴_y(x, y)^2)

# use either sparse or dense, whichever you like. Sparse is faster but less accurate
@time xh = DenseHamiltonian(xlimits, ylimits; isperiodic=true, M, 𝐻=[𝑈;;], 𝐻_iseven=[false;;], 𝐴_x, 𝐴_y);
@time xh = SparseHamiltonian(xlimits, ylimits; 𝑈, 𝐴_x, 𝐴_y, M=2M); # this is one-component only
@time diagonalize!(xh, nev=5);
xh.ε

stateno = 1
@time xs, ys, ψ = make_eigenfunction(xh, stateno, 100, 100);
heatmap(xs, ys, abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs, ys, abs2.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_phase)