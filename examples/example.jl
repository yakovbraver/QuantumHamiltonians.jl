using XSpaceHamiltonians

using Plots, DelimitedFiles
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_cyclic = cgrad(:cyclic_mrybm_35_75_c68_n256);
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
χ::Float = 0

########## χ = 0

xlimits = (0, π) .|> Float
ylimits = (0, π) .|> Float
M = 50
dh = DenseHamiltonian(xlimits, ylimits; isperiodic=false, M, 𝑈, 𝐴_x, 𝐴_y)

@time diagonalize!(dh, nev=1);
dh.ε

stateno = 1
xs, ys, ψ = make_wavefunction(dh, stateno, M, M)
surface(xs, ys, abs2.(ψ)')

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

@time dh = DenseHamiltonian(xlimits, ylimits; isperiodic=false, M, 𝑈, 𝐴_x, 𝐴_y);
@time diagonalize!(dh, nev=1);
dh.ε

stateno = 1
xs, ys, ψ = make_wavefunction(dh, stateno, 100, 100)
heatmap(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_cyclic)

########## χ = π/2, FULL

χ = π/2

xlimits = (0, 2π) .|> Float
ylimits = (0, 2π) .|> Float

# plot potential
M = 60
xs = range(xlimits[1], xlimits[2], 2M)
ys = range(ylimits[1], ylimits[2], 2M)
surface(xs, ys, 𝑈)
surface(xs, ys, (x, y) -> 𝐴_x(x, y)^2 + 𝐴_y(x, y)^2)

# use either sparse or dense, whichever you like. Sparse is faster but less accurate
@time dh = DenseHamiltonian(xlimits, ylimits; isperiodic=true, M, 𝑈, 𝐴_x, 𝐴_y);
@time dh = SparseHamiltonian(xlimits, ylimits; 𝑈, 𝐴_x, 𝐴_y, M=2M);
@time diagonalize!(dh, nev=5);
dh.ε

stateno = 1
@time xs, ys, ψ = make_wavefunction(dh, stateno, 100, 100);
heatmap(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_cyclic)
writedlm("exact_lambda_no1_256.txt", ψ, ',')

@time diagonalize!(dh, nev=5);
