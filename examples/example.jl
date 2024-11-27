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

const ϵ = 0.1f0
const ϵc = 1f0

########## χ = 0

const χ = 0f0

xlimits = (0, π) .|> Float32
ylimits = (0, π) .|> Float32
N = 2^5-1
dh = DirichletHamiltonian(xlimits, ylimits; 𝑈, N)

heatmap(dh.H, yaxis=:flip)

@time diagonalize!(dh, nev=1);

stateno = 1
xs, ys, ψ = make_wavefunction(dh, stateno, M, M)
surface(xs, ys, abs2.(ψ)')

########## χ = π/2, x from -pi/2 to pi/2

const χ = π/2 |> Float32

xlimits = (-π/2, π/2) .|> Float32
ylimits = (0, π) .|> Float32

# plot potential
N = 2^6 - 1
M = 2N + 1
xs = range(xlimits[1], xlimits[2], M)
ys = range(ylimits[1], ylimits[2], M)
surface(xs, ys, 𝑈)
surface(xs, ys, (x, y) -> 𝐴_x(x, y)^2 + 𝐴_y(x, y)^2)

########## χ = π/2, FULL

const χ = π/2 |> Float32

xlimits = (0, 2π) .|> Float32
ylimits = (0, 2π) .|> Float32

# plot potential
N = 2^6
M = 2N
xs = range(xlimits[1], xlimits[2], M)
ys = range(ylimits[1], ylimits[2], M)
surface(xs, ys, 𝑈)
surface(xs, ys, (x, y) -> 𝐴_x(x, y)^2 + 𝐴_y(x, y)^2)

########## Calculate

@time dh = PeriodicHamiltonian(xlimits, ylimits; 𝑈, 𝐴_x, 𝐴_y, M=N);
@time diagonalize!(dh, nev=1);

stateno = 1
xs, ys, ψ = make_wavefunction(dh, stateno, M, M)
heatmap(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
surface(xs, ys, abs2.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_rainbow)
heatmap(xs, ys, angle.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_cyclic)
writedlm("exact_lambda_no1_256.txt", ψ, ',')

@time dh = DirichletHamiltonian(xlimits, ylimits; 𝑈, 𝐴_x, 𝐴_y, N);
@time diagonalize!(dh, nev=5);

dh.ε