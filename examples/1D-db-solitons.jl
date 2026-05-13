using QuantumHamiltonians, AppleAccelerate

using Plots, LaTeXStrings
plotlyjs()
theme(:dark, size=(600, 500))
CMAP = cgrad(:Spectral, rev=true);
include("helpers.jl")

Float = Float64 # operating type

#=
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Stationary Manakov dark-bright soliton, see e.g. https://doi.org/10.1093/oso/9780192843234.001.0001 Section 27.1 or https://dx.doi.org/10.1103/PhysRevLett.87.010401 ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
=#

basis = :cos
M = get_M(basis, 8)

R = 10 |> Float
xlimits = (-R, R)

ph_db = PSpaceHamiltonian{:dense}([xlimits], fill(nothing, 2, 2); basis, 𝑈_iseven=trues(2, 2), M, δ=√0.5);

μ₀ = 4 |> Float
η = 1 |> Float
D = √(μ₀ - η^2)
g = ones(Float, 2, 2)
μs = [μ₀, (μ₀+η^2)/2] # actual chemical potentials of the two components

# Get stationary state starting from a basic tanh-sech trial
@time xs, sol = find_stationary(ph_db, [tanh, sech], g, μs; abstol=1e-12, show_trace=Val(true))
ψ_db = sol.u
E, μs, ηs = get_EμN(ph_db, ψ_db, g; state_is_pspace=false)
plot_comps(xs, ψ_db)

# Calculate BdG spectrum. Depending on the (random) init of ArnoldiMethod, this might yield -- in the best case -- a *real* array, with 6 eigenvalues ~1e-7.
# In other cases it yields a complex array, where four of those eigenvalues have real part of ~1e-7 and strictly zero imaginary part, while the remaining two are 1e-14 ± 1e-7im.
vals, vecs = bdg_spectrum(ph_db, ψ_db, g, μs) # also works with nev=10 and/or `storage=:sparse`. `storage=:lazy` fails.
scatter(vals, legend=false, markersize=2, markerstrokewidth=0)
smallindx = findall(x -> abs(x) < 1e-2, vals) # find indices of very small values
vals[smallindx] # yields the two imaginary frequencies (~1e-8) and four real ones (~2e-7)

# Side Code: Can also calculate in p-space
# ph_half = PSpaceHamiltonian{:dense}([xlimits], fill(nothing, 2, 2); basis, 𝑈_iseven=trues(2, 2), M=M÷2, δ=√0.5);
# vals, vecs = QuantumHamiltonians.bdg_spectrum_pspace(ph_half, [ψ_db[1:end÷2] ψ_db[end÷2+1:end]], g, μs)
# scatter(vals, legend=false, markersize=2, markerstrokewidth=0)
# smallindx = findall(x -> abs(x) < 1e-2, vals) # find indices of very small values
# vals[smallindx] # usually yields the two imaginary frequencies (~1e-8) and four real ones (~2e-7)

# Calculate BdG for exact analytical result
Ψ_exact = [x -> √μ₀ * tanh(D*x), x -> η * sech(D*x)]
plot(xs, Ψ_exact[1]); plot!(xs, Ψ_exact[2])
Ψ₀_sampled = [Ψ_exact[1].(xs); Ψ_exact[2].(xs)] |> vec
vals, vecs = bdg_spectrum(ph_db, Ψ₀_sampled, g, μs)
scatter(vals, legend=false, markersize=2, markerstrokewidth=0) # all are real, good
smallindx = findall(x -> abs(x) < 1e-2, vals) # find indices of very small values
vals[smallindx]

# Real-time evolution (checking that the state indeed remains stationary)
nsaves = 100
T_max = 50 |> Float
dt = 1e-3 |> Float
@time sol = propagate(ph_db, [ψ_db[1:end÷2], ψ_db[end÷2+1:end]], g; T_max, dt, itime=false, nsaves, solver=QuantumHamiltonians.ODE.ETDRK2())
V = sol.u[end]
E, μs = get_EμN(ph_db, V, g)
xs, U = make_map_comps(ph_db, sol; itime=false)
figs = [heatmap(xs, sol.t, abs2.(U[:, :, c])', c=CMAP, xlabel="x", ylabel="t") for c in 1:ph_db.nc]
plot(figs..., layout=(ph_db.nc, 1))

#=
╔═════════════════════════════════════════════════════════════════════════════════════════════════╗ 
║ DB lattices from http://dx.doi.org/10.1103/PhysRevA.91.023619 (https://arxiv.org/abs/1402.1895) ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════╝
=#

using ProgressMeter
"""
For each 𝑔ᵢ in `gs` find the stationary state and compute BdG. The initial trial is taken as `sol.u`, and each next step uses the previous solution as a trial.
x-space approach is faster because it succeeds in doing partialschur, while in p-space it fails (yields random numbers) and hence requires doing full diagonalisation.
`whichg=12` will replace elements g₁₂ and g₂₁, while `whichg=22` will replace g₂₂.
`qs` are in the format [qxs, qys, …], where elements are vectors.
"""
function scan_gs(gs, sol, qs=nothing; whichg=12, nev=0, abstol=1e-8, bdg_verbose=false, xspace=true)
     D = length(ph.xlims)
     B = (2M + 1)^D # block size
     nsaves = nev == 0 ? 2B*ph.nc : nev # number of eigenvalues and eigenvectors to store: if `nev` is zero (or not passed), then store all
     # data[:, i] holds all eigenvalues for `i`th gs; in case of passed `qs`, the values corresponding to different qs are linearised and lumped together
     data = Matrix{Complex{Float}}(undef, nsaves * (isnothing(qs) ? 1 : sum(length, qs)), length(gs))
     @showprogress for (i, gi) in enumerate(gs)
          if whichg == 12
               g[1, 2] = g[2, 1] = gi
          else
               g[2, 2] = gi
          end
          xs, sol = find_stationary(ph, [sol.u[1:end÷2], sol.u[end÷2+1:end]], g, μs; abstol)
          Int(sol.retcode) != 1 && println("\n i = $i: sol.retcode = $(sol.retcode)\n maximum(resid) = $(maximum(sol.resid))") # print retcode if unsuccessful
          if xspace
               if isnothing(qs)
                    vals, _ = bdg_spectrum(ph, sol.u, g, μs; nev, verbose=bdg_verbose)
                    data[:, i] = vals[1:nsaves]
               else
                    vals, _ = bdg_spectrum(ph, sol.u, g, μs, qs; nev, verbose=bdg_verbose)
                    data[:, i] = reshape(vals, :)
               end
          else
               if isnothing(qs)
                    vals, _ = QuantumHamiltonians.bdg_spectrum_pspace(ph_half, [sol.u[1:end÷2] sol.u[end÷2+1:end]], g, μs; nev)
                    sp = sortperm(vals, by=abs)
                    data[:, i] = vals[sp[1:nsaves]]
               else
                    for (iq, q) in enumerate(qs[1]) # specific for 1 dimension
                         vals, _ = QuantumHamiltonians.bdg_spectrum_pspace(ph_half, [sol.u[1:end÷2] sol.u[end÷2+1:end]], g, μs, [q]; nev)
                         sp = sortperm(vals, by=abs)
                         data[(iq-1)nsaves+1:iq*nsaves, i] = vals[sp[1:nsaves]]
                    end
               end
          end
     end
     return data
end

basis = :cis
M = get_M(basis, 7) # in x-space, use 7 for single period floquet, 9 for 5 periods. In p-space, increase by 1.

# trial
nT = 1 # number of periods
Ψ₀ = [x -> sin(2π/(2R/nT)*x), x -> cos(2π/(2R/nT)*x)]

R = 20*nT |> Float
xlimits = (-R, R)

ph = PSpaceHamiltonian{:dense}([xlimits], fill(nothing, 2, 2); basis, 𝑈_iseven=trues(2, 2), M, δ=√0.5);
# for calculating in momentum space (using `bdg_spectrum_pspace`) we need ph with half harmonics. Note that then M must be divisible by 2, so `get_M(basis, 9) = 247` won't work -- use 262 instead. However, the sults are then unconverged anyway -- use `get_M(basis, 10)`.
ph_half = PSpaceHamiltonian{:dense}([xlimits], fill(nothing, 2, 2); basis, 𝑈_iseven=trues(2, 2), M=M÷2, δ=√0.5);

xs = range(xlimits..., 100)
plot(xs, Ψ₀[1]); plot!(xs, Ψ₀[2])

μs = [1.5, 1.23]

g = [1    1
     1 0.95] .|> Float

# nlsolve
@time xs, sol = find_stationary(ph, Ψ₀, g, μs, abstol=1e-11, show_trace=Val(true)) 
ψ_lattice = sol.u
get_EμN(ph, ψ_lattice, g, state_is_pspace=false)
plot_comps(xs, ψ_lattice)

@time vals, vecs = bdg_spectrum(ph, ψ_lattice, g, μs, verbose=true, nev=100)
# @time vals, vecs = QuantumHamiltonians.bdg_spectrum_pspace(ph_half, [ψ_lattice[1:end÷2] ψ_lattice[end÷2+1:end]], g, μs; verbose=true, nev=100)

scatter(vals, legend=false, markersize=2, markerstrokewidth=0)

smallindx = findall(x -> abs(x) < 1e-2, vals)
vals[smallindx]

# scanning g without quasimomentum, cf. Fig. 7, left plots
g12s = range(1, 0.5, 50)
g12s = range(1, 1.5, 50)
data = scan_gs(g12s, sol; nev=0, abstol=1e-11) # use nev=0 in x-space (Arnoldi yields noise, so need full diagonalisation)
data = scan_gs(g12s, sol; nev=100, abstol=1e-11, xspace=false) # in p-space, Arnoldi works fine, so can use e.g. nev=100. Remember to increase M compared to x-space

# scanning g without quasimomentum, cf. Fig. 7, right plots
qs = range(-π/2R, π/2R, 11)
data = scan_gs(g12s, sol, [qs]; nev=0, abstol=1e-11, bdg_verbose=false)
data = scan_gs(g12s, sol, [qs]; nev=100, abstol=1e-11, bdg_verbose=false, xspace=false)

fig_real = plot(); fig_imag = plot();
scatter!(fig_real, g12s, real.(data)', c=1, legend=false, markersize=2, markerstrokewidth=0, ylims=(0, 0.1));
scatter!(fig_imag, g12s, imag.(data)', c=1, legend=false, markersize=2, markerstrokewidth=0, ylims=(0, 0.04));
plot(fig_real, fig_imag, layout=(1, 2))
title!("x-space, M = $M")

#=
╔════════════════════════════════════════════════════════════════════════════════════════════════════╗ 
║ DB soliton pair from https://doi.org/10.1103/PhysRevA.97.043623 (https://arxiv.org/abs/1802.06230) ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════╝
=#

basis = :cos
M = get_M(basis, 9)

R = 30 |> Float
xlimits = (-R, R)

ph = PSpaceHamiltonian{:dense}([xlimits], fill(nothing, 2, 2); basis, 𝑈_iseven=trues(2, 2), M, δ=√0.5);

# trial from eqs. (9)-(10)
x₀ = 1.8; D = 1; η = 0.5
Ψ₀ = [x -> tanh(D*(x-x₀))tanh(D*(x+x₀)),
      x -> η*sech(D*(x-x₀)) - η*sech(D*(x+x₀))]
xs = range(xlimits..., 500)
plot(xs, Ψ₀[1]); plot!(xs, Ψ₀[2], ylims=(-1.1, 1.1))

# exact solution
μ₀ = 0.7
a = √(2-2μ₀)
δ = 0
δ₁ = δ
δ₂ = -δ₁
ξ₁(x) = x - δ₁
ξ₂(x) = a*(x - δ₂)
Ψ₀ = [x -> ((1-a)cosh(ξ₁(x)+ξ₂(x)) - (1+a)cosh(ξ₁(x)-ξ₂(x))) / ((1-a)cosh(ξ₁(x)+ξ₂(x)) + (1+a)cosh(ξ₁(x)-ξ₂(x))),
      x -> 2(1-a^2)sinh(ξ₁(x)) / ((1-a)cosh(ξ₁(x)+ξ₂(x)) + (1+a)cosh(ξ₁(x)-ξ₂(x)))]

xs = range(xlimits..., 500)
plot(xs, Ψ₀[1]); plot!(xs, Ψ₀[2], ylims=(-1.1, 1.1))
plot(xs, abs2∘Ψ₀[1]); plot!(xs, abs2∘Ψ₀[2])

μs = [1, μ₀] .|> Float

g = [1 1
     1 1.05] .|> Float

@time xs, sol = find_stationary(ph, Ψ₀, g, μs; abstol=1e-10, show_trace=Val(true))
ψ_lattice = sol.u
plot_comps(xs, ψ_lattice)
get_EμN(ph, ψ_lattice, g, state_is_pspace=false)

### BdG (Fig. 1)
# using x-space, for this problem Arnoldi works fine
g22s = range(0.95, 1.125, 50)
nev = 100
data = scan_gs(g22s, sol; whichg=22, nev, abstol=1e-10)

fig_real = plot(); fig_imag = plot();
scatter!(fig_real, g22s, real.(data)', c=1, legend=false, markersize=2, markerstrokewidth=0, ylims=(0, 0.3));
scatter!(fig_imag, g22s, imag.(data)', c=1, legend=false, markersize=2, markerstrokewidth=0, ylims=(0, 0.08));
plot(fig_real, fig_imag, layout=(2, 1))

### Real-time evolution (Fig. 1 for g22 = 1.05)
nsaves = 100
T_max = 1000 |> Float
dt = 1e-3 |> Float
@time sol = propagate(ph, [ψ_lattice[1:end÷2], ψ_lattice[end÷2+1:end]], g; T_max, dt, itime=false, nsaves, solver=QuantumHamiltonians.ODE.ETDRK2()) # takes ~45 seconds; the time point at which the instability sets in (𝑡 = 500 in the paper) depends on the accuracy of the ODE solver and the accuracy of the initial state
V = sol.u[end]
E, μ = get_EμN(ph, V, g)
xs, U = make_map_comps(ph, sol; itime=false)
figs = [heatmap(xs, sol.t, abs2.(U[:, :, c])', c=CMAP, xlabel="x", ylabel="t") for c in 1:ph.nc]
plot(figs..., layout=(ph.nc, 1))