#=
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║ Analysis of https://doi.org/10.1103/PhysRevLett.117.233001 (https://arxiv.org/abs/1607.07338) ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝
=#
using PSpaceHamiltonians, AppleAccelerate

using Plots, LaTeXStrings
plotlyjs()
theme(:dark, size=(600, 500))
include("helpers.jl")

################ Dark state analysis

function 𝑈(x::Real)
    (ϵ*cos(x) / (ϵ^2 + sin(x)^2))^2
end

Float = Float32 # operating type

ϵ::Float = 0.1

P = 2π # use 2π (the period of the full 3-level Hamiltonian) to get the "folded" spectrum like in Fig. 2(c), or use π (the period of 𝑈) for the unfolded spectrum
xlimits = (0, P) .|> Float
M = 200
xs = range(xlimits..., 2M+1)
plot(xs, 𝑈)

@time xh = PSpaceHamiltonian{:dense}([xlimits], 𝑈; basis=:cis, 𝑈_iseven=true, M); # P=2π, M=200, ncells=101, nev=8: 0.0005 s construct + 0.28 s diagonalise
@time xh = PSpaceHamiltonian{:sparse}([xlimits], 𝑈; basis=:cis, 𝑈_iseven=true, M, fft_threshold=1f-3); # P=2π, M=200, ncells=101, nev=8: 0.0006 s construct + 0.26 s diagonalise (fft_threshold can be increased to 1e-3)
matrix_density(xh)

diagonalize!(xh; nev=5)
xh.ε
xs, ψ = make_eigenfunctions(xh, statenos=1:5, nx=100)
plot(xs, real(ψ[:, 1, 1]))

# calculate and plot components 1 and 2
ψD = ψ[:, 1, 1] |> real
ψ1 = @. ψD / √(1 + 1/ϵ^2 * sin(xs)^2)
ψ2 = @. -ψ1 * 1/ϵ * sin(xs)
f1 = plot(xs, abs2.(ψ1));
f2 = plot(xs, abs2.(ψ2));
plot(f1, f2, layout=(2,1))

# diagonalise with quasimomentum (Fig. 2(c))
ncells = 101
qlimits = (-π/P, π/P)
qs = range(qlimits[1], qlimits[2], ncells)
@time diagonalize!(xh, [qs]; nev=8);
nlevels = 6 # use 6 for folded spectrum or 3 for unfolded (depending on `P` above)
fig = plot();
for n in 1:nlevels
    plot!(qs, real.(xh.ε_q[n, :]), c=1, legend=false)
end
fig
ylims!(0.95, 1.1)


################ 3-component analysis


𝛺₁(x) = Ω₁₀/ϵ*cos(x) / 2 # we take cos so that `𝛺₁` is even in (0, P). Alternatively, can use sin as in the paper and use shifted xlimits. Or use sin with (0, P) without setting 𝑈_iseven
𝛺₂(x) = Ω₁₀ / 2

Ω₁₀::Float = 2f3
Γ₃::Float = 10
ϵ::Float = 0.1

P = 2π
xlimits = (0, P) .|> Float
M = 200

𝑉 = [nothing nothing 𝛺₁
     nothing nothing 𝛺₂
     nothing nothing nothing]
@time xh = PSpaceHamiltonian{:dense}([xlimits], 𝑉; basis=:cis, 𝑈_iseven=trues(3, 3), Γ=[0, 0, Γ₃], M); # M=200, ncells=101, nev=8: 0.003 s construct + 2.6 s diagonalise
@time xh = PSpaceHamiltonian{:sparse}([xlimits], 𝑉; basis=:cis, 𝑈_iseven=trues(3, 3), Γ=[0, 0, Γ₃], M, fft_threshold=1f-1); # M=200, ncells=101, nev=8: 0.0005 s construct + 0.33 s diagonalise
matrix_density(xh)

@time diagonalize!(xh; nev=5)
xh.ε

# plot eigenstates; should coincide with the dark state calculation above
xs, ψ = make_eigenfunctions(xh; statenos=1:5, nx=100)
plot_comps_complex(xs, ψ; stateno=1)

# diagonalise with quasimomentum (Fig. 2(c))
ncells = 101
qlimits = (-π/P, π/P)
qs = range(qlimits[1], qlimits[2], ncells)
@time diagonalize!(xh, [qs]; nev=8);

# optionally, set to zero elements whose real part is not in filterrange (not to pollute the view)
filterrange = (0.98, 1.1)
xh.ε_q[(real.(xh.ε_q) .< filterrange[1]) .| (real.(xh.ε_q) .> filterrange[2])] .= 0
xh.ε_q[abs.(imag.(xh.ε_q)) .> 1] .= 0 # or filter elements whose imaginary part is too large

# Plot energy spectrum (Fig 2(d)/(e)) with vertical lines indicating the imaginary part of the energy
fig = plot();
for (iq, q) in enumerate(qs)
    scatter!(fill(q, size(xh.ε_q, 1)), real.(xh.ε_q[:, iq]), c=1, legend=false, markerstrokewidth=0, markersize=2)
    for n in axes(xh.ε_q, 1)
        E = real.(xh.ε_q[n, iq])
        γ = imag.(xh.ε_q[n, iq])
        plot!([q, q], [E+γ/2, E-γ/2], c=2)
    end
end
fig
ylims!(0.95, 1.1)