# Parameters in https://dx.doi.org/10.1103/PhysRevA.91.053602
using XSpaceHamiltonians

using Plots, DelimitedFiles
plotlyjs()
theme(:dark, size=(600, 500))

### Lattice potential

function 𝑈(x::Real)
    V₀*sin(x)^2 - V₁*sin(2x)^2
end

Float = Float32 # operating type

V₀::Float = 10
V₁::Float = 5

xlimits = (-3π, 3π) .|> Float
M = 150

dh = DenseHamiltonian1D(xlimits; isperiodic=true, iseven=true, M, 𝑈)
@time diagonalize!(dh, nev=0);
foreach(println, dh.ε)

# plot eigenfunctions
statenos = 1:6
xs, ψ = make_eigenfunctions(dh; statenos, nx=500)
plot(xs./π, 𝑈.(xs), label=false)
plot!(xs./π, abs2.(ψ) .+ dh.ε[statenos]', label=statenos) # in this system we can have (two-fold) degeneracies, so extracting real wf's is not so straightforward. Therefore, plotting densities

# study wanniers
@time compute_wanniers!(dh, targetlevels=7:12);
xs, ψ, w = make_wannierfunctions(dh; nx=500);
plot(xs./π, 𝑈.(xs), legend=false, ylims=(-V₀, V₀), xlabel="x / π");
dh.wanniers.pos
w_real = make_wanniers_real(w)
plot!(xs./π, w_real .+ dh.wanniers.E')

J = compute_tunneling(dh; i=1, j=2)
angle(J)
abs(J)

compute_tb_hamiltonian(dh) .|> abs

# calculating the s-p band coupling 𝜂 = ⟨𝑤ᵢ|𝑥|𝑤ⱼ⟩
𝑓(x) = x
X = XSpaceHamiltonians.p_space_matrix(dh; 𝑓, iseven=false)
levels_lo = 1:6
compute_wanniers!(dh, targetlevels=levels_lo);
w_s = dh.wanniers.V
levels_hi = 7:12
compute_wanniers!(dh, targetlevels=levels_hi);
w_p = dh.wanniers.V
n = 3
η10 = abs(w_s[:, n]' * dh.V[:, levels_lo]' * X * dh.V[:, levels_hi] * w_p[:, n])
K = 0.5
γ = K*η10/2 # matches the value of 0.13 in the paper if no division by π in `η10` is used