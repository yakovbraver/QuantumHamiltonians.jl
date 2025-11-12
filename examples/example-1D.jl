using XSpaceHamiltonians

using Plots, DelimitedFiles
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_phase = cgrad(:RdBu_9);
theme(:dark, size=(600, 500))

### Harmonic potential

function 𝑈(x::Real)
    x^2
end

Float = Float32 # operating type

xlimits = (-2π, 2π) .|> Float
M = 50

dh = DenseHamiltonian1D(xlimits; isperiodic=true, iseven=true, M, 𝑈)
@time diagonalize!(dh, nev=5);
dh.ε

# For real Hamiltonian, the eigenstates can be chosen real. But since we are using a complex cis basis, the coefficients are complex.
# However, if potential is even, the eigenstates have definite parity. E.g. ground state is even, and so can be expressed in terms of cos, i.e. the real part of cis.
# First excited state is odd, and so can be expressed in terms of sin, i.e. the imaginary part of cis.

plot(xs, 𝑈)
stateno = 5
part = iseven(stateno-1) ? real : imag
xs, ψ = make_wavefunction(dh, stateno, M)
plot!(xs, part(ψ) .+ dh.ε[stateno])

### Lattice potential

function 𝑈(x::Real)
    10sin(x)^2
end

Float = Float32 # operating type

xlimits = (-2π, 2π) .|> Float
M = 50

dh = DenseHamiltonian1D(xlimits; isperiodic=true, iseven=true, M, 𝑈)
@time diagonalize!(dh);
dh.ε

plot(xs, 𝑈, label=false)
stateno = 5
xs, ψ = make_wavefunction(dh, stateno, M)
plot!(xs, abs2.(ψ) .+ dh.ε[stateno], label="state $stateno") # in this system we can have (two-fold) degeneracies, so extracting real wf's is not so straightforward. Therefore, plotting densities
