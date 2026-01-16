using XSpaceHamiltonians

using Plots
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_phase = cgrad(:RdBu_9);
theme(:dark, size=(600, 500))

function 𝑈(x::Real, y::Real)
    2(ϵ / (1 + ϵ^2*(x^2+y^2)))^2
end

function 𝐴_x(x::Real, y::Real)
    ϵ^2 * √(x^2+y^2) / (1 + ϵ^2*(x^2+y^2)) * sin(atan(y, x))
end

function 𝐴_y(x::Real, y::Real)
    -ϵ^2 * √(x^2+y^2) / (1 + ϵ^2*(x^2+y^2)) * cos(atan(y, x))
end

Float = Float32 # operating type

ϵ::Float = 1

xlimits = (-4, 4) .|> Float
ylimits = (-4, 4) .|> Float

# plot potential
M = 50
N = 2M + 1 # see how the potential looks with number of points that will be used for FFT
xs = range(xlimits[1], xlimits[2], N)
ys = range(ylimits[1], ylimits[2], N)
surface(xs, ys, 𝑈)
surface(xs, ys, (x, y) -> 𝐴_x(x, y)^2 + 𝐴_y(x, y)^2)

# Calculate
@time xh = XSpaceHamiltonian{:dense}(𝑈, xlimits, ylimits; isperiodic=false, M, 𝐴_x, 𝐴_y)
@time diagonalize!(xh, nev=5)
xh.ε[1:5]

stateno = 1
@time xs, ys, ψ = make_eigenfunction(xh, stateno, 100, 100);
surface(xs, ys, abs2.(ψ[1])', xlabel="x/w_0", ylabel="y/w_0", c=cmap_rainbow, title="|ψ|^2")
heatmap(xs, ys, angle.(ψ[1])', xlabel="x/a", ylabel="y/a", c=cmap_phase)