using XSpaceHamiltonians

using Plots
plotlyjs()
cmap_rainbow = cgrad(:rainbow_bgyrm_35_85_c69_n256);
cmap_cyclic = cgrad(:cyclic_mrybm_35_75_c68_n256);
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
M = 70
N = 2M + 1 # see how the potential looks with number of points that will be used for FFT
xs = range(xlimits[1], xlimits[2], N)
ys = range(ylimits[1], ylimits[2], N)
surface(xs, ys, 𝑈)
surface(xs, ys, (x, y) -> 𝐴_x(x, y)^2 + 𝐴_y(x, y)^2)

# Calculate
@time dh = DenseHamiltonian(xlimits, ylimits; isperiodic=false, M, 𝑈, 𝐴_x, 𝐴_y);
@time diagonalize!(dh, nev=5);
dh.ε[1:5]

stateno = 1
@time xs, ys, ψ = make_wavefunction(dh, stateno, 100, 100);
surface(xs, ys, abs2.(ψ)', xlabel="x/w_0", ylabel="y/w_0", c=cmap_rainbow, title=L"|\psi|^2")
heatmap(xs, ys, angle.(ψ)', xlabel="x/a", ylabel="y/a", c=cmap_cyclic)

using Measures
theme(:dark, size=(1200, 500))
fig_abs = heatmap(xs, ys, abs2.(ψ)', xlabel=L"x/w_0", ylabel=L"y/w_0", c=cmap_rainbow, title=L"|\psi|^2");
fig_phi = heatmap(xs, ys, angle.(ψ)' ./ π, c=:viridis, xlabel=L"x/w_0", ylabel=L"y/w_0", title="phase", cbar_title="phase ("*L"\pi"*" rad)");
fig_phi = heatmap(xs, ys, angle.(ψ)' ./ π, c=cmap_cyclic, xlabel=L"x/w_0", ylabel=L"y/w_0", title="phase", cbar_title="phase ("*L"\pi"*" rad)");
plot(fig_abs, fig_phi, plot_title=L"\beta=0, \epsilon=%$ϵ"*". State no. $stateno, "*L"E=%$(round(dh.ε[stateno], sigdigits=3))", bottommargin=5mm, leftmargin=7mm, plot_titlefontcolor=:white)
savefig("lg_wf_xy4_epsilon$(ϵ)_state$stateno.png")