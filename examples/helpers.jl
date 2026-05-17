# Helper functions for plotting 1D multicomponent wave functions and dynamics maps
using LinearAlgebra: dot

"Return optimal number `M` of harmonics for `basis`, approximately 2^p. Relevant for GPE-related calculations, where FFT is performed repeatedly."
function get_M(basis, p=7)
    basis == :cis ? (p == 7 ? 62 : p == 8 ? 122 : p == 9 ? 247 : p == 10 ? 500 : 1012) : # cis uses 2M+1 internally, which is odd and not a power of two, but there are some good choices
    basis == :sin ? 2^p-1 : 2^p
end

"Compute overlap ⟨𝜓₁|𝜓₂⟩"
function get_overlap(ψ₁, ψ₂, dx, basis; nc=1)
    O = zero(eltype(ψ₁))
    B = length(ψ₁) ÷ nc
    for i in 1:nc
        ψ₁ᵢ = @view ψ₁[(i-1)B+1:i*B]
        ψ₂ᵢ = @view ψ₂[(i-1)B+1:i*B]
        O += dot(ψ₁ᵢ, ψ₂ᵢ) * dx
        basis == :cos && (O -= (ψ₁ᵢ[1]'*ψ₂ᵢ[1] + ψ₁ᵢ[end]'*ψ₂ᵢ[end])/2 * dx)
    end
    return O
end

"""
Plot components for the case of complex `ψ`: abs and phase side by side.
`ψ` is either the return of `make_eigenfunctions` or `make_wavefunction`.
"""
function plot_comps_complex(xs, ψ; stateno=1)
    if ndims(ψ) == 3 # for ψ returned by make_eigenfunctions
        ncomps = size(ψ, 2)
        figs = [plot() for _ in 1:2ncomps]
        for i in 1:2:2ncomps
            c = (i+1) ÷ 2 # component number
            figs[i]   = plot(xs, abs2.(ψ[:, c, stateno]), xlabel=L"x", ylabel=L"y", title=L"|\psi_{%$c}|^2")
            figs[i+1] = plot(xs, angle.(ψ[:, c, stateno]) ./ π, xlabel=L"x", ylabel=L"y", title=L"\arg(\psi_{%$c})")
        end
    elseif ndims(ψ) == 1 # for ψ returned by make_wavefunction
        ncomps = length(ψ)
        figs = [plot() for _ in 1:2ncomps]
        for i in 1:2:2ncomps
            c = (i+1) ÷ 2 # component number
            figs[i]   = plot(xs, abs2.(ψ[c]), xlabel=L"x", ylabel=L"y", title=L"|\psi_{%$c}|^2")
            figs[i+1] = plot(xs, angle.(ψ[c]) ./ π, xlabel=L"x", ylabel=L"y", title=L"\arg(\psi_{%$c})")
        end
    else
        println("ndims(ψ) = $ndims not supported.")
        return
    end
    plot(figs..., layout=(ncomps, 2), legend=false)
end

"Plot components for the case of real `ψ`."
function plot_comps(xs, ψ; stateno=1)
    # determine the number of components
    nc = ndims(ψ) == 1 ? length(ψ) ÷ length(xs) : size(ψ, 2)
    figs = [plot() for _ in 1:nc]
    if ndims(ψ) == 3 # for ψ returned by make_eigenfunctions
        for c in 1:nc
            figs[c] = plot(xs, ψ[:, c, stateno], xlabel=L"x", ylabel=L"y", title=L"\psi_{%$c}");
        end
    elseif ψ isa Vector{<:Vector} # for ψ returned by make_wavefunction
        for c in 1:nc
            figs[c] = plot(xs, ψ[c], xlabel=L"x", ylabel=L"y", title=L"\psi_{%$c}");
        end
    elseif ψ isa Vector{<:Number} # for ψ returned by find_stationary
        nx = length(xs)
        for c in 1:nc
            figs[c] = plot(xs, ψ[(c-1)nx+1:c*nx], xlabel=L"x", ylabel=L"y", title=L"\psi_{%$c}");
        end
    end
    plot(figs..., layout=(nc, 1), legend=false)
end

"Return a 1-component 2D (time-space) evolution map using the solution."
function make_map(xh, sol)
    U = Matrix{eltype(sol.u[1])}(undef, length(sol.u[1]), length(sol.u))
    xs = [] # initialise for storing coordinates
    for it in axes(U, 2)
        xs, temp = make_wavefunction(xh, sol.u[it])
        U[:, it] = temp[1]
    end
    return vec(xs), U
end

"Return a multi-component 2D (time-space) evolution map using the solution. Output format: u[x, t, c]"
function make_map_comps(xh, sol; itime=false, pad=0)
    nt = itime ? length(sol.u) ÷ 2 + 1 : length(sol.u) # for imaginary time, save only every second solution point (we don't need unnormalised solutions before the callback)

    # determine number of x-point for each component (if padding is used, then it's not simply length(sol.u[1]÷nc))
    D = length(xh.xlims) # number of spatial dimensions
    M = 2^pad * xh.ft.M
    B_padded = xh.basis == :cis ? (2M+1)^D :
               xh.basis == :sin ?      M^D : (M+1)^D

    U = Array{eltype(sol.u[1])}(undef, B_padded, nt, xh.nc) # u[x, t, c]
    xs = [] # initialise for storing coordinates
    for it in axes(U, 2)
        iu = itime ? 2it - 1 : it
        xs, temp = make_wavefunction(xh, sol.u[iu]; pad)
        for c in 1:xh.nc
            U[:, it, c] = temp[c]
        end
    end
    return vec(xs), U
end

function make_animation(U, xs)
    @gif for it in axes(U, 2)
        plot(xs, abs2.(U[:, it, 1]))
        plot!(xs, abs2.(U[:, it, 2]))
    end
end

function plot_comps_2D(xs, ys, ψ)
    nc = length(ψ)
    theme(:dark, size=(600, 200nc+20))
    figs = [plot() for _ in 1:2nc]
    for c in 1:nc
        figs[2(c-1)+1] = heatmap(xs, ys, abs2.(ψ[c])', c=CMAP)
        figs[2(c-1)+2] = heatmap(xs, ys, angle.(ψ[c])', c=:coolwarm);
    end
    plot(figs..., layout=(nc, 2))
end