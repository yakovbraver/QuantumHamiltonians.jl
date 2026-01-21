"A type for storing the Wannier functions."
mutable struct Wanniers{R<:AbstractFloat}
    targetlevels::Vector{Int} # numbers of energy levels to use for constructing wanniers
    E::Vector{R} # mean energies
    pos::Vector{R} # positions (wannier centres)
    V::Matrix{Complex{R}} # position eigenvectors
end

"Default-construct an empty `Wanniers` object."
Wanniers{R}() where R <: AbstractFloat = Wanniers(Int[], R[], R[], Complex{R}[;;])

"""
Calculate Wannier states using the energy eigenstates `targetlevels`. The vector `targetlevels` will be saved in `dh`.
`dh` is assumed to have been diagonalised, without quasimomentum.
Implemented for the 1-component case only.
"""
function compute_wanniers!(dh::XSpaceHamiltonian{:dense}; targetlevels::AbstractVector{<:Integer})
    dh.wanniers.targetlevels = targetlevels # store the target levels
    minlevel = targetlevels[1]
    Lx = dh.L[1]
    xlims = dh.xlims[1]
    if dh.basis == :cis
        X = @view(dh.V[2:end, targetlevels])' * @view(dh.V[1:end-1, targetlevels])
        pos_complex, dh.wanniers.V = eigen(X)
        pos_real = @. mod2pi(angle(pos_complex))/2π * Lx + xlims[1] # `mod2pi` converts the angle from [-π, π) to [0, 2π)
        sp = sortperm(pos_real)               # sort the eigenvalues
        dh.wanniers.pos = pos_real[sp]
        Base.permutecols!!(dh.wanniers.V, sp) # sort the eigenvectors in the same way
    elseif dh.basis == :sin
        n_w = length(targetlevels)
        R = typeof(dh.δ)
        X = Matrix{R}(undef, n_w, n_w) # position operator, will fill only upper triangle
        nj = size(dh.V, 1)
        for n in 1:n_w
            for n′ in 1:n
                X[n′, n] = (n == n′ ? Lx/2 + xlims[1] : 0) - 8Lx/π^2*sum(dh.V[j, minlevel+n-1] * sum(dh.V[j′, minlevel+n′-1]*j*j′/(j^2-j′^2)^2
                                                                         for j′ = (iseven(j) ? 1 : 2):2:nj) for j = 1:nj)
            end
        end
        dh.wanniers.pos, dh.wanniers.V = eigen(Hermitian(X))
    end
    dh.wanniers.E = transpose(dh.ε[targetlevels]) * abs2.(dh.wanniers.V) |> vec
end

"""
Construct Wannier functions `w` on a grid having `nx` points in `x` direction. All Wannier functions contained in `dh` are constructed.
In the process, energy eigenfunctions `ψ` are also constructed.
Return (`xs`, `ψ`, `w`).
This assumes that wanniers have been calculated; and this is only implemented for the 1-component case.
"""
function make_wannierfunctions(dh::XSpaceHamiltonian{:dense}; nx::Integer)
    xs, ψ = make_eigenfunctions(dh; statenos=dh.wanniers.targetlevels, nx)
    w = dropdims(ψ; dims=2) * dh.wanniers.V # drop the dimesion corresponding to the component number
    return xs, ψ, w
end

"""
Given complex coordinate-space wanniers `w`, which are actually purely real or purely imaginary,
construct a real array by extracting either the real or imaginary part, whichever is larger.
"""
function make_wanniers_real(w)
    w_real = real(w)
    w_imag = imag(w)
    w_result = similar(w_real)
    for i in axes(w, 2)
        if sum(abs, @view(w_real[:, i])) > sum(abs, @view(w_imag[:, i]))
            w_result[:, i] .= w_real[:, i] # can be optimised using `copyto!`
        else
            w_result[:, i] .= w_imag[:, i]
        end
    end
    return w_result
end

"Compute tunnelling element ⟨𝑤ᵢ|𝐻|𝑤ⱼ⟩."
function compute_tunneling(dh::XSpaceHamiltonian{:dense}; i::Integer=1, j::Integer=2)
    wᵢ = dh.V[:, dh.wanniers.targetlevels] * dh.wanniers.V[:, i] # one wannier basis vector |𝑤ᵢ⟩ = ∑ₚ |𝜓ₚ⟩ 𝑉ᵢₚ
    wⱼ = dh.V[:, dh.wanniers.targetlevels] * dh.wanniers.V[:, j]
    return dot(wᵢ, dh.H, wⱼ)
end

"Compute TB Hamiltonian matrix, with elements ⟨𝑤ᵢ|𝐻|𝑤ⱼ⟩."
function compute_tb_hamiltonian(dh::XSpaceHamiltonian{:dense})
    dh.wanniers.V' * dh.V[:, dh.wanniers.targetlevels]' * dh.H * dh.V[:, dh.wanniers.targetlevels] * dh.wanniers.V
end