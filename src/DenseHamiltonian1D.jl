"A type for storing the Wannier functions."
mutable struct Wanniers{R<:Real}
    targetlevels::Vector{Int} # numbers of quasienergy levels to use for constructing wanniers (this is used in the Floquet case)
    E::Vector{R} # mean energies
    pos::Vector{R} # positions (wannier centres)
    V::Matrix{Complex{R}} # position eigenvectors
end

"Default-construct an empty `Wanniers` object."
Wanniers{R}() where R <: Real = Wanniers(Int[], R[], R[], Complex{R}[;;])

"""
A type representing a spatial, possibly quasimomentum-dependent 1D Hamiltonian
    𝐻(𝑥) = (-i𝛿∂ₓ + 𝑞)² + 𝑈(𝑥)
as a dense matrix.
"""
mutable struct DenseHamiltonian1D{R<:Real,T<:Number} # in practice `T` shoudld be `R` if there is no 𝑞 and 𝑈 is even, or `Complex{R}` otherwise -- always check this. If this is not the case, probably your 𝑈 or 𝐴 do not return R's.
    xlims::Tuple{R, R}
    Lx::R # length along 𝑥
    δ::R # coefficient of the momentum term
    isperiodic::Bool
    M::Int # maximum harmonic number (will use -M:M for periodic, 1:M for nonperiodic)
    𝑈::Union{Function,Nothing}
    H::Matrix{T}
    ε::Vector{R} # eigenvalues
    V::Matrix{T} # eigenvectors matrix
    ε_q::Array{R,2} # ε_q[n, iqx] = `n`th band eigenvalue at quasimomentum at index `iqx`
    V_q::Array{T,3} # V_q[:, n, iqx] = `n`th band eigenvector at quasimomentum at index `iqx`
    wanniers::Wanniers{R}
end

"""
Construct a `DenseHamiltonian1D` object.
`M` is the maximum harmonic number. In the periodic case, the size of the Hamiltonian will be (`2M`) × (`2M`).
In nonperiodic case, the size will be `(2M+1)` × `(2M+1)`.
To make sure that the resulting Hamiltonian matrix is of the desired type `T`, the type of elements of `xlims`, `ylims`,
and the return type of the passed functions has to be the same. E.g., if all are `Float32`, then `T` will be `Float32` if only `𝑈` is passed,
and `ComplexF32` if `𝐴`'s are passed. Inconsistency in the types of arguments will result in widening.
`iseven` shows whether 𝑈 is an even function. If it is so, its Fourier image is real and even (𝑈 is assumed real),
so the constructed Hamiltonian will be real, and symmetric diagonalisation will be used.
"""
function DenseHamiltonian1D(xlims::Tuple{R,R}; isperiodic::Bool, iseven::Bool, M::Integer, δ::R=one(R), 𝑈::Union{Function,Nothing}=nothing) where R <: Real
    Lx = xlims[2] - xlims[1]

    if isperiodic
        N = 4M # number of points for FFT. This will yield harmonics from -2M to 2M
        dx = Lx/N
        xs = range(xlims[1], xlims[2]-dx, N)

        f = dx/Lx
        u = 𝑈.(xs)

        F = FFTW.plan_rfft(u)
        H = dft_to_matrix_1D(F * u * f, iseven) # initialising the Hamiltonian with the potential
    
        H += -Diagonal(R[-(2π*δ)^2 * (jx/Lx)^2 for jx in -M:M]) # adding to the Hamiltonian the term -δ²Δ
    else # non-periodic
        N = 2M + 1
        xs = range(xlims[1], xlims[2], N)
        dx = xs[2] - xs[1]

        f = dx/Lx
        u = 𝑈.(xs)

        F = FFTW.plan_r2r!(u, FFTW.REDFT00)
        (F * u) .*= f
        H = dct_to_matrix_1D(u) # initialising the Hamiltonian with the potential

        H += -Diagonal(R[-(π*δ)^2 * (jx/Lx)^2 for jx in 1:M]) # adding to the Hamiltonian the term -δ²Δ
    end
    
    return DenseHamiltonian1D(xlims, Lx, δ, isperiodic, M, 𝑈, H, R[], eltype(H)[;;], R[;;], eltype(H)[;;;], Wanniers{R}())
end

"""
Based on results of a real 1D fft `u`, return the matrix indexed by (𝑗′ₓ, 𝑗ₓ).
If potential is even (`iseven=true`), then a real matrix is constructed, using the real part of `u`.
"""
function dft_to_matrix_1D(u, iseven::Bool)
    if iseven # if potential is even, then the Fourier image must be real, so we create a real matrix and save only the real part
        H = zeros(real(eltype(u)), length(u), length(u))
        for (i, val) in enumerate(u)
            H[diagind(H, 1-i)] .= real(val) # fill lower triangle (including the diagonal)
            H[diagind(H, i-1)] .= real(val) # fill upper triangle (again including the diagonal)
        end
    else
        H = zeros(eltype(u), length(u), length(u))
        for (i, val) in enumerate(u)
            H[diagind(H, 1-i)] .= val  # fill lower triangle (including the diagonal)
            H[diagind(H, i-1)] .= val' # fill lower triangle (including the diagonal)
        end
    end
    return H
end

"""
Based on results of 1D DCT `u`, return the matrix indexed by (𝑗′ₓ, 𝑗ₓ).
"""
function dct_to_matrix_1D(u)
    N = length(u) # number of points used for FFT
    M = (N-1) ÷ 2 # maximum harmonic number
    H = Matrix{eltype(u)}(undef, M, M)
    @floop for jx in 1:M
        for j′x in 1:M
            j₋x = abs(j′x-jx)
            H[j′x, jx] = (u[j₋x+1] - u[j′x+jx+1]) / 2
        end
    end
    return H
end

"""
Construct eigenfunctions of state numbers `statenos` on a grid having `nx` points in `x` direction.
Return (`xs`, `ψ`). If `iqx` is passed, then construct `ψ` at the corresponding quasimomentum.
"""
function make_eigenfunctions(dh::DenseHamiltonian1D{R,T}; statenos::AbstractVector{<:Integer}, nx::Integer, iqx::Integer=0) where {R<:Real, T<:Number}
    (;Lx, xlims, M, V, V_q) = dh
    xs = range(0, Lx, nx) # these are the differences `x - xlims[1]`, with `x ∈ xlims`
    nstates = length(statenos)
    ψ = Matrix{complex(R)}(undef, nx, nstates) # construct complex wf even if Hamiltonian is real because degeneracies are possible
    for (is, stateno) in enumerate(statenos)
        if dh.isperiodic
            if iqx != 0 # if quasimomentum index has been passed
                @floop for (ix, x) in enumerate(xs)
                    ψ[ix, is] = sum(V_q[j, stateno, iqx]cis(2π*jx*x/Lx) for (j, jx) in enumerate(-M:M)) / √Lx
                end
            else # no quasimomentum index
                @floop for (ix, x) in enumerate(xs)
                    ψ[ix, is] = sum(V[j, stateno]cis(2π*jx*x/Lx) for (j, jx) in enumerate(-M:M)) / √Lx
                end
            end
        else # nonperiodic
            @floop for (ix, x) in enumerate(xs)
                ψ[ix, is] = sum(V[jx, stateno]sin(π*jx*x/Lx) for jx in 1:M) * 2 / √Lx
            end
        end
    end
    return xs .+ xlims[1], ψ # return "normal" coordinates, in `x ∈ xlims`
end

"""
Calculate `nev` lowest eigenvectors and eigenvalues using `ArnoldiMethod`.
If `nev=0` or not passed, then full diagonalisation using `LinearAlgebra` is performed.
"""
function diagonalize!(dh::DenseHamiltonian1D; nev::Integer=0)
    H = Hermitian(dh.H) # if `dh.H` is real, the appropriate routine will be selected automatically, no need to use `Symmetric` instead of `Hermitian`
    if nev == 0
        dh.ε, dh.V = eigen(H)
    else
        S, info = partialschur(dense_linear_map(H); nev, which=:LM, tol=1e-7); # `which=:SR` does not converge, so we use "shift-invert" (although shift is zero)
        @show info
        dh.V = S.Q
        dh.ε = inv.(real.(S.eigenvalues)) # invert back
    end
end

"""
Calculate eigenenergies for all quasimomenta in `qxs`.
Calculate `nev` lowest bands using `ArnoldiMethod`.
If `nev=0` or not passed, then full diagonalisation using `LinearAlgebra` is performed.
"""
function diagonalize!(dh::DenseHamiltonian1D{R,T}, qxs::AbstractVector{<:Real}; nev::Integer=0) where {R<:Real, T<:Number}
    (;M, Lx, δ) = dh
   
    nsaves = nev == 0 ? 2M+1 : nev # number of eigenvalues and eigenvectors to allocate
    dh.ε_q = Array{R,2}(undef, nsaves, length(qxs))
    dh.V_q = Array{T,3}(undef, 2M+1, nsaves, length(qxs))
    
    H_diag = diagview(dh.H)
    H_diag_copy = diag(dh.H) # a copy for restoring after the calculation
    U_diag = H_diag[(end+1) ÷ 2] # generally, `H = -Δ + U`, but this element is purely `U`, since Laplace is zero (see construction of Δ in `DenseHamiltonian1D` constructor)
    
    H = Hermitian(dh.H) # if `dh.H` is real, the appropriate routine will be selected automatically, no need to use `Symmetric` instead of `Hermitian`

    # iterate quasimomenta
    for (iqx, qx) in enumerate(qxs)
        # update diagonal
        H_diag .= [(2π*δ*jx/Lx + qx)^2 + U_diag for jx in -M:M]

        # diagonalise
        if nev == 0
            dh.ε_q[:, iqx], dh.V_q[:, :, iqx] = eigen(H)
        else
            S, info = partialschur(dense_linear_map(H); nev, which=:LM, tol=1e-7); # `which=:SR` does not converge, so we use "shift-invert" (although shift is zero)
            @show info
            dh.V_q[:, :, iqx] = S.Q
            dh.ε_q[:, iqx] = inv.(real.(S.eigenvalues)) # invert back
        end
    end
    H_diag .= H_diag_copy # restore initial values
end

"""
Calculate Wannier states using the energy eigenstates `targetlevels`. The vector `targetlevels` will be saved in `dh`.
`dh` is assumed to have been diagonalised, without quasimomentum.
"""
function compute_wanniers!(dh::DenseHamiltonian1D{R,T}; targetlevels::AbstractVector{<:Integer}) where {R<:Real, T<:Number}
    dh.wanniers.targetlevels = targetlevels # store the target levels
    minlevel = targetlevels[1]
    if dh.isperiodic
        X = @view(dh.V[2:end, targetlevels])' * @view(dh.V[1:end-1, targetlevels])
        pos_complex, dh.wanniers.V = eigen(X)
        pos_real = @. mod2pi(angle(pos_complex))/2π * dh.Lx + dh.xlims[1] # `mod2pi` converts the angle from [-π, π) to [0, 2π)
        sp = sortperm(pos_real)               # sort the eigenvalues
        dh.wanniers.pos = pos_real[sp]
        Base.permutecols!!(dh.wanniers.V, sp) # sort the eigenvectors in the same way
    else 
        n_w = length(targetlevels)
        X = Matrix{R}(undef, n_w, n_w) # position operator, will fill only upper triangle
        nj = size(dh.V, 1)
        for n in 1:n_w
            for n′ in 1:n
                X[n′, n] = (n == n′ ? dh.Lx/2 + dh.xlims[1] : 0) - 8dh.Lx/π^2*sum(dh.V[j, minlevel+n-1] * sum(dh.V[j′, minlevel+n′-1]*j*j′/(j^2-j′^2)^2
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
Return (`xs`, `ψ`, `w`). If `iqx` is passed, then construct `ψ` at the corresponding quasimomentum.
"""
function make_wannierfunctions(dh::DenseHamiltonian1D; nx::Integer)
    xs, ψ = make_eigenfunctions(dh; statenos=dh.wanniers.targetlevels, nx)
    w = ψ * dh.wanniers.V
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
function compute_tunneling(dh::DenseHamiltonian1D; i::Integer=1, j::Integer=2)
    wᵢ = dh.V[:, dh.wanniers.targetlevels] * dh.wanniers.V[:, i] # One wannier basis vector |𝑤ᵢ⟩ = ∑ₚ |𝜓ₚ⟩ 𝑉ᵢₚ
    wⱼ = dh.V[:, dh.wanniers.targetlevels] * dh.wanniers.V[:, j]
    return dot(wᵢ, dh.H, wⱼ)
end

"Compute TB Hamiltonian matrix, with elements ⟨𝑤ᵢ|𝐻|𝑤ⱼ⟩."
function compute_tb_hamiltonian(dh::DenseHamiltonian1D)
    dh.wanniers.V' * dh.V[:, dh.wanniers.targetlevels]' *  dh.H * dh.V[:, dh.wanniers.targetlevels] * dh.wanniers.V
end

"Return momentum-space matrix of a function `𝑓`, with problem geometry contained in `dh`. `iseven` is only relevant for periodic case, yielding real result for even `𝑓`."
function p_space_matrix(dh::DenseHamiltonian1D; 𝑓::Function, iseven::Bool=false)
    (;M, Lx, xlims, isperiodic) = dh

    if isperiodic
        N = 4M # number of points for FFT. This will yield harmonics from -2M to 2M
        dx = Lx/N
        xs = range(xlims[1], xlims[2]-dx, N)
        u = 𝑓.(xs) .* dx/Lx
        return dft_to_matrix_1D(FFTW.rfft(u), iseven)
    else # non-periodic
        N = 2M + 1
        xs = range(xlims[1], xlims[2], N)
        dx = xs[2] - xs[1]
        u = 𝑓.(xs) .* dx/Lx
        return dct_to_matrix_1D(FFTW.r2r!(u, FFTW.REDFT00))
    end
end