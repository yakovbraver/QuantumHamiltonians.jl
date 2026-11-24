module QuantumHamiltonians

import FFTW
import OrdinaryDiffEq as ODE
import OrdinaryDiffEqLinear as ODE_LIN
import OrdinaryDiffEqExponentialRK as ODE_EXP
import SciMLOperators
import KrylovKit
using ArnoldiMethod: partialschur, partialeigen
using LinearAlgebra: Hermitian, Diagonal, diag, diagind, diagview, factorize, lu!, eigen, dot, mul!, normalize!, copy_adjoint!
import LinearAlgebra: ldiv! # overloaded in XSpaceHamiltonians for preconditioning
import LinearAlgebra as LA # mainly for the identity operator LA.I
import LinearMaps as LM
import LinearSolve as LS
import NonlinearSolve as NLS
using SparseArrays
using FLoops: @floop
using LoopVectorization: @turbo

export PSpaceHamiltonian, diagonalize!, make_eigenfunction, make_eigenfunctions, matrix_density, make_wavefunction,
       compute_wanniers!, make_wannierfunctions, make_wanniers_real, compute_tunneling, compute_tb_hamiltonian,
       propagate, get_EμN, bdg_spectrum, find_stationary
export StateVector, XSpaceHamiltonian

include("PSpaceHamiltonian.jl")
include("Wanniers.jl")
include("FourierTransformerP.jl")
include("DenseHamiltonian.jl")
include("SparseHamiltonian.jl")
include("momentum.jl")

include("StateVector.jl")
include("FourierTransformerX.jl")
include("XSpaceHamiltonian.jl")
include("XSpacePreconditioners.jl")

include("gpe_stationary.jl")
include("gpe_dynamics_pspace.jl")
include("split_step_xspace.jl")
include("gpe_dynamics_xspace.jl")
include("BdGMaps.jl")

"A linear map holding a `LinearSolve` object, used for applying the inverse map."
struct LinSolveLinMap{T,L} <: LM.LinearMap{T}
    linsolve::L
    size::Dims{2}
end

Base.size(lm::LinSolveLinMap) = lm.size

function LM._unsafe_mul!(y, lm::LinSolveLinMap, x::AbstractVector)
    copy!(lm.linsolve.b, x)
    sol = LS.solve!(lm.linsolve) # `solve!` allocates up to 50 KiB :(
    copy!(y, sol.u)
    # println(sol.stats) # TODO add verbose option
end

"""
Calculate inner product ∫𝑣𝑤d𝑥 (but without d𝑥) between `v` and `w` representing single-component x-space vectors.
For cis and sin basis, this is just a dot, but for cos we have to treat the edges.
The information of the basis and problem dimensions is contained in `qh`.
"""
@views function inner_prod(v::AbstractVector{<:Number}, w::AbstractVector{<:Number}, qh::Union{XSpaceHamiltonian, PSpaceHamiltonian})
    s = if qh.basis != :cos
        dot(v, w)
    else
        D = length(qh.xlims)
        if D == 1
            dot(v, w) - (v[1]w[1] + v[end]w[end]) / 2
        elseif D == 2
            # create reshaped views for convenient indexing
            v_tensor = reshape(v, size(qh.ft.plan_forward))
            w_tensor = reshape(w, size(qh.ft.plan_forward))
            dot(v, w) -
            ( dot(v_tensor[1, 2:end-1]  , w_tensor[1, 2:end-1])   +
              dot(v_tensor[end, 2:end-1], w_tensor[end, 2:end-1]) +
              dot(v_tensor[2:end-1, 1]  , w_tensor[2:end-1, 1])   +
              dot(v_tensor[2:end-1, end], w_tensor[2:end-1, end])) / 2 -
            (v_tensor[1, 1]'*w_tensor[1, 1] + v_tensor[1, end]'*w_tensor[1, end] + v_tensor[end, 1]'*w_tensor[end, 1] + v_tensor[end, end]'*w_tensor[end, end]) * 3/4
        else
            error("inner_prod not implemented for cos basis in $D dimensions.")
        end
    end
    dV = prod(qh.ft.xs[2, i] - qh.ft.xs[1, i] for i in axes(qh.ft.xs, 2)) # volume element
    return s*dV
end

"""
Calculate the squared norm ‖𝑣‖² = ∫|𝑣|²d𝑉 for `v` representing single-component x-space vectors.
For cis and sin basis, this is `sum(abs2, v)`, but for cos we have to treat the edges.
The information of the basis and problem dimensions is contained in `xh`.
"""
@views function norm²(v::AbstractVector{<:Number}, qh::Union{XSpaceHamiltonian, PSpaceHamiltonian})
    s = if qh.basis != :cos
        sum(abs2, v)
    else
        D = length(qh.xlims)
        if D == 1
            sum(abs2, v) - (abs2(v[1]) + abs2(v[end])) / 2
        elseif D == 2
            # create reshaped views for convenient indexing
            v_tensor = reshape(v, size(qh.ft.plan_forward))
            sum(abs2, v) -
            (sum(abs2, v_tensor[1, 2:end-1]  )  +
             sum(abs2, v_tensor[end, 2:end-1])  +
             sum(abs2, v_tensor[2:end-1, 1]  )  +
             sum(abs2, v_tensor[2:end-1, end])) / 2 -
            (abs2(v_tensor[1, 1]) + abs2(v_tensor[1, end]) + abs2(v_tensor[end, 1]) + abs2(v_tensor[end, end])) * 3/4
        else
            error("norm² not implemented for cos basis in $D dimensions.")
        end
    end
    dV = prod(qh.ft.xs[2, i] - qh.ft.xs[1, i] for i in axes(qh.ft.xs, 2)) # volume element
    return s*dV
end

"""
Calculate the integral ∫𝑣d𝑉 for `v` representing an x-space state (either flattened or not) of a single component.
For cis and sin basis, this is `sum(v)`, but for cos we have to treat the edges.
The information of the basis and problem dimensions is contained in `xh`.
"""
@views function integrate(v::AbstractArray{<:Number}, qh::Union{XSpaceHamiltonian, PSpaceHamiltonian})
    s = if qh.basis != :cos
        sum(v)
    else
        D = length(qh.xlims)
        if D == 1
            sum(v) - (v[1] + v[end]) / 2
        elseif D == 2
            # create reshaped views for convenient indexing
            v_tensor = reshape(v, size(qh.ft.plan_forward))
            sum(v) -
            (sum(v_tensor[1, 2:end-1]  )  +
             sum(v_tensor[end, 2:end-1])  +
             sum(v_tensor[2:end-1, 1]  )  +
             sum(v_tensor[2:end-1, end])) / 2 -
            (v_tensor[1, 1] + v_tensor[1, end] + v_tensor[end, 1] + v_tensor[end, end]) * 3/4
        else
            error("integrate not implemented for cos basis in $D dimensions.")
        end
    end
    dV = prod(qh.ft.xs[2, i] - qh.ft.xs[1, i] for i in axes(qh.ft.xs, 2)) # volume element
    return s*dV
end

"""
Normalise an x-space flattened multi-component state `v` so that ∑ᶜ∫|𝑣ᶜ|²d𝑉 = 1.
The information of the basis and problem dimensions is contained in `qh`.
"""
function LA.normalize!(v::AbstractVector{T}, qh::Union{XSpaceHamiltonian, PSpaceHamiltonian}) where T <: Number
    (;B, nc) = qh
    N = zero(real(T))
    for c in 1:nc
        N += norm²(v[(c-1)B+1:c*B], qh)
    end
    v ./= √N
end

end
