"""
Return the momentum-squared matrix (𝛿𝑝)² = -𝛿²Δ in `D = length(L)` dimensions. If `basis=:cis`, then return (𝛿𝑝 + 𝑞)² with quasimomenta provided in `qs`.
A real `Diagonal` matrix is returned.
"""
function make_p²(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol, qs=zeros(R, length(L))) where R <: AbstractFloat # ntuple(Returns(zero(R)), length(L))
    D = length(L)
    PI = R(π) # for type stability. Otherwise 2*π*2f0 gets converted to Float64 (and so does 2f0*2π, while 2f0*2*π does not...)
    if basis == :cis
        cx = 2PI*δ/L[1]
        if D == 1
            return [(cx*jx + qs[1])^2 for jx in -M:M] |> Diagonal
        elseif D == 2
            cy = 2PI*δ/L[2]
            return [((cx*jx + qs[1])^2 + (cy*jy + qs[2])^2) for jx in -M:M for jy in -M:M] |> Diagonal # in D-dimensions: Ms = [-M:M for _ in 1:D]; L_r = reverse(L); [sum(@. (J * 2PI*δ/L + qs)^2) for J in Iterators.product(Ms...)] |> vec |> Diagonal 
        end
    elseif basis == :sin
        bx = PI*δ/L[1] # different name than `cx` above to prevent Core.Box
        if D == 1
            return [(bx*jx)^2 for jx in 1:M] |> Diagonal
        elseif D == 2
            by = PI*δ/L[2]
            return [(bx*jx)^2 + (by*jy)^2 for jx in 1:M for jy in 1:M] |> Diagonal
        end
    else
        error("make_p² not implemented for basis $basis.")
    end
    return Diagonal(R[]) # for type stability
end

"Return the matrix of 𝑝ᵢ = -iδ𝜕ᵢ if `basis=:cis` and δ𝜕ᵢ otherwise. The output is real in both cases."
function make_p_i(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol, i::Integer) where R <: AbstractFloat
    if i == 1
        return make_p_x(L, M, δ, basis)
    elseif i == 2
        return make_p_y(L, M, δ, basis)
    end
end

"Return the matrix of 𝑝ₓ = -iδ𝜕ₓ if `basis=:cis` and δ𝜕ₓ otherwise. The output is real in both cases."
function make_p_x(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol) where R <: AbstractFloat
    D = length(L)
    Lx = L[1]
    if basis == :cis
        PI = R(π)
        if D == 1
            return Diagonal([2PI * δ * jx/Lx for jx in -M:M])
        elseif D == 2
            return Diagonal([2PI * δ * jx/Lx for jx in -M:M for jy in -M:M])
        end
    elseif basis == :sin
        ∂_x = zeros(R, M^D, M^D)
        if D == 1
            ##
        elseif D == 2
            @floop for jx in 1:M
                for j′x in 1+isodd(jx):2:M, jy in 1:M
                    ∂_x[(j′x-1)M+jy, (jx-1)M+jy] = 4δ * j′x*jx/(Lx * (j′x^2 - jx^2))
                end
            end
        end
        return ∂_x
    end
end

"Return the matrix of 𝑝_𝑦 = -i𝛿𝜕_𝑦 if `basis=:cis` and 𝛿𝜕_𝑦 otherwise. The output is real in both cases."
function make_p_y(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol) where R <: AbstractFloat
    D = length(L)
    # D == 1 && @error "Only D ≥ 2 supported. For D = 1, use make_p_x."; return
    Ly = L[2]
    if basis == :cis
        PI = R(π)
        return Diagonal([2PI * δ * jy/Ly for jx in -M:M for jy in -M:M])
    elseif basis == :sin
        ∂_y = zeros(R, M^D, M^D)
        @floop for jx in 1:M
            for jy in 1:M, j′y in 1+isodd(jy):2:M
                ∂_y[(jx-1)M+j′y, (jx-1)M+jy] = 4δ * j′y*jy/(Ly * (j′y^2 - jy^2))
            end
        end
        return ∂_y
    end
end