"""
Return the momentum-squared matrix (𝛿𝑝)² = -𝛿²Δ in `D = length(L)` dimensions. If `basis=:cis`, then return (𝛿𝑝 + 𝑞)² with quasimomenta provided in `qs`.
A real `Diagonal` matrix is returned.
"""
function make_p²(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol, qs=zeros(R, length(L))) where R <: AbstractFloat
    D = length(L)
    PI = R(π) # for type stability. Otherwise 2*π*2f0 gets converted to Float64 (and so does 2f0*2π, while 2f0*2*π does not...)
    if basis == :cis
        cx = 2PI*δ/L[1]
        if D == 1
            return [(cx*jˣ + qs[1])^2 for jˣ in -M:M] |> Diagonal
        elseif D == 2
            cy = 2PI*δ/L[2]
            return [((cx*jˣ + qs[1])^2 + (cy*jʸ + qs[2])^2) for jʸ in -M:M for jˣ in -M:M] |> Diagonal
        # else # arbitrary `D`
        #     Ms = ntuple(Returns(-M:M), D)
        #     return [sum(@. (J * 2PI*δ/L + qs)^2) for J in Iterators.product(Ms...)] |> vec |> Diagonal 
        end
    else # basis == :sin || basis == :cos
        bx = PI*δ/L[1] # different name than `cx` above to prevent Core.Box
        j₁ = basis == :sin ? 1 : 0
        if D == 1
            return [(bx*jˣ)^2 for jˣ in j₁:M] |> Diagonal
        elseif D == 2
            by = PI*δ/L[2]
            return [(bx*jˣ)^2 + (by*jʸ)^2 for jʸ in j₁:M for jˣ in j₁:M] |> Diagonal
        end
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
            return Diagonal([2PI * δ * jˣ/Lx for jˣ in -M:M])
        elseif D == 2
            return Diagonal([2PI * δ * jˣ/Lx for jʸ in -M:M for jˣ in -M:M])
        end
    elseif basis == :sin
        ∂_x = zeros(R, M^D, M^D)
        if D == 1
            ##
        elseif D == 2
            @floop for jˣ in 1:M
                for jʸ in 1:M, jˣ′ in 1+isodd(jˣ):2:M
                    ∂_x[(jʸ-1)M+jˣ′, (jʸ-1)M+jˣ] = 4δ * jˣ′*jˣ/(Lx * (jˣ′^2 - jˣ^2))
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
        return Diagonal([2PI * δ * jʸ/Ly for jʸ in -M:M for jˣ in -M:M])
    elseif basis == :sin
        ∂_y = zeros(R, M^D, M^D)
        @floop for jʸ in 1:M
            for jʸ′ in 1+isodd(jʸ):2:M, jˣ in 1:M
                ∂_y[(jʸ′-1)M+jˣ, (jʸ-1)M+jˣ] = 4δ * jʸ′*jʸ/(Ly * (jʸ′^2 - jʸ^2))
            end
        end
        return ∂_y
    end
end

"""
Return the momentum-squared tensor (𝛿𝑝)² = -𝛿²Δ in `D = length(L)` dimensions. If `basis=:cis`, then return (𝛿𝑝 + 𝑞)² with quasimomenta provided in `qs`.
A real rank-D tensor is returned.
"""
function make_p²_tensor(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol, qs=zeros(R, length(L))) where R <: AbstractFloat
    D = length(L)
    PI = R(π) # for type stability. Otherwise 2*π*2f0 gets converted to Float64 (and so does 2f0*2π, while 2f0*2*π does not...)
    if basis == :cis
        cx = 2PI*δ/L[1]
        M_range = [0:M-1; -M:-1]
        if D == 1
            return [(cx*jˣ + qs[1])^2 for jˣ in M_range]
        elseif D == 2
            cy = 2PI*δ/L[2]
            return [((cx*jˣ + qs[1])^2 + (cy*jʸ + qs[2])^2) for jˣ in M_range, jʸ in M_range]
        # else # arbitrary `D`
        #     Ms = ntuple(Returns(M_range), D)
        #     L_r = reverse(L) # because `J` will be be enumerated in "reverse" order compared to that used by us (first dimesnion is 𝑥, then 𝑦, ...)
        #     qs_r = reverse(L)
        #     return [sum(@. (J * 2PI*δ/L_r + qs_r)^2) for J in Iterators.product(Ms...)]
        end
    else # basis == :sin || basis == :cos
        bx = PI*δ/L[1] # different name than `cx` above to prevent Core.Box
        j₁ = basis == :sin ? 1 : 0
        if D == 1
            return [(bx*jˣ)^2 for jˣ in j₁:M]
        elseif D == 2
            by = PI*δ/L[2]
            return [(bx*jˣ)^2 + (by*jʸ)^2 for jˣ in j₁:M, jʸ in j₁:M]
        end
    end
    return R[] # for type stability
end

"""
Return the tensor of 𝑝ᵢ = -iδ𝜕ᵢ if `basis=:cis` and δ𝜕ᵢ otherwise. The output is real in both cases."
A real rank-D tensor is returned.
"""
function make_p_i_tensor(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol, i::Integer) where R <: AbstractFloat
    if i == 1
        return make_p_x_tensor(L, M, δ, basis)
    elseif i == 2
        return make_p_y_tensor(L, M, δ, basis)
    end
end

"""
Return the tensor of 𝑝ₓ = -iδ𝜕ₓ if `basis=:cis` and δ𝜕ₓ otherwise. The output is real in both cases.
A real rank-D tensor is returned.
"""
function make_p_x_tensor(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol) where R <: AbstractFloat
    D = length(L)
    PI = R(π)
    if basis == :cis
        M_range = [0:M-1; -M:-1]
        cx = 2PI*δ/L[1]
        if D == 1
            return [cx * jˣ for jˣ in M_range]
        elseif D == 2
            return [cx * jˣ for jˣ in M_range, jʸ in M_range]
        end
    else # basis == :sin || basis == :cos # TODO reconsider because inverse transform for sin is cos and for cos is sin
        j₁ = basis == :sin ? 1 : 0
        bx = (-1)^j₁ * PI*δ/L[1] # different name than `cx` above to prevent Core.Box. The prefactor is because we need a minus in the sin case.
        if D == 1
            return [bx*jˣ for jˣ in j₁:M] 
        elseif D == 2
            return [bx*jˣ for jˣ in j₁:M, jʸ in j₁:M]
        end
    end
end

"""
Return the tensor of 𝑝ʸ = -i𝛿𝜕ʸ if `basis=:cis` and 𝛿𝜕ʸ otherwise. The output is real in both cases.
A real rank-D tensor is returned.
"""
function make_p_y_tensor(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol) where R <: AbstractFloat
    D = length(L)
    # D == 1 && @error "Only D ≥ 2 supported. For D = 1, use make_p_x."; return
    PI = R(π)
    if basis == :cis
        M_range = [0:M-1; -M:-1]
        cy = 2PI*δ/L[2]
        return [cy * jʸ for jˣ in M_range, jʸ in M_range]
    else # basis == :sin || basis == :cos # TODO reconsider because inverse transform for sin is cos and for cos is sin
        j₁ = basis == :sin ? 1 : 0
        by = (-1)^j₁ * PI*δ/L[2] # different name than `cx` above to prevent Core.Box. The prefactor is because we need a minus in the sin case.
        if D == 2
            return [by*jʸ for jˣ in j₁:M, jʸ in j₁:M]
        end
    end
end