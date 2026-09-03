################ p-space operators ################

# The following methods create matrices of momentum operators in p-space, such as ⟨𝑗′ʸ𝑗′ˣ|𝑝²|𝑗ʸ𝑗ˣ⟩. These are used in the p-space approach.

"""
Return the momentum-squared matrix (𝛿𝑝)² = -𝛿²Δ in `D = length(L)` dimensions. If `basis=:cis`, then return (𝛿𝑝 + 𝑞)² with quasimomenta provided in `qs`.
A real `Diagonal` matrix is returned.
"""
function make_p²_matrix(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol, qs=zeros(R, length(L))) where R <: AbstractFloat
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
    error("make_p²_matrix not implemented for $D dimensions and $basis basis.")
end

"Return the matrix of 𝑝ⁱ = -i𝛿𝜕ⁱ if `basis=:cis` and 𝛿𝜕ⁱ otherwise. The output is real in both cases."
function make_pⁱ_matrix(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol, i::Integer) where R <: AbstractFloat
    if i == 1
        return make_pˣ_matrix(L, M, δ, basis)
    elseif i == 2
        return make_pʸ_matrix(L, M, δ, basis)
    end
    error("make_pⁱ_matrix not implemented for $i=1.")
end

"Return the matrix of 𝑝ˣ = -i𝛿𝜕ˣ if `basis=:cis` and 𝛿𝜕ˣ otherwise. The output is real in both cases."
function make_pˣ_matrix(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol) where R <: AbstractFloat
    D = length(L)
    Lˣ = L[1]
    if basis == :cis
        PI = R(π)
        if D == 1
            return Diagonal([2PI * δ * jˣ/Lˣ for jˣ in -M:M])
        elseif D == 2
            return Diagonal([2PI * δ * jˣ/Lˣ for jʸ in -M:M for jˣ in -M:M])
        end
    elseif basis == :sin
        ∂ˣ = zeros(R, M^D, M^D)
        if D == 1
            ##
        elseif D == 2
            @floop for jˣ in 1:M
                for jʸ in 1:M, jˣ′ in 1+isodd(jˣ):2:M
                    ∂ˣ[(jʸ-1)M+jˣ′, (jʸ-1)M+jˣ] = 4δ * jˣ′*jˣ/(Lˣ * (jˣ′^2 - jˣ^2))
                end
            end
        end
        return ∂ˣ
    else # basis == :cos
        b = M + 1
        ∂ˣ_cos = zeros(R, b^D, b^D) # different name to prevent Core.Box
        if D == 1
            ##
        elseif D == 2
            @floop for jˣ in 0:M
                ζˣ = ifelse(jˣ == 0, 2, 1)
                for jʸ in 0:M, jˣ′ in iseven(jˣ):2:M
                    ζˣ′ = ifelse(jˣ′ == 0, 2, 1)
                    ∂ˣ_cos[jʸ*b+jˣ′+1, jʸ*b+jˣ+1] = 4δ * jˣ^2/(Lˣ * (jˣ′^2 - jˣ^2)) / √(ζˣ′*ζˣ)
                end
            end
        end
        return ∂ˣ_cos
    end
    error("make_pˣ_matrix not implemented for $D dimensions and $basis basis.")
end

"Return the matrix of 𝑝ʸ = -i𝛿𝜕ʸ if `basis=:cis` and 𝛿𝜕ʸ otherwise. The output is real in both cases."
function make_pʸ_matrix(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol) where R <: AbstractFloat
    D = length(L)
    Lʸ = L[2]
    if basis == :cis
        PI = R(π)
        return Diagonal([2PI * δ * jʸ/Lʸ for jʸ in -M:M for jˣ in -M:M])
    elseif basis == :sin
        ∂ʸ = zeros(R, M^D, M^D)
        @floop for jʸ in 1:M
            for jʸ′ in 1+isodd(jʸ):2:M, jˣ in 1:M
                ∂ʸ[(jʸ′-1)M+jˣ, (jʸ-1)M+jˣ] = 4δ * jʸ′*jʸ/(Lʸ * (jʸ′^2 - jʸ^2))
            end
        end
        return ∂ʸ
    else # basis == :cos
        b = M + 1
        ∂ʸ_cos = zeros(R, b^D, b^D) # different name to prevent Core.Box
        if D == 1
            ##
        elseif D == 2
            @floop for jʸ in 0:M
                ζʸ = ifelse(jʸ == 0, 2, 1)
                for jʸ′ in iseven(jʸ):2:M, jˣ in 0:M
                    ζʸ′ = ifelse(jʸ′ == 0, 2, 1)
                    ∂ʸ_cos[jʸ′*b+jˣ+1, jʸ*b+jˣ+1] = 4δ * jʸ^2/(Lʸ * (jʸ′^2 - jʸ^2)) / √(ζʸ′*ζʸ)
                end
            end
        end
        return ∂ʸ_cos
    end
    error("make_pʸ_matrix not implemented for $D dimensions and $basis basis.")
end

"""
Return the momentum-squared tensor (𝛿𝑝)² = -𝛿²Δ in `D = length(L)` dimensions. If `basis=:cis`, then return (𝛿𝑝 + 𝑞)² with quasimomenta provided in `qs`.
A flattened vector is returned.
"""
function make_p²(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol, qs=zeros(R, length(L))) where R <: AbstractFloat
    D = length(L)
    PI = R(π) # for type stability. Otherwise 2*π*2f0 gets converted to Float64 (and so does 2f0*2π, while 2f0*2*π does not...)
    if basis == :cis
        cx = 2PI*δ/L[1]
        M_range = [0:M-1; -M:-1]
        if D == 1
            return [(cx*jˣ + qs[1])^2 for jˣ in M_range]
        elseif D == 2
            cy = 2PI*δ/L[2]
            return [((cx*jˣ + qs[1])^2 + (cy*jʸ + qs[2])^2) for jʸ in M_range for jˣ in M_range]
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
            return [(bx*jˣ)^2 + (by*jʸ)^2 for jʸ in j₁:M for jˣ in j₁:M]
        end
    end
    error("make_p² not implemented for $D dimensions and $basis basis.")
end

################ Fourier (p-space) images ################

# The following methods create flattened vectors representing Fourier images of momentum operators, such as ℱ[𝑝²(𝑥, 𝑦)](𝑗ʸ𝑗ˣ). These are used in the x-space approach.

"""
Return the Fourier image of 𝑝ⁱ = -i𝛿𝜕ⁱ if `basis=:cis` and 𝛿𝜕ⁱ otherwise. The output is real in both cases.
A flattened vector is returned.
"""
function make_pⁱ(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol, i::Integer) where R <: AbstractFloat
    if i == 1
        return make_pˣ(L, M, δ, basis)
    elseif i == 2
        return make_pʸ(L, M, δ, basis)
    end
    error("make_pⁱ not implemented for i=$i.")
end

"""
Return the Fourier image of 𝑝ˣ = -i𝛿𝜕ˣ if `basis=:cis` and 𝛿𝜕ˣ otherwise. The output is real in both cases.
A flattened vector is returned.
"""
function make_pˣ(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol) where R <: AbstractFloat
    D = length(L)
    PI = R(π)
    if basis == :cis
        M_range = [0:M-1; -M:-1]
        cx = 2PI*δ/L[1]
        if D == 1
            return [cx * jˣ for jˣ in M_range]
        elseif D == 2
            return [cx * jˣ for jʸ in M_range for jˣ in M_range]
        end
    else # basis == :sin || basis == :cos # TODO reconsider because inverse transform for sin is cos and for cos is sin
        j₁ = basis == :sin ? 1 : 0
        bx = (-1)^j₁ * PI*δ/L[1] # different name than `cx` above to prevent Core.Box. The prefactor is because we need a minus in the sin case.
        if D == 1
            return [bx*jˣ for jˣ in j₁:M] 
        elseif D == 2
            return [bx*jˣ for jʸ in j₁:M for jˣ in j₁:M]
        end
    end
    error("make_pˣ not implemented for $D dimensions and $basis basis.")
end

"""
Return the Fourier image of 𝑝ʸ = -i𝛿𝜕ʸ if `basis=:cis` and 𝛿𝜕ʸ otherwise. The output is real in both cases.
A flattened vector is returned.
"""
function make_pʸ(L::AbstractVector{R}, M::Integer, δ::R, basis::Symbol) where R <: AbstractFloat
    D = length(L)
    PI = R(π)
    if basis == :cis
        M_range = [0:M-1; -M:-1]
        cy = 2PI*δ/L[2]
        return [cy * jʸ for jʸ in M_range for jˣ in M_range]
    else # basis == :sin || basis == :cos # TODO reconsider because inverse transform for sin is cos and for cos is sin
        j₁ = basis == :sin ? 1 : 0
        by = (-1)^j₁ * PI*δ/L[2] # different name than `cx` above to prevent Core.Box. The prefactor is because we need a minus in the sin case.
        if D == 2
            return [by*jʸ for jʸ in j₁:M for jˣ in j₁:M]
        end
    end
    error("make_pʸ not implemented for $D dimensions and $basis basis.")
end