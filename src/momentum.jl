"Return the momentum squared matrix 𝑝² = -𝛿Δ."
function make_p²(L::AbstractVector{<:Real}, M::Integer, δ::Real, basis::Symbol)
    D = length(L)
    PI = eltype(L)(π)
    if basis == :cis
        c = (2PI*δ)^2
        if D == 1
            return Diagonal([c * (jx/L[1])^2 for jx in -M:M])
        elseif D == 2
            return Diagonal([c * ((jx/L[1])^2 + (jy/L[2])^2) for jx in -M:M for jy in -M:M])
        end
    else
        c = (PI*δ)^2
        if D == 1
            return Diagonal([c * (jx/L[1])^2 for jx in 1:M])
        elseif D == 2
            return Diagonal([c * ((jx/L[1])^2 + (jy/L[2])^2) for jx in 1:M for jy in 1:M])
        end
    end
end

"Return the matrix of 𝑝ᵢ = -iδ𝜕ᵢ if `basis=:cis` and δ𝜕ᵢ otherwise. The output is real in both cases."
function make_p_i(L::AbstractVector{<:Real}, M::Integer, δ::Real, basis::Symbol, i::Int)
    if i == 1
        return make_p_x(L, M, δ, basis)
    elseif i == 2
        return make_p_y(L, M, δ, basis)
    end
end

"Return the matrix of 𝑝ₓ = -iδ𝜕ₓ if `basis=:cis` and δ𝜕ₓ otherwise. The output is real in both cases."
function make_p_x(L::AbstractVector{<:Real}, M::Integer, δ::Real, basis::Symbol)
    D = length(L)
    Lx = L[1]
    if basis == :cis
        PI = typeof(Lx)(π)
        if D == 1
            return Diagonal([2PI * δ * jx/Lx for jx in -M:M])
        elseif D == 2
            return Diagonal([2PI * δ * jx/Lx for jx in -M:M for jy in -M:M])
        end
    elseif basis == :sin
        ∂_x = zeros(typeof(Lx), M^D, M^D)
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

"Return the matrix of 𝑝_𝑦 = -i𝛿𝜕_𝑦 if `basis=:cis` and δ𝜕_𝑦 otherwise. The output is real in both cases."
function make_p_y(L::AbstractVector{<:Real}, M::Integer, δ::Real, basis::Symbol)
    D = length(L)
    # D == 1 && @error "Only D ≥ 2 supported. For D = 1, use make_p_x."; return
    Ly = L[2]
    PI = typeof(Ly)(π)
    if basis == :cis
        return Diagonal([2PI * δ * jy/Ly for jx in -M:M for jy in -M:M])
    elseif basis == :sin
        ∂_y = zeros(typeof(Ly), M^D, M^D)
        @floop for jx in 1:M
            for jy in 1:M, j′y in 1+isodd(jy):2:M
                ∂_y[(jx-1)M+j′y, (jx-1)M+jy] = 4δ * j′y*jy/(Ly * (j′y^2 - jy^2))
            end
        end
        return ∂_y
    end
end