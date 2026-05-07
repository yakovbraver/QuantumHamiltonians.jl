import KrylovKit.VectorInterface

"""
The type encoding the physical x-space state of the system. Contains a single field `data` that is an n-component vector of rank-D tensors.
`Basis` is needed only to distinguish the cos case because then the inner product is calculated a bit differently.
"""
struct StateVector{T, D, Basis}
    data::Vector{Array{T, D}}
end

"Convenience constructor where `Basis` is specified, but remaining parameters are determined automatically."
StateVector{Basis}(data::Vector{Array{T, D}}) where {Basis, T, D} = StateVector{T, D, Basis}(data)

# These methods enable iterating over a StateVector directly (as if iterating over the underlying `data` array)
Base.length(v::StateVector) = length(v.data)
Base.eltype(v::StateVector) = eltype(v.data)
Base.getindex(v::StateVector, i::Int) = v.data[i]
Base.iterate(v::StateVector, i=1) = i > length(v) ? nothing : (v[i], i+1)

# All remaining methods are needed for KrylovKit to be able to handle our custom `StateVector` type

function Base.similar(v::StateVector{T, D, Basis}, ::Type{S}) where {T, D, Basis, S}
    data = [similar(d, S) for d in v]
    return StateVector{Basis}(data)
end

function Base.similar(v::StateVector{T}) where T
    return similar(v, T)
end

function Base.show(io::IO, v::StateVector{T, D, Basis}) where {T, D, Basis}
    # Examples:
    # "2-component StateVector{Float64, 1, :cis} on a 5-point grid"
    # "3-component StateVector{Float64, 3, :sin} on a 5×6×7 grid"
    print(io, length(v.data), "-component StateVector{$T, $D, $Basis} on a ", size(v.data[1], 1),
          (D == 1 ? "-point" : prod("×$(size(v.data[1], i))" for i in 2:D)), " grid")
end

VectorInterface.scalartype(::Type{<:StateVector{T}}) where T = T

function VectorInterface.zerovector(v::StateVector{T, D, Basis}, ::Type{S}) where {T, D, Basis, S<:Number}
    StateVector{Basis}([zeros(S, size(d)) for d in v])
end

function VectorInterface.zerovector!(v::StateVector{T}) where T
    ZERO = zero(T)
    for vᶜ in v
        vᶜ .= ZERO
    end
    return v
end

VectorInterface.zerovector!!(v::StateVector) = VectorInterface.zerovector!(v)

VectorInterface.scale(v::StateVector, α::Number) = VectorInterface.scale!!(similar(v), v, α)

function VectorInterface.scale!(v::StateVector{T}, α::Number) where T
    for vᶜ in v
        LA.BLAS.scal!(T(α), vᶜ) # ~2x faster than `vᶜ .*= α`. Casting to `T` because KrylovKit solver sometimes passes a Bool here
    end
    return v
end

VectorInterface.scale!!(v::StateVector, α::Number) = VectorInterface.scale!(v, α)

function VectorInterface.scale!!(w::StateVector, v::StateVector, α::Number)
    for (vᶜ, wᶜ) in zip(v, w)
        @. wᶜ = α * vᶜ
    end
    return w
end

function VectorInterface.add(w::StateVector, v::StateVector, α::Number, β::Number)
    u = similar(v)
    if β === VectorInterface.One()
        for (vᶜ, wᶜ, uᶜ) in zip(v, w, u)
            @. uᶜ = α * vᶜ + wᶜ
        end
    else
        for (vᶜ, wᶜ, uᶜ) in zip(v, w, u)
            @. uᶜ = α * vᶜ + β * wᶜ
        end
    end
    return u
end

function VectorInterface.add!(w::StateVector, v::StateVector, α::Number, β::Number)
    if β === VectorInterface.One()
        for (vᶜ, wᶜ) in zip(v, w)
            @. wᶜ += α * vᶜ
        end
    else
        for (vᶜ, wᶜ) in zip(v, w)
            @. wᶜ = α * vᶜ + β * wᶜ
        end
    end
    return w
end

VectorInterface.add!!(w::StateVector, v::StateVector, α::Number, β::Number) = VectorInterface.add!(w, v, α, β)

"Calculate inner product that is correct for basis cis and sin."
VectorInterface.inner(v::StateVector, w::StateVector) = sum(LA.dot.(v, w)) # dot works fine on arbitrary-dimensional arrays (sum of element-wise products)

"Calculate 1D cos inner product: endpoints contribute with coefficients 1/2."
function VectorInterface.inner(v::StateVector{T, 1, :cos}, w::StateVector{T, 1, :cos}) where T
    sum(LA.dot.(v, w)) - sum(vᶜ[1]*wᶜ[1] + vᶜ[end]*wᶜ[end] for (vᶜ, wᶜ) in zip(v, w))/2
end

"Calculate 2D cos inner product: edges contribute with coefficients 1/2 and corners with coefficients 1/4."
@views function VectorInterface.inner(v::StateVector{T, 2, :cos}, w::StateVector{T, 2, :cos}) where T
    sum(LA.dot.(v, w)) -
    sum(LA.dot(vᶜ[1, 2:end-1]  , wᶜ[1, 2:end-1])   +
        LA.dot(vᶜ[end, 2:end-1], wᶜ[end, 2:end-1]) +
        LA.dot(vᶜ[2:end-1, 1]  , wᶜ[2:end-1, 1])   +
        LA.dot(vᶜ[2:end-1, end], wᶜ[2:end-1, end]) for (vᶜ, wᶜ) in zip(v, w)) / 2 -
    sum(vᶜ[1, 1]'*wᶜ[1, 1] + vᶜ[1, end]'*wᶜ[1, end] + vᶜ[end, 1]'*wᶜ[end, 1] + vᶜ[end, end]'*wᶜ[end, end] for (vᶜ, wᶜ) in zip(v, w)) * 3/4
end

"Calculate general-D cos inner product"
function VectorInterface.inner(v::StateVector{T, D, :cos}, w::StateVector{T, D, :cos}) where {T, D}
    println("Inner product for the cos basis is not implemented in $D dimensions.")
end

"Return the norm ∑ᶜ|𝜓ᶜ|² (sum over components); note that physical normalisation includes the volume element: ∑ᶜ|𝜓ᶜ|²d𝑉 = 1."
VectorInterface.norm(v::StateVector) = √VectorInterface.inner(v, v) # could use `LA.norm(v.data)`, but this is faster