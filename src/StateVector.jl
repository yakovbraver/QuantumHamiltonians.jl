import KrylovKit.VectorInterface

"""
The type encoding the physical x-space state of the system. Contains a single field `data` that is an n-component vector of rank-D tensors.
"""
struct StateVector{T, D}
    data::Vector{Array{T, D}}
end

# These methods enable iterating over a StateVector directly (as if iterating over the underlying `data` array)
Base.length(v::StateVector) = length(v.data)
Base.eltype(v::StateVector) = eltype(v.data)
Base.getindex(v::StateVector, i::Int) = v.data[i]
Base.iterate(v::StateVector, i=1) = i > length(v) ? nothing : (v[i], i+1)

function Base.similar(v::StateVector{T, D}, ::Type{S}) where {T, D, S}
    data = [similar(d, S) for d in v.data]
    return StateVector(data)
end

function Base.similar(v::StateVector{T}) where T
    return similar(v, T)
end

function Base.show(io::IO, v::StateVector{T, D}) where {T, D}
    # Examples:
    # "2-component StateVector{Float64, 1} on a 5-point grid"
    # "3-component StateVector{Float64, 3} on a 5×6×7 grid"
    print(io, length(v.data), "-component StateVector{$T, $D} on a ", size(v.data[1], 1),
          (D == 1 ? "-point" : prod("×$(size(v.data[1], i))" for i in 2:D)), " grid")
end

VectorInterface.scalartype(::Type{<:StateVector{T}}) where T = T

VectorInterface.zerovector(v::StateVector, ::Type{T}) where T <: Number = StateVector([zeros(T, size(d)) for d in v])

function VectorInterface.zerovector!(v::StateVector{T}) where T
    ZERO = zero(T)
    for d in v
        d .= ZERO
    end
    return v
end

VectorInterface.zerovector!!(v::StateVector) = VectorInterface.zerovector!(v)

VectorInterface.scale(v::StateVector, α::Number) = VectorInterface.scale!!(similar(v), v, α)

function VectorInterface.scale!(v::StateVector, α::Number)
    for d in v
        d .*= α
    end
    return v
end

VectorInterface.scale!!(v::StateVector, α::Number) = VectorInterface.scale!(v, α)

function VectorInterface.scale!!(w::StateVector, v::StateVector, α::Number)
    for (vd, wd) in zip(v, w)
        @. wd = α * vd
    end
    return w
end

function VectorInterface.add(w::StateVector, v::StateVector, α::Number, β::Number)
    u = similar(v)
    if β === VectorInterface.One()
        for (vd, wd, ud) in zip(v, w, u)
            @. ud = α * vd + wd
        end
    else
        for (vd, wd, ud) in zip(v, w, u)
            @. ud = α * vd + β * wd
        end
    end
    return u
end

function VectorInterface.add!(w::StateVector, v::StateVector, α::Number, β::Number)
    if β === VectorInterface.One()
        for (vd, wd) in zip(v, w)
            @. wd += α * vd
        end
    else
        for (vd, wd) in zip(v, w)
            @. wd = α * vd + β * wd
        end
    end
    return w
end

VectorInterface.add!!(w::StateVector, v::StateVector, α::Number, β::Number) = VectorInterface.add!(w, v, α, β)
VectorInterface.inner(v::StateVector, w::StateVector) = sum(LA.dot.(v, w)) # dot works fine on arbitrary-dimensional arrays (sum of element-wise products)

# We calculate the norm as ∑ᵢ|𝜓ᵢ|² (sum over components); note that physical normalisation includes the volume element: ∑ᵢ|𝜓ᵢ|²d𝑉 = 1.
VectorInterface.norm(v::StateVector) = LA.norm(v.data) # LA.norm works recursively on nested arrays; 2-norm gives the desired result
# VectorInterface.norm(v::StateVector) = √VectorInterface.inner(v, v) # TODO: benchmark to see if this is faster