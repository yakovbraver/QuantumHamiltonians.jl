abstract type AbsType{Storage,R} end

function AbsType{:dense}(x::R) where R
    DenseHam(x)
end

struct DenseHam{R} <: AbsType{:dense,R}
    x::R
end

dh = DenseHam(4)

dh2 = AbsType{:dense}(6)

 function get_type(ah::AbsType{Storage,R}) where {Storage,R}
    println(R)
 end

 get_type(dh2)

############################

abstract type AbsType2{Storage,R<:AbstractFloat,T<:Union{R,Complex{R}}} end

function AbsType2{:dense}(x::R, y::T) where {R, T}
    DenseHam2(x, y)
end

struct DenseHam2{R,T} <: AbsType2{:dense,R,T}
    x::R
    y::T
end

dh = DenseHam2("heh")

dh2 = AbsType2{:dense}(6.0, 4.0f0+3im)

 function get_type(ah::AbsType2{Storage,R}) where {Storage,R}
    println(R)
 end

 get_type(dh2)