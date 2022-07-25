abstract type Space{T} end
function getSpaceType(::Space{T}) where {T} return T end
abstract type DiscreteSpace{T <: Integer} <: Space{T} end
abstract type ContinuousSpace{T <: Real} <: Space{T} end
struct DiscreteContiguousSpace <: DiscreteSpace{Int64}
	min::Int64
	max::Int64
    DiscreteContiguousSpace(max::Int64) = DiscreteContiguousSpace(1, max)
    DiscreteContiguousSpace(min::Int64, max::Int64) = begin
        @assert min <= max "Condition min <= max does not hold, given max = $max, min = $min."
        new(min, max)
    end
end
Base.maximum(space::DiscreteContiguousSpace) = space.max
Base.minimum(space::DiscreteContiguousSpace) = space.min
Base.iterate(space::DiscreteContiguousSpace) = (space.min, space.min)
Base.iterate(space::DiscreteContiguousSpace, state::Int64) = begin 
    if (state < space.max)
        return (state + 1, state + 1)
    else 
        return nothing
    end
end
Base.length(space::DiscreteContiguousSpace) = 1 + space.max - space.min



struct ContinuousContiguousSpace <: ContinuousSpace{Float64}
	min::Float64
	max::Float64
    ContinuousContiguousSpace(max::Int64) = ContinuousContiguousSpace(1, max)
    ContinuousContiguousSpace(min::Int64, max::Int64) = begin
        @assert min <= max "Condition min <= max does not hold, given max = $max, min = $min."
        new(min, max)
    end
end	
Base.maximum(space::ContinuousContiguousSpace) = space.max
Base.minimum(space::ContinuousContiguousSpace) = space.min
Base.length(space::ContinuousContiguousSpace) = 1 + space.max - space.min
