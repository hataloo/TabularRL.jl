abstract type VariateForm end

abstract type ArrayLikeVariate{N} <: VariateForm end

const Univariate = ArrayLikeVariate{0}
const MultiVariate = ArrayLikeVariate{1}
const Matrixvariate = ArrayLikeVariate{2}

begin

abstract type Space{T <: Real} end
function getSpaceType(::Space{T}) where {T} return T end
abstract type DiscreteSpace{T <: Integer} <: Space{T} end
abstract type ContinuousSpace{T} <: Space{T} end
struct ContiguousDiscreteSpace <: DiscreteSpace{Int64}
	min::Int64
	max::Int64
end
struct ContiguousContinuousSpace <: ContinuousSpace{Float64}
	min::Float64
	max::Float64
end	
using Distributions
const TupleUnion{T} = Union{T, Tuple{Vararg{T}}}
abstract type AbstractMDP{
	S_space <: TupleUnion{Space},
	A_space <: TupleUnion{Space},
	R_dist <: Distribution
} end
struct MDP{
	S_space, 
	A_space,
	R_dist
} <: AbstractMDP{S_space, A_space, R_dist} end

end

D_type = TupleUnion{Space}
Space <: D_type
DiscreteSpace <: D_type
Tuple{Space, Space} <: D_type
Tuple{DiscreteSpace{Int16}, DiscreteSpace{Int32}} <: D_type
Tuple{ContinuousSpace{Float64}, DiscreteSpace{Int64}} <: D_type
Tuple{DiscreteSpace{Int16}, ContinuousSpace{Float32}} <: D_type
x = Tuple{ContinuousSpace, DiscreteSpace}((ContiguousContinuousSpace(1,5), ContiguousDiscreteSpace(1,3)))

const AbstractTabularMDP{
		S_space <: TupleUnion{DiscreteSpace}, 
		A_space <: TupleUnion{DiscreteSpace}, 
		R_dist
	} = AbstractMDP{S_space, A_space, R_dist}
const AbstractDiscreteMDP{
		S_space <: TupleUnion{DiscreteSpace}, A_space <: TupleUnion{DiscreteSpace}
	} = AbstractTabularMDP{S_space, A_space, DiscreteNonParametric}


x = MDP{DiscreteSpace{Int64}, DiscreteSpace{Int64}, Distribution}()
x = MDP{Tuple{DiscreteSpace{Int64}, DiscreteSpace{Int64}}, 
		DiscreteSpace{Int64}, Distribution}()
x = MDP{Tuple{DiscreteSpace{Int64}, ContinuousSpace{Float64}}, 
		DiscreteSpace{Int64}, Distribution}()



struct MyCont{N, T} 
	MyCont(T::K) where {N, K <: NTuple{N, Int64}} = new{N, T}()
end
x = MyCont((1,2,3))


struct MyCont2{N, T <: NTuple{N, Int64}}
	x::T
end
x = MyCont2{3,NTuple{3, Int64}}((1,2,3))
y = MyCont2{0, Int64}(1)
struct ABC{T <: Union{Int64, NTuple{N, Int64}} where N} end
x = ABC{Int64}()
x = ABC{NTuple{3, Int64}}()
