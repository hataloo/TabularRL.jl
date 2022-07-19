
abstract type Container{T} end

struct DiscreteContainer{T <: Integer} <: Container{T} end

abstract type MetaContainer{A, B, ContA <: Container{A}, ContB <: Container{B}} end

struct DiscreteMetaContainer{A, B} <: MetaContainer{A, B, DiscreteContainer{A}, DiscreteContainer{B}} end

dmc = DiscreteMetaContainer{Int32, Int64}()

using Distributions
abstract type Space{T} end
struct DiscreteSpace{T <:  Integer} <: Space{T} end
abstract type AbstractMDP{
    S_type, S_space <: Space, 
    A_type, A_space <: Space, 
    R_dist <: Distribution}
end
const AbstractTabularMDP{S_type, A_type, R_dist} = AbstractMDP{
    S_type, DiscreteSpace{S_type}, 
    A_type, DiscreteSpace{A_type}, 
    R_dist}
const AbstractDiscreteMDP{S_type, A_type} = AbstractTabularMDP{S_type, A_type, DiscreteDistribution}

struct CustomMDP <: AbstractDiscreteMDP{Int64, Int64}
    S::Vector{Int64}
    A::Vector{Int64}
    s::Int64
end

abstract type AbstractMDPWrapper{
    S_type, S_space <: Space, 
    A_type, A_space <: Space, 
    R_dist <: Distribution,
    MDP_type <: AbstractMDP{
        S_type, S_space, 
        A_type, S_space, 
        R_dist}
    } <: AbstractMDP{
        S_type, S_space, 
        A_type, A_space, 
        R_dist} 
end

struct MDPWrapper{
    S_type, S_space <: Space, 
    A_type, A_space <: Space, 
    R_dist <: Distribution,
    MDP_type <: AbstractMDP{
        S_type, S_space, 
        A_type, A_space, 
        R_dist}
    } <: AbstractMDPWrapper{
            S_type, S_space, 
            A_type, A_space, 
            R_dist, MDP_type}
    MDP::MDP_type
    MDPWrapper(mdp::AbstractMDP{
        S_type, S_space, 
        A_type, A_space, 
        R_dist}) where {
            S_type, S_space <: Space,
            A_type, A_space <: Space,
            R_dist} = 
        begin
            new{S_type, S_space, A_type, A_space, R_dist,
                AbstractMDP{
                    S_type, S_space, 
                    A_type, A_space, 
                    R_dist}}(mdp)
        end 
end

m = CustomMDP([1,2], [1,2,3], 1)
w = MDPWrapper(m)
