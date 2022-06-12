using Distributions
abstract type Space end
struct DiscreteSpace <: Space end
struct ContinuousSpace <: Space end

"""
AbstractMDP - Any MDP with state space S, action space A and reward outcomes R.
All three sets can be discrete or continuous.
"""
abstract type AbstractMDP{S_space<:Space, A_space<:Space,R_dist<:Distribution} end
"""
TabularMDP - AbstractMDP with restriction that state space S and action space A are
discrete and finite.
"""
const AbstractTabularMDP{R_dist<:Distribution} = AbstractMDP{DiscreteSpace, DiscreteSpace, R_dist}
"""
DiscreteMDP - TabularMDP with further restriction that reward space R is discrete.
"""
const AbstractDiscreteMDP = AbstractTabularMDP{DiscreteDistribution}

struct TabularMDP{R<:Distribution} <: AbstractTabularMDP{R}
    S::Vector{Int64} # Set of states - 1, 2, ..., length(S)
    A::Vector{Int64} # Set of actions - 1, 2, ..., length(A)
    P::Array{Float64,3} # P[s_1, s_0, a] : probability of transitioning to s_1 from s_0 when choosing a
    P_dist::Array{Categorical,2}
    R::Union{Array{R, 2}, Array{R,3}} # R[s_0, a], or R[s_1, s_0, a] : reward distribution (3-dimensional if conditioned on next state)
    μ::Categorical # μ[s] : initial state distribution
    γ::Float64 # Discount factor
    terminal::Vector{Bool} # terminal[s] is true if the state s is terminal. State is terminal if P[s,s,a] = 1 ∀a∈A.
end
