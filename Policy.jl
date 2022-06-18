using Distributions
#using NNlib: softmax
using LogExpFunctions: softmax

abstract type AbstractPolicy end
abstract type GLIEPolicy <: AbstractPolicy end

struct Policy <: AbstractPolicy
    S::Vector{Int64}
    A::Vector{Int64}
    π_vals::Array{Float64,2} # π_vals[s,a], sum(π[s,:]) == 1 ∀s ∈ S
    π::Vector{DiscreteNonParametric{Int64, Float64, Vector{Int64}, Vector{Float64}}}
    Policy(S::Vector{Int64}, A::Vector{Int64}, π_vals::Array{Float64,2}) = begin
        for s in S
            isDiscreteDistribution = abs(sum( π_vals[s,:] ) -1)  <= 1e-10
            @assert isDiscreteDistribution "Row $(s) of π_vals does not sum to 1 (deviation larger than 1e-12)"
        end
        new(S, A, π_vals, [DiscreteNonParametric(A, π_vals[s, :]) for s in S])
    end
end

function createUniformPolicy(mdp :: TabularMDP)
    S_size, A_size = length(mdp.S), length(mdp.A)
    π_vals = ones(S_size, A_size) ./ A_size
    return Policy(mdp.S, mdp.A, π_vals)
end

function sample(π::Policy, s::Int64)
    return rand(π.π[s])
end 

function getActionProbabilities(π::Policy, s::Int64)::Vector{Float64}
    return π.π_vals[s,:]
end

mutable struct EpsilonGreedyPolicy <: GLIEPolicy
    #Linearly annealing EpsilonGreedyPolicy
    S::Vector{Int64}
    A::Vector{Int64}
    #Q::Array{Float64,2}
    UniformAction::DiscreteNonParametric{Int64, Float64, Vector{Int64}, Vector{Float64}}
    ϵ_init::Float64
    ϵ_final::Float64
    ϵ_current_iteration::Int64
    ϵ_total_iterations::Int64
    EpsilonGreedyPolicy(mdp::TabularMDP, ϵ_init, ϵ_final, ϵ_total_iterations) = begin
        uniformAction = DiscreteNonParametric(mdp.A, [1/length(mdp.A) for _ in mdp.A])
        new(mdp.S, mdp.A, uniformAction, ϵ_init, ϵ_final, 0, ϵ_total_iterations)
    end
end

function reset!(π::EpsilonGreedyPolicy)
    π.ϵ_current_iteration = 0
end

function getEpsilon(π::EpsilonGreedyPolicy)
    ϵ = π.ϵ_init - (π.ϵ_init - π.ϵ_final)* minimum([1, π.ϵ_current_iteration/π.ϵ_total_iterations])
    return ϵ
end

function sample(π::EpsilonGreedyPolicy, s::Int64, Q::Array{Float64,2})
    p = rand()
    ϵ = getEpsilon(π)
    π.ϵ_current_iteration += 1
    if p < ϵ
        return rand(π.UniformAction)
    else
        return argmax(Q[s,:])
    end
end

function getActionProbabilities(π::EpsilonGreedyPolicy, s::Int64, Q::Array{Float64, 2})::Vector{Float64}
    numActions = size(Q, 2)
    ϵ = getEpsilon(π)
    p = fill(ϵ / numActions, numActions)
    p[argmax(Q[s,:])] += (1 - ϵ)
    return p
end

mutable struct BoltzmannPolicy <: GLIEPolicy
    S::Vector{Int64}
    A::Vector{Int64}
    #Q::Array{Float64,2}
    β_init::Float64
    β_final::Float64
    β_current_iteration::Int64
    β_total_iterations::Int64
    BoltzmannPolicy(mdp::TabularMDP, β_init, β_final, β_total_iterations) = begin
        new(mdp.S, mdp.A, β_init, β_final, 0, β_total_iterations)
    end
end

function reset!(π::BoltzmannPolicy)
    π.β_current_iteration = 0
end

function getBeta(π::BoltzmannPolicy)
    β = π.β_init - (π.β_init - π.β_final) * minimum([1, π.β_current_iteration/π.β_total_iterations])
    return β
end

function sample(π::BoltzmannPolicy, s::Int64, Q::Array{Float64,2})
    β = getBeta(π)
    π.β_current_iteration += 1
    d = DiscreteNonParametric(π.A, softmax(β * Q[s,:]))
    return rand(d)
end

function getActionProbabilities(π::BoltzmannPolicy, s::Int64, Q::Array{Float64, 2})::Vector{Float64}
    β = getBeta(π)
    return softmax(β * Q[s,:])
end

function getActionProbabilities(π::Union{EpsilonGreedyPolicy, BoltzmannPolicy}, s::Int64, a::Int64, Q::Array{Float64, 2})
    return getActionProbabilities(π, s, Q)[a]
end

function sampleEpisode(mdp::TabularMDP, π::Policy, T::Number)
    return sampleEpisode(mdp, π.π_vals, T)
end