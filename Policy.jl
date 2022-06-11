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

function createUniformPolicy(mdp :: MDP)
    S_size, A_size = length(mdp.S), length(mdp.A)
    π_vals = ones(S_size, A_size) ./ A_size
    return Policy(mdp.S, mdp.A, π_vals)
end

function sample(π::Policy, s::Int64)
    return rand(π.π[s])
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
    EpsilonGreedyPolicy(mdp::MDP, ϵ_init, ϵ_final, ϵ_total_iterations) = begin
        uniformAction = DiscreteNonParametric(mdp.A, [1/length(mdp.A) for _ in mdp.A])
        new(mdp.S, mdp.A, uniformAction, ϵ_init, ϵ_final, 0, ϵ_total_iterations)
    end
end

function reset!(π::EpsilonGreedyPolicy)
    π.ϵ_current_iteration = 0
end

function sample(π::EpsilonGreedyPolicy, s::Int64, Q::Array{Float64,2})
    p = rand()
    ϵ = π.ϵ_init - (π.ϵ_init - π.ϵ_final)* minimum([1, π.ϵ_current_iteration/π.ϵ_total_iterations])
    π.ϵ_current_iteration += 1
    if p < ϵ
        return rand(π.UniformAction)
    else
        return argmax(Q[s,:])
    end
end

mutable struct BoltzmannPolicy <: GLIEPolicy
    S::Vector{Int64}
    A::Vector{Int64}
    #Q::Array{Float64,2}
    β_init::Float64
    β_final::Float64
    β_current_iterations::Int64
    β_total_iterations::Int64
    BoltzmannPolicy(mdp::MDP, β_init, β_final, β_total_iterations) = begin
        new(mdp.S, mdp.A, uniformAction, β_init, β_final, 0, β_total_iterations)
    end
end

function reset!(π::BoltzmannPolicy)
    π.β_current_iteration = 0
end

function sample(π::BoltzmannPolicy, s::Int64, Q::Array{Float64,2})
    p = rand()
    β = π.β - (π.β_init - π.β_final) * minimum([1, π.β_current_iteration/π.β_total_iterations])
    π.β_current_iteration += 1
    d = DiscreteNonParametric(π.A, softmax(β * Q[s,:]))
    return rand(d)
end