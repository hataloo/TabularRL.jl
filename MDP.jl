using StatsBase, Distributions
import Base.length

struct MDP
    S::Vector{Int64} # Set of states
    A::Vector{Int64} # Set of actions
    P::Array{Float64,3} # P[s_1, s_0, a] : probability of transitioning to s_1 from s_0 when choosing a
    P_dist::Array{DiscreteNonParametric,2}
    R::Array{Distribution,2} # R[s_0, a] : reward distribution
    μ::Vector{Float64} # μ[s] : initial state distribution
    γ::Float64 # Discount factor
    terminal::Vector{Bool} # terminal[s] is true if the state s is terminal. State is terminal if P[s,s,a] = 1 ∀a∈A.
    
    MDP(S::Vector{Int64}, A::Vector{Int64}, P::Array{Float64,3}, R::Array{Float64,2}, μ::Vector{Float64}, γ::Float64) = begin
        N = length(S)
        terminal = Vector{Bool}(undef, N)
        @assert any(abs.(sum(P, dims = 1) .- 1.0) .< 1e-12) "Transition probs must equal 1 for all sum(P[:,s,a])"
        @assert abs(sum(μ) - 1.0) < 1e-12 "Initial state dist., μ, must equal 1"
        for s in 1:N terminal[s] = all(abs.(P[s,s,:] .- 1.0) .< 1e-12) end
        P_dist = Array{DiscreteNonParametric,2}(undef, length(S), length(A))
        for s in S, a in A
            P_dist[s,a] = DiscreteNonParametric(S, P[:, s, a])
        end
        new(S, A, P, P_dist, Dirac.(R), μ, γ, terminal)
    end

    MDP(S::Vector{Int64}, A::Vector{Int64}, P::Array{Float64,3}, R::Array{Distribution,2}, μ::Vector{Float64}, γ::Float64) = begin
        N = length(S)
        terminal = Vector{Bool}(undef, N)
        @assert any(abs.(sum(P, dims = 1) .- 1.0) .< 1e-12) "Transition probs must equal 1 for all sum(P[:,s,a])"
        @assert abs(sum(μ) - 1.0) < 1e-12 "Initial state dist., μ, must equal 1"
        for s in 1:N terminal[s] = all(abs.(P[s,s,:] .- 1.0) .< 1e-12) end
        P_dist = Array{DiscreteNonParametric,2}(undef, length(S), length(A))
        for s in S, a in A
            P_dist[s,a] = DiscreteNonParametric(S, P[:, s, a])
        end
        new(S, A, P, P_dist, R, μ, γ, terminal)
    end
end

function sampleInitialState(mdp::MDP)
    d = DiscreteNonParametric(mdp.S, mdp.μ)
    return rand(d)
end

function sampleStateFromDist(mdp::MDP, s::Int64, a::Int64)
    return rand(mdp.P_dist[s,a])
end

function step(mdp::MDP, s::Int64, a::Int64)
    s_new = sampleStateFromDist(mdp, s, a)
    r = rand(mdp.R[s,a])
    done = mdp.terminal[s_new]
    return s_new, r, done
end

struct Episode
    s::Vector{Int64}
    a::Vector{Int64}
    r::Vector{Int64}
end

function length(episode::Episode)
    return length(episode.s)
end

function sampleEpisode(mdp::MDP, π::Array{Float64,2},T::Number)
    π_d = [DiscreteNonParametric(mdp.A, π[s,:]) for s in mdp.S]
    states, actions, rewards = Vector{Int64}(undef,0), Vector{Int64}(undef,0), Vector{Float64}(undef,0)
    append!(states, sampleInitialState(mdp))
    for t = 1:T
        s = states[t]
        a = rand(π_d[s])
        append!(actions, a)
        s, r, done = step(mdp, s, a)
        append!(rewards, r)
        if done break end
        if t < T
            append!(states, s)
        end
    end
    return Episode(states, actions, rewards)
end