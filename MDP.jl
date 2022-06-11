using StatsBase, Distributions

struct MDP
    S::Vector{Int64} # Set of states
    A::Vector{Int64} # Set of actions
    P::Array{Float64,3} # P[s_1, s_0, a] : probability of transitioning to s_1 from s_0 when choosing a
    R::Array{Float64,2} # R[s, a] : expected reward
    μ::Vector{Float64} # μ[s] : initial state distribution
    γ::Float64 # Discount factor
    terminal::Vector{Bool} # terminal[s] is true if the state s is terminal. State is terminal if P[s,s,a] = 1 ∀a∈A.
    MDP(S::Vector{Int64}, A::Vector{Int64}, P::Array{Float64,3}, R::Array{Float64,2}, μ::Vector{Float64}, γ::Float64) = begin
        N = length(S)
        terminal = Vector{Bool}(undef, N)
        @assert any(sum(P, dims = 1) .== 1.0), "Transition probs must equal 1 for all sum(P[:,s,a])"
        @assert sum(μ) .== 1.0, "Initial state dist., μ, must equal 1"
        for s in 1:N terminal[s] == all(P[s,s,:] .== 1.0) end
        new(S, A, P, R, μ, γ, terminal)
    end
end

function sampleInitialState(mdp::MDP)
    d = DiscreteNonParametric(mdp.S, mdp.μ)
    return rand(d)
end

function sampleNextState(mdp::MDP, s::Int64, a::Int64)
    p = rand()
    cdp = 0
    for s_new in mdp.S
        cdp += mdp.P[s_new, s, a]
        if p < cdp
            return s_new
        end
    end
end
function sampleStateFromDist(mdp::MDP, s::Int64, a::Int64)
    d = DiscreteNonParametric(mdp.S, mdp.P[:,s,a])
    return rand(d)
end

function step(mdp::MDP, s::Int64, a::Int64)
    r = mdp.R[s,a]
    s_new = sampleNextState(mdp, s, a)
    done = mdp.terminal[s_new]
    return s_new, r, done
    #s_new = StatsBase.sample(S, P[:,s,a], 1)
    #return s_new
end

struct Episode
    s::Vector{Int64}
    a::Vector{Int64}
    r::Vector{Int64}
end

function sampleEpisode(mdp::MDP, π::Array{Float64,2},T)
    # π[s,a]
    #e = Episode([Vector{Int64}(undef, T) for _ in 1:3]...)
    #e.s[1] = sampleInitialState(mdp)
    π_d = [DiscreteNonParametric(mdp.A, π[s,:]) for s in mdp.S]
    states, actions, rewards = Vector{Int64}(undef), Vector{Int64}(undef), Vector{Float64}(undef)
    states.append(sampleInitialState(mdp))
    for t = 1:T
        s = states[t]
        actions.append(rand(π_d[s]))
        a = actions[t]
        s, r, done = step(mdp, s, a)
        actions.append(r); rewards.append(r)
        if done: break
        if t < T
            states.append(s)
        end
    end
    return Episode(states, actions, rewards)
end