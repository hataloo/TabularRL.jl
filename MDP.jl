using StatsBase, Distributions

struct MDP
    S::Vector{Int64} # Set of states
    A::Vector{Int64} # Set of actions
    P::Array{Float64,3} # P[s_1, s_0, a] : probability of transitioning to s_1 from s_0 when choosing a
    R::Array{Float64,2} # R[s, a] : expected reward
    μ::Vector{Float64} # u[s] : initial state distribution
    γ::Float64 # Discount factor
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
    return s_new, r
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
    e = Episode([Vector{Int64}(undef, T) for _ in 1:3]...)
    e.s[1] = sampleInitialState(mdp)
    π_d = [DiscreteNonParametric(mdp.A, π[s,:]) for s in mdp.S]
    for t = 1:(T)
        s = e.s[t]
        e.a[t] = a = rand(π_d[s])
        s, r = step(mdp, s, a)
        e.r[t] = r
        if t < T
            e.s[t+1] = s
        end
    end
    return e
end