abstract type Space{T} end
abstract type DiscreteSpace{T <: Integer} <: Space{T} end
abstract type ContinuousSpace{T <: Real} <: Space{T} end

"""
AbstractMDP - Any MDP with state space S, action space A and reward outcomes R.
All three sets can be discrete or continuous.
"""
abstract type AbstractMDP{
    S_type, S_space <: Space, 
    A_type, A_space <: Space, 
    R_dist <: Distribution}
end

function hasTerminated(mdp::AbstractMDP)
    throw(NotImplementedError("hasTerminated not implemented for type $(typeof(mdp))"))
end
function reset(mdp::AbstractMDP)
    throw(NotImplementedError("reset not implemented for type $(typeof(mdp))"))
end
function reset!(mdp::AbstractMDP)
    throw(NotImplementedError("reset! not implemented for type $(typeof(mdp))"))
end
function step(mdp::AbstractMDP{S_type, S_space, A_type, A_space, R_dist}, a::A_type) where 
        {S_type, S_space <: Space, A_type, A_space <: Space, R_dist <: Distribution}
    throw(NotImplementedError(
        "step not implemented for MDP type $(typeof(mdp)) and action type $(typeof(a))")
        )
end

"""
TabularMDP - AbstractMDP with restriction that state space S and action space A are
discrete and finite.
"""
const AbstractTabularMDP{S_type, A_type, R_dist} = AbstractMDP{
    S_type, DiscreteSpace{S_type}, 
    A_type, DiscreteSpace{A_type}, 
    R_dist}
"""
DiscreteMDP - TabularMDP with further restriction that reward space R is discrete.
"""
const AbstractDiscreteMDP{S_type, A_type} = AbstractTabularMDP{S_type, A_type, DiscreteNonParametric}

mutable struct TabularMDP{S_type, A_type, R <: Distribution} <: AbstractTabularMDP{S_type, A_type, R}
    S::Vector{S_type} # Set of states - 1, 2, ..., length(S)
    A::Vector{A_type} # Set of actions - 1, 2, ..., length(A)
    P::Array{Float64,3} # P[s_1, s_0, a] : probability of transitioning to s_1 from s_0 when choosing a
    P_dist::Array{DiscreteNonParametric,2}
    R::Union{Array{R, 2}, Array{R,3}} # R[s_0, a], or R[s_1, s_0, a] : reward distribution (3-dimensional if conditioned on next state)
    μ::DiscreteNonParametric # μ[s] : initial state distribution
    γ::Float64 # Discount factor
    terminal::Vector{Bool} # terminal[s] is true if the state s is terminal. State is terminal if P[s,s,a] = 1 ∀a∈A.
    s::S_type # Current state
    terminated::Bool # If episode has terminated.
    TabularMDP(S::Vector{S_type}, A::Vector{A_type}, P::Array{Float64,3}, 
            R::Union{Array{Float64, 2}, Array{Float64,3}}, μ::Vector{Float64}, γ::Float64) where {S_type, A_type} = begin
                
        @assert any(abs.(sum(P, dims = 1) .- 1.0) .< 1e-12) "Transition probs must equal 1 for all sum(P[:,s,a])"
        @assert abs(sum(μ) - 1.0) < 1e-12 "Initial state dist., μ, must equal 1"
        @assert all(μ .>= 0.0) "Initial state dist., μ, must contain only non-negative elements, given μ = $μ."
        
        N = length(S)
        terminal = Vector{Bool}(undef, N)
        for s in 1:N terminal[s] = all(abs.(P[s,s,:] .- 1.0) .< 1e-12) end
        @assert sum(μ[terminal]) < 1e-12 "μ[s] for terminal states must be zero, given μ = $μ and terminal = $terminal."
        
        P_dist = Array{DiscreteNonParametric,2}(undef, length(S), length(A))
        for s in S, a in A
            P_dist[s,a] = DiscreteNonParametric(S, P[:, s, a])
        end
        
        initialDistribution = DiscreteNonParametric(S, μ)
        initialState = rand(initialDistribution)
        
        new{S_type, A_type, Dirac{Float64}}(S, A, P, P_dist, Dirac.(R), initialDistribution, γ, terminal, initialState, true)
    end

    TabularMDP(P::Array{Float64,3}, 
            R::Union{Array{Float64, 2}, Array{Float64,3}}, μ::Vector{Float64}, γ::Float64) = begin
        numberOfStates, numberOfActions = length(μ), size(P,3)
        TabularMDP(Vector{Int64}(1:numberOfStates), Vector{Int64}(1:numberOfActions), P, R, μ, γ)
    end

    TabularMDP(S::Vector{S_type}, A::Vector{A_type}, P::Array{Float64,3}, 
            R::Union{Array{T,2},Array{T,3}}, μ::Vector{Float64}, γ::Float64) where {S_type, A_type, T <: Distribution} = begin
               
        @assert any(abs.(sum(P, dims = 1) .- 1.0) .< 1e-12) "Transition probs must equal 1 for all sum(P[:,s,a])"
        @assert abs(sum(μ) - 1.0) < 1e-12 "Initial state dist., μ, must equal 1"
        @assert all(μ .>= 0.0) "Initial state dist., μ, must contain only non-negative elements, given μ = $μ."
        
        N = length(S)
        terminal = Vector{Bool}(undef, N)
        for s in 1:N terminal[s] = all(abs.(P[s,s,:] .- 1.0) .< 1e-12) end
        @assert sum(μ[terminal]) < 1e-12 "μ[s] for terminal states s must be zero, given μ = $μ and terminal = $terminal."
        
        P_dist = Array{DiscreteNonParametric,2}(undef, length(S), length(A))
        for s in S, a in A
            P_dist[s,a] = DiscreteNonParametric(S, P[:, s, a])
        end
        
        initialDistribution = DiscreteNonParametric(S, μ)
        initialState = rand(initialDistribution)
        
        new{S_type, A_type, T}(S, A, P, P_dist, R, initialDistribution, γ, terminal, initialState, true)
    end

    TabularMDP(P::Array{Float64,3}, 
            R::Union{Array{T,2},Array{T,3}}, μ::Vector{Float64}, γ::Float64) where T <: Distribution = begin
        numberOfStates, numberOfActions = length(μ), size(P,3)
        TabularMDP(Vector{Int64}(1:numberOfStates), Vector{Int64}(1:numberOfActions), 
            P, R, μ, γ)
    end
end

function getStates(mdp::TabularMDP)::Vector{Int64} return mdp.S end
function getActions(mdp::TabularMDP)::Vector{Int64} return mdp.A end
function getTransitionProbabilities(mdp::TabularMDP)::Array{Float64,3} return mdp.P end
function getTransitionDistributions(mdp::TabularMDP)::Array{DiscreteNonParametric,2} return mdp.P_dist end
function getRewardDistributions(mdp::TabularMDP{S_type, A_type, R})::Union{Array{R, 2}, Array{R, 3}} where {S_type, A_type, R<: Distribution} return mdp.R end
function getInitialStateDistribution(mdp::TabularMDP)::DiscreteNonParametric return mdp.μ end
function getTerminalMask(mdp::TabularMDP)::Vector{Bool} return mdp.terminal end
function getDiscountFactor(mdp::TabularMDP)::Float64 return mdp.γ end
function getCurrentState(mdp::TabularMDP) return mdp.s end
function hasTerminated(mdp::TabularMDP) return mdp.terminated end

function setState(mdp::TabularMDP{S_type, A_type, R_dist}, s_new) where {S_type, A_type, R_dist <: Distribution} 
    mdp.s = s_new
end

function setTerminated(mdp::TabularMDP, terminated_new::Bool) mdp.terminated = terminated_new end

function sampleInitialState(mdp::TabularMDP)
    #d = DiscreteNonParametric(mdp.S, mdp.μ)
    μ = getInitialStateDistribution(mdp)
    return rand(μ)
end

function isTerminalState(mdp::TabularMDP, s::Int64)
    return getTerminalMask(mdp)[s]
end

"""
E[r | s, a] for all s ∈ S, a ∈ A.
"""
function getMeanRewards(mdp::TabularMDP)
    if ndims(getRewardDistributions(mdp)) == 2
        return mean.(getRewardDistributions(mdp))
    else
        # Dims: [S', S, A] * [S', S, A]
        # E[r | s, a] = \sum_{s' \in S} E[R_t | s', s, a] * P(s' | s, a)
        P = getTransitionProbabilities(mdp)
        R = getRewardDistributions(mdp)
        expectedRewards = sum(mean.(R) .* P, dims = 1)
        return dropdims(expectedRewards, dims = 1)
    end
end

function sampleState(mdp::TabularMDP, s::Int64, a::Int64)
    P_dist = getTransitionDistributions(mdp)
    s_new = rand(P_dist[s,a])

    return s_new
end

function sampleReward(mdp::TabularMDP, s::Int64, a::Int64)
    R = getRewardDistributions(mdp)
    return rand(R[s,a])
end

function sampleReward(mdp::TabularMDP, s_1::Int64, s_0::Int64, a::Int64)
    R = getRewardDistributions(mdp)
    return rand(R[s_1,s_0,a])
end

function step(mdp::TabularMDP{S_type, A_type, R_dist}, a::A_type) where {S_type, A_type, R_dist <: Distribution}
    if hasTerminated(mdp) throw(InvalidStateException("MDP has terminated, must call reset or reset! before calling step.", :hasTerminated)) end
    s = getCurrentState(mdp)
    if isTerminalState(mdp, s) throw(InvalidStateException("MDP state $s is terminal, cannot call step with terminal state.", :terminated)) end
    s_new = sampleState(mdp, s, a)
    R = getRewardDistributions(mdp)
    if ndims(R) == 2
        r = sampleReward(mdp, s, a)
    else
        r = sampleReward(mdp, s_new, s, a)    
    end
    terminated = isTerminalState(mdp, s_new)
    setState(mdp, s_new)
    setTerminated(mdp, terminated)
    return s_new, r, terminated
end

function reset(mdp::TabularMDP{S_type, A_type, R_dist}) where {S_type, A_type, R_dist <: Distribution}
    s_new = sampleInitialState(mdp)
    setState(mdp, s_new)
    setTerminated(mdp, false)
    return s_new
end

function reset!(mdp::TabularMDP{S_type, A_type, R_dist}) where {S_type, A_type, R_dist <: Distribution}
    s_new = sampleInitialState(mdp)
    setState(mdp, s_new)
    setTerminated(mdp, false)
end

struct Episode
    states::Vector{Int64}
    actions::Vector{Int64}
    rewards::Vector{Float64}
    Episode(states::Vector{Int64}, actions::Vector{Int64}, rewards::Vector{Float64}) = begin
        S, A, R = length.([states, actions, rewards])
        @assert S == A & S == R "Length of states, actions and rewards must be equal, given lengths $S, $A, $R."
        new(states, actions, rewards)
    end
end

function length(episode::Episode)
    return length(episode.states)
end

function sampleEpisode(mdp::TabularMDP, π::Array{Float64,2}, T::Number)
    π_d = [DiscreteNonParametric(mdp.A, π[s,:]) for s in mdp.S]
    states, actions, rewards = Vector{Int64}(undef,0), Vector{Int64}(undef,0), Vector{Float64}(undef,0)
    s_init = reset(mdp)
    append!(states, s_init)
    for t = 1:T
        s = states[t]
        a = rand(π_d[s])
        append!(actions, a)
        s, r, terminated = step(mdp, a)
        append!(rewards, r)
        if terminated break end
        if t < T
            append!(states, s)
        end
    end
    return Episode(states, actions, rewards)
end
