const TupleUnion{T} = Union{T, Tuple{Vararg{T}}}

"""
AbstractMDP - Any MDP with state space S, action space A and reward outcomes R.
All three sets can be discrete or continuous.
"""
abstract type AbstractMDP{
    S_space <: TupleUnion{Space}, 
    A_space <: TupleUnion{Space}, 
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
function step(mdp::AbstractMDP{S_space, A_space, R_dist}, a) where 
        {S_space <: Space, A_space <: Space, R_dist <: Distribution}
    throw(NotImplementedError(
        "step not implemented for MDP type $(typeof(mdp)) and action type $(typeof(a))")
        )
end

"""
TabularMDP - AbstractMDP with restriction that state space S and action space A are
discrete and finite.
"""
const AbstractTabularMDP{
    S_space <: TupleUnion{DiscreteSpace}, 
    A_space <: TupleUnion{DiscreteSpace}, 
    R_dist} = AbstractMDP{
        S_space, 
        A_space, 
        R_dist}
"""
DiscreteMDP - TabularMDP with further restriction that reward space R is discrete.
"""
const AbstractDiscreteMDP{
    S_space <: TupleUnion{DiscreteSpace}, 
    A_space <: TupleUnion{DiscreteSpace}} = AbstractTabularMDP{S_space, A_space, DiscreteNonParametric}

mutable struct TabularMDP{R <: Distribution} <: AbstractTabularMDP{DiscreteContiguousSpace, DiscreteContiguousSpace, R}
    S::DiscreteContiguousSpace # Set of states - 1, 2, ..., length(S)
    A::DiscreteContiguousSpace # Set of actions - 1, 2, ..., length(A)
    P::Array{Float64,3} # P[s_1, s_0, a] : probability of transitioning to s_1 from s_0 when choosing a
    P_dist::Array{DiscreteNonParametric,2}
    R::Union{Array{R, 2}, Array{R,3}} # R[s_0, a], or R[s_1, s_0, a] : reward distribution (3-dimensional if conditioned on next state)
    μ::DiscreteNonParametric # μ[s] : initial state distribution
    γ::Float64 # Discount factor
    terminal::Vector{Bool} # terminal[s] is true if the state s is terminal. State is terminal if P[s,s,a] = 1 ∀a∈A.
    s::Int64 # Current state
    terminated::Bool # If episode has terminated.
    TabularMDP(numStates::Int64, numActions::Int64, P::Array{Float64,3}, 
            R::Union{Array{Float64, 2}, Array{Float64,3}}, μ::Vector{Float64}, γ::Float64) = begin
        S = DiscreteContiguousSpace(numStates)
        A = DiscreteContiguousSpace(numActions)
        
        @assert any(abs.(sum(P, dims = 1) .- 1.0) .< 1e-12) "Transition probs must equal 1 for all sum(P[:,s,a])"
        @assert abs(sum(μ) - 1.0) < 1e-12 "Initial state dist., μ, must equal 1"
        @assert all(μ .>= 0.0) "Initial state dist., μ, must contain only non-negative elements, given μ = $μ."
        @assert size(P, 1) == numStates && size(P, 2) == numStates "Dim 1 and 2 of transition probabilities must equal number of states, size(P) = $(size(P)) and numStates = $numStates"
        @assert size(P, 3) == numActions "Dim 3 of transition probabilities must equal number of states, size(P) = $(size(P)) and numStates = $numActions"
        @assert length(μ) == numStates "Length of μ must equal numStates, length(μ) = $(length(μ)) and numStates = $numStates."

        terminal = Vector{Bool}(undef, numStates)
        for s in 1:numStates terminal[s] = all(abs.(P[s,s,:] .- 1.0) .< 1e-12) end
        @assert sum(μ[terminal]) < 1e-12 "μ[s] for terminal states must be zero, given μ = $μ and terminal = $terminal."
        
        P_dist = Array{DiscreteNonParametric,2}(undef, numStates, numActions)
        for s in S, a in A
            P_dist[s,a] = DiscreteNonParametric(1:numStates, P[:, s, a])
        end
        
        initialDistribution = DiscreteNonParametric(1:numStates, μ)
        initialState = rand(initialDistribution)
        
        new{Dirac{Float64}}(S, A, P, P_dist, Dirac.(R), initialDistribution, γ, terminal, initialState, true)
    end

    TabularMDP(P::Array{Float64,3}, 
            R::Union{Array{Float64, 2}, Array{Float64,3}}, μ::Vector{Float64}, γ::Float64) = begin
        numberOfStates, numberOfActions = length(μ), size(P,3)
        TabularMDP(numberOfStates, numberOfActions, P, R, μ, γ)
    end

    TabularMDP(numStates::Int64, numActions::Int64, P::Array{Float64,3}, 
            R::Union{Array{T,2},Array{T,3}}, μ::Vector{Float64}, γ::Float64) where {T <: Distribution} = begin
        S = DiscreteContiguousSpace(numStates)
        A = DiscreteContiguousSpace(numActions)
        
        @assert any(abs.(sum(P, dims = 1) .- 1.0) .< 1e-12) "Transition probs must equal 1 for all sum(P[:,s,a])"
        @assert abs(sum(μ) - 1.0) < 1e-12 "Initial state dist., μ, must equal 1"
        @assert all(μ .>= 0.0) "Initial state dist., μ, must contain only non-negative elements, given μ = $μ."
        @assert size(P, 1) == numStates && size(P, 2) == numStates "Dim 1 and 2 of transition probabilities must equal number of states, size(P) = $(size(P)) and numStates = $numStates"
        @assert size(P, 3) == numActions "Dim 3 of transition probabilities must equal number of states, size(P) = $(size(P)) and numStates = $numActions"
        @assert length(μ) == numStates "Length of μ must equal numStates, length(μ) = $(length(μ)) and numStates = $numStates."

        terminal = Vector{Bool}(undef, numStates)
        for s in S terminal[s] = all(abs.(P[s,s,:] .- 1.0) .< 1e-12) end
        @assert sum(μ[terminal]) < 1e-12 "μ[s] for terminal states s must be zero, given μ = $μ and terminal = $terminal."
        
        P_dist = Array{DiscreteNonParametric,2}(undef, numStates, numActions)
        for s in S, a in A
            P_dist[s,a] = DiscreteNonParametric(1:numStates, P[:, s, a])
        end
        
        initialDistribution = DiscreteNonParametric(1:numStates, μ)
        initialState = rand(initialDistribution)
        
        new{T}(S, A, P, P_dist, R, initialDistribution, γ, terminal, initialState, true)
    end

    TabularMDP(P::Array{Float64,3}, 
            R::Union{Array{T,2},Array{T,3}}, μ::Vector{Float64}, γ::Float64) where T <: Distribution = begin
        numberOfStates, numberOfActions = length(μ), size(P,3)
        TabularMDP(numberOfStates, numberOfActions, P, R, μ, γ)
    end
end

function getStates(mdp::TabularMDP)::DiscreteContiguousSpace return mdp.S end
function getActions(mdp::TabularMDP)::DiscreteContiguousSpace return mdp.A end
function getTransitionProbabilities(mdp::TabularMDP)::Array{Float64,3} return mdp.P end
function getTransitionDistributions(mdp::TabularMDP)::Array{DiscreteNonParametric,2} return mdp.P_dist end
function getRewardDistributions(mdp::TabularMDP{R})::Union{Array{R, 2}, Array{R, 3}} where {R<: Distribution} return mdp.R end
function getInitialStateDistribution(mdp::TabularMDP)::DiscreteNonParametric return mdp.μ end
function getTerminalMask(mdp::TabularMDP)::Vector{Bool} return mdp.terminal end
function getDiscountFactor(mdp::TabularMDP)::Float64 return mdp.γ end
function getCurrentState(mdp::TabularMDP) return mdp.s end
function hasTerminated(mdp::TabularMDP) return mdp.terminated end

function setState(mdp::TabularMDP{R_dist}, s_new) where {R_dist <: Distribution} 
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

function step(mdp::TabularMDP{R_dist}, a::Int64) where {R_dist <: Distribution}
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

function reset(mdp::TabularMDP{R_dist}) where {R_dist <: Distribution}
    s_new = sampleInitialState(mdp)
    setState(mdp, s_new)
    setTerminated(mdp, false)
    return s_new
end

function reset!(mdp::TabularMDP{R_dist}) where {R_dist <: Distribution}
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
    numActions = length(getActions(mdp))
    π_d = [DiscreteNonParametric(1:numActions, π[s,:]) for s in getStates(mdp)]
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
