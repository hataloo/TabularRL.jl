function ValueIteration(iterations::Int64, P::Array{Float64, 3}, R::Array{Float64,2}, γ::Float64)
    A = 1:size(P,3)
    S = 1:size(P,1)
    V_0 = zeros(size(P,1))
    V_1 = zeros(size(P,1))
    for t in 1:iterations
        for s in S
            V_1[s] = maximum([R[s,a] + γ .* sum([P[s_1,s,a] .* V_0[s_1] for s_1 in S]) for a in A])
        end
        V_0 = copy(V_1)
    end
    return V_1
end

function ValueIteration(mdp::TabularMDP, iterations::Int64)
    R = getRewardDistributions(mdp)
    γ = getDiscountFactor(mdp)
    return ValueIteration(iterations, getTransitionProbabilities(mdp), mean.(R), γ)
end

function PolicyIteration(iterations::Int64, P::Array{Float64, 3}, R::Array{Float64,2}, γ::Float64)
    A = 1:size(P,3)
    S = 1:size(P,1)
    π_0 = Array{Float64,2}(undef, length(S), length(A))
    # π[s,a]
    π_0 .= 1/A
    π_1 = copy(π_0)
    # [s,a] * [s_1,s,a]
    Vᵖ = zeros(length(S))
    Rᵖ = zeros(length(S))
    Pᵖ = zeros(length(S), length(S))

    for t in 1:iterations
        # Step 1: Policy evaluation
        Pᵖ = [sum([P[s_1, s_0, a] .* π_0[s_0, a] for a in A]) for s_0 in S, s_1 in S]
        Rᵖ = [sum([R[s,a] .* π_0[s,a] for a in A]) for s in S]
        Vᵖ = inv(I - γ .* Pᵖ) * Rᵖ
        
        # Step 2: Policy improvement
        for s in S
            a = argmax([R[s,a] + γ .* sum([P[s_1,s,a] * Vᵖ[s_1] for s_1 in S]) for a in A])
            π_0[s,:] .= 0
            π_0[s,a] = 1
        end
        if t > 1 && π_1 == π_0
            break
        end
        π_1 = copy(π_0)
    end
    return π_0, Vᵖ, Pᵖ
end 

function PolicyIteration(mdp::TabularMDP, iterations::Int64)
    γ = getDiscountFactor(mdp)
    return PolicyIteration(iterations, getTransitionProbabilities(mdp), meanReward(mdp), γ)
end

function ActionValueIteration(iterations::Int64, P::Array{Float64, 3}, R::Array{Float64,2}, γ::Float64)
    A = 1:size(P,3)
    S = 1:size(P,1)
    Q_0 = zeros(length(S), length(A))
    Q_1 = zeros(length(S), length(A))
    for t in 1:iterations
        for s in S, a in A
            Q_1[s,a] = R[s,a] + γ * sum(P[:, s,a] .* maximum(Q_0, dims = 2))
        end
        Q_0 = copy(Q_1)
    end
    return Q_1
end

function ActionValueIteration(mdp::TabularMDP, iterations::Int64)
    γ = getDiscountFactor(mdp)
    return ActionValueIteration(iterations, getTransitionProbabilities(mdp), meanReward(mdp), γ)
end