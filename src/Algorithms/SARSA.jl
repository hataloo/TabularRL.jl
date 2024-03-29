function SARSA(π::Array{Float64,2}, mdp::TabularMDP, N_episodes::Number, T::Number, α = nothing)
    γ = getDiscountFactor(mdp)
    if α === nothing 
        α = LinRange(1,1e-6, N_episodes) 
    end
    numActions = length(getActions(mdp))
    π_d = [DiscreteNonParametric(1:numActions, π[s,:]) for s in getStates(mdp)]
    Q = zeros(length(getStates(mdp)), length(getActions(mdp)))
    i = 1
    for n in 1:N_episodes
        s = reset(mdp)
        a = rand(π_d[s])
        for t in 1:T
            s_new, r, done = step(mdp, a)
            a_new = rand(π_d[s_new])
            Q[s, a] += α[i] * (r + γ * Q[s_new, a_new] - Q[s,a])
            s, a = s_new, a_new
            if done break end
        end
        i += 1
    end
    return Q
end

function SARSA(π::GLIEPolicy, mdp::TabularMDP, N_episodes::Number, T::Number, α = nothing)
    reset!(π)
    γ = getDiscountFactor(mdp)
    if α === nothing 
        α = LinRange(1,1e-6, N_episodes) 
    end
    Q = zeros(length(getStates(mdp)), length(getActions(mdp)))
    i = 1
    for n in 1:N_episodes
        s = reset(mdp)
        a = sample(π, s, Q)
        for t in 1:T
            s_new, r, done = step(mdp, a)
            a_new = sample(π, s_new, Q)
            Q[s,a] += α[i] * (r + γ * Q[s_new, a_new] - Q[s,a])
            s, a = s_new, a_new 
            if done break end
        end
        i += 1
    end
    return Q
end


function ExpectedSARSA(π::GLIEPolicy, mdp::TabularMDP, N_episodes::Number, T::Number, α = nothing)
    reset!(π)
    γ = getDiscountFactor(mdp)
    if α === nothing 
        α = LinRange(1,1e-6, N_episodes) 
    end
    Q = zeros(length(getStates(mdp)), length(getActions(mdp)))
    i = 1
    for n in 1:N_episodes
        s = reset(mdp)
        a = sample(π, s, Q)
        for t in 1:T
            s_new, r, done = step(mdp, a)
            a_new = sample(π, s_new, Q)
            Q[s,a] += α[i] * (r + γ * sum(getActionProbabilities(π, s_new, Q) .* Q[s_new, :]) - Q[s,a])
            s, a = s_new, a_new 
            if done break end
        end
        i += 1
    end
    return Q
end