function QLearning(π::GLIEPolicy, mdp::TabularMDP, N_episodes::Number, T::Number, α = nothing)
    reset!(π)
    γ = getDiscountFactor(mdp)
    if α === nothing
        α = LinRange(1, 1e-6, N_episodes)
    end
    Q = zeros(length(getStates(mdp)), length(getActions(mdp)))
    i = 1
    for n in 1:N_episodes
        s = reset(mdp)
        for t in 1:T
            a = sample(π, s, Q)
            s_new, r, done = step(mdp, a)
            Q[s,a] += α[i] * (r + γ * maximum(Q[s_new,a]) - Q[s,a])
            s = s_new
            if done break end
        end
        i += 1
    end
    return Q
end

function DoubleQLearning(π::GLIEPolicy, mdp::TabularMDP, N_episodes::Number, T::Number, α = nothing)
    reset!(π)
    γ = getDiscountFactor(mdp)
    if α === nothing
        α = LinRange(1, 1e-6, N_episodes)
    end
    Q_1 = zeros(length(getStates(mdp)), length(getActions(mdp)))
    Q_2 = zeros(length(getStates(mdp)), length(getActions(mdp)))
    i = 1
    for n in 1:N_episodes
        s = reset(mdp)
        for t in 1:T
            a = sample(π, s, (Q_1 .+ Q_2) ./ 2)
            s_new, r, done = step(mdp, a)
            p = rand()
            if p < 0.5
                Q_1[s,a] += α[i] * (r + γ * Q_2[s_new,argmax(Q_1[s_new,:])] - Q_1[s,a])
            else
                Q_2[s,a] += α[i] * (r + γ * Q_1[s_new,argmax(Q_2[s_new,:])] - Q_2[s,a])
            end
            s = s_new
            if done break end
        end
        i += 1
    end
    return Q_1, Q_2
end
