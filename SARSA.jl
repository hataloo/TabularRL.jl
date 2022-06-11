function SARSA(π::Array{Float64,2}, mdp::MDP, N_episodes::Number, T::Number)
    α = LinRange(1, 1e-6, N_episodes*T)
    π_d = [DiscreteNonParametric(mdp.A, π[s,:]) for s in mdp.S]
    Q = zeros(length(mdp.S), length(mdp.A))
    i = 1
    for n in 1:N_episodes
        s = sampleInitialState(mdp)
        a = rand(π_d[s])
        for t in 1:T
            s_new, r = step(mdp, s, a)
            a_new = rand(π_d[s_new])
            Q[s, a] += α[i] * (r + mdp.γ * Q[s_new, a_new] - Q[s,a])
            i += 1
            s, a = s_new, a_new
        end
    end
    return Q
end

function SARSA(π::GLIEPolicy, mdp::MDP, N_episodes::Number, T::Number)
    reset!(π)
    Q = zeros(length(mdp.S), length(mdp.A))
    i = 1
    α = LinRange(1, 1e-6, N_episodes*T)
    for n in 1:N_episodes
        s = sampleInitialState(mdp)
        a = sample(π, s, Q)
        for t in 1:T
            s_new, r = step(mdp, s, a)
            a_new = sample(π, s_new, Q)
            Q[s,a] += α[i] * (r + mdp.γ * Q[s_new, a_new] - Q[s,a])
            i += 1
            s, a = s_new, a_new 
        end
    end
    return Q
end