
function MonteCarlo(π::Array{Float64,2}, mdp::TabularMDP, N_episodes::Number, T::Number)
    #π[s,a]
    max_dev = maximum(abs.([sum(π[s,:]) for s in getStates(mdp)] .- 1))
    if max_dev > 1e-10
        println("Improper policy, sum(π[s,:]) deviates from 1 by at most:", max_dev)
    end
    n = zeros(length(getStates(mdp)))
    V = zeros(length(getStates(mdp)))
    γ = getDiscountFactor(mdp)
    γ_power = [γ^(t) for t in 0:T] # Need to +1 to account for γ^0
    for k in 1:N_episodes
        e = sampleEpisode(mdp, π, T)
        for (t,s) in enumerate(e.states)
            G = sum([γ_power[τ-t+1] * e.rewards[τ] for τ in (t):length(e)])
            n[s] += 1
            V[s] += 1/n[s]*(G - V[s])
        end
    end
    return V, n
end