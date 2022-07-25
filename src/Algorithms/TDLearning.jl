function TD0(π::Array{Float64,2}, mdp::TabularMDP, N_episodes::Number, T::Number, α = nothing)
    γ = getDiscountFactor(mdp)
    if α === nothing 
        α = LinRange(1,1e-6, N_episodes) 
    end
    i = 1
    actions = [a for a in getActions(mdp)]
    π_d = [DiscreteNonParametric(actions, π[s,:]) for s in getStates(mdp)]
    V = zeros(length(getStates(mdp)))
    for n in 1:N_episodes
        s = reset(mdp)
        for t in 1:T
            a = rand(π_d[s])
            s_new, r, done = step(mdp, a)
            V[s] += α[i]*(r + γ*V[s_new] - V[s])
            s = s_new
            if done break end
        end
        i += 1
    end
    return V
end

function TDnStep(π::Array{Float64,2}, mdp::TabularMDP, n::Int64, N_episodes::Number, T::Number, α = nothing)
    γ = getDiscountFactor(mdp)
    if α === nothing 
        α = LinRange(1,1e-6, N_episodes) 
    end
    i = 1
    V = zeros(length(getStates(mdp)))
    γ_power = [γ^(t) for t in 0:T] # Need to +1 to account for γ^0
    for k in 1:N_episodes
        e = sampleEpisode(mdp, π, T)
        L = length(e)
        for (t,s) in enumerate(e.states)
            G = 0
            if t + n <= L
                for i in 0:(n-1)
                    G += e.rewards[t+i] * γ_power[i + 1]
                end
                G += γ_power[n+1] * V[e.states[t+n]]
            else 
                G += sum(e.rewards[t:end] .* γ_power[1:(L-t+1)])
            end
            V[s] += α[i]*(G - V[s])
        end
        i += 1
    end
    return V
end

function TDλ(π::Array{Float64,2}, mdp::TabularMDP, λ::Float64, N_episodes::Number, T::Number, α = nothing)
    if α === nothing 
        α = LinRange(1,1e-6, N_episodes) 
    end
    i = 1
    V = zeros(length(getStates(mdp)))
    actions = [a for a in getActions(mdp)]
    π_d = [DiscreteNonParametric(actions, π[s,:]) for s in getStates(mdp)]
    e_t = zeros(length(getStates(mdp)))
    γ = getDiscountFactor(mdp)
    S = Vector{Int64}(1:length(getStates(mdp)))
    for n in 1:N_episodes
        s = reset(mdp)
        e_t = zeros(length(getStates(mdp)))
        for t in 1:T
            a = rand(π_d[s])
            s_new, r, done = step(mdp, a)
            #TD-error
            δ_t = r + γ * V[s_new] - V[s]
            #Eligibility trace
            e_t = (γ * λ) .* e_t .+ (S .== s) 
            #Value update
            V .= V .+ (α[n] * δ_t) .* e_t
            s = s_new
            if done break end
        end
        i += 1
    end
    return V
end