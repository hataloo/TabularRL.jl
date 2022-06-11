function TD0(π::Array{Float64,2}, mdp::MDP, N_episodes::Number, T::Number, α = nothing)
    if α === nothing 
        α = LinRange(1,1e-6, N_episodes) 
    end
    i = 1
    π_d = [DiscreteNonParametric(mdp.A, π[s,:]) for s in mdp.S]
    V = zeros(length(mdp.S))
    for n in 1:N_episodes
        s = sampleInitialState(mdp)
        for t in 1:T
            a = rand(π_d[s])
            s_new, r, done = step(mdp, s, a)
            V[s] += α[i]*(r + mdp.γ*V[s_new] - V[s])
            s = s_new
            if done break end
        end
        i += 1
    end
    return V
end

function TDnStep(π::Array{Float64,2}, mdp::MDP, n::Int64, N_episodes::Number, T::Number, α = nothing)
    if α === nothing 
        α = LinRange(1,1e-6, N_episodes) 
    end
    i = 1
    V = zeros(length(mdp.S))
    γ_power = [mdp.γ^(t) for t in 0:T] # Need to +1 to account for γ^0
    for k in 1:N_episodes
        e = sampleEpisode(mdp, π, T)
        L = length(e)
        for (t,s) in enumerate(e.s)
            G = 0
            if t + n <= L
                for i in 0:(n-1)
                    G += e.r[t+i] * γ_power[i + 1]
                end
                G += γ_power[n+1] * V[e.s[t+n]]
            else 
                G += sum(e.r[t:end] .* γ_power[1:(L-t+1)])
            end
            V[s] += α[i]*(G - V[s])
        end
        i += 1
    end
    return V
end

function TDλ(π::Array{Float64,2}, mdp::MDP, λ::Float64, N_episodes::Number, T::Number, α = nothing)
    if α === nothing 
        α = LinRange(1,1e-6, N_episodes) 
    end
    i = 1
    V = zeros(length(mdp.S))
    π_d = [DiscreteNonParametric(mdp.A, π[s,:]) for s in mdp.S]
    e_t = zeros(length(mdp.S))
    γ = mdp.γ
    for n in 1:N_episodes
        s = sampleInitialState(mdp)
        e_t = zeros(length(mdp.S))
        for t in 1:T
            a = rand(π_d[s])
            s_new, r, done = step(mdp, s, a)
            #TD-error
            δ_t = r + γ * V[s_new] - V[s]
            #Eligibility trace
            e_t = (γ * λ) .* e_t .+ (mdp.S .== s) 
            #Value update
            V .= V .+ (α[n] * δ_t) .* e_t
            s = s_new
            if done break end
        end
        i += 1
    end
    return V
end