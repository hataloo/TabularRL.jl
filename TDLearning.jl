function TD0(π::Array{Float64,2}, mdp::MDP, N_episodes::Number, T::Number)
    α = LinRange(1,1e-6, N_episodes*T)
    i = 1
    π_d = [DiscreteNonParametric(mdp.A, π[s,:]) for s in mdp.S]
    V = zeros(length(mdp.S))
    for n in 1:N_episodes
        s = sampleInitialState(mdp)
        for t in 1:T
            a = rand(π_d[s])
            s_new, r = step(mdp, s, a)
            V[s] += α[i]*(r + mdp.γ*V[s_new] - V[s])
            s = s_new
            i += 1
        end
    end
    return V
end

function TDnStep(π::Array{Float64,2}, mdp::MDP, n::Int64 ,N_episodes::Number, T::Number)
    α = LinRange(1,1e-6, N_episodes*T)
    i = 1
    V = zeros(length(mdp.S))
    for k in 1:N_episodes
        e = sampleEpisode(mdp, π, T)
        for (t,s) in enumerate(e.s[1:(length(e.s)-1)])
            G = sum( [ mdp.γ^(τ-t) * e.r[τ] for τ in t:minimum([t+n-1,T-1])]) + mdp.γ^(n) * V[e.s[t+1]]
            V[s] += α[i]*(G - V[s])
            i += 1
        end
    end
    return V
end

function TDλ(π::Array{Float64,2}, mdp::MDP, λ::Float64, N_episodes::Number, T::Number)
    α = LinRange(0.1,1e-6, N_episodes*T)
    i = 1
    V = zeros(length(mdp.S))
    π_d = [DiscreteNonParametric(mdp.A, π[s,:]) for s in mdp.S]
    e_t = zeros(length(mdp.S))
    γ = mdp.γ
    for n in 1:N_episodes
        s = sampleInitialState(mdp)
        for t in 1:T
            a = rand(π_d[s])
            s_new, r = step(mdp, s, a)
            #TD-error
            δ_t = r + γ * V[s_new] - V[s]
            #Eligibility trace
            e_t = (γ * λ) .* e_t .+ (mdp.S .== s) 
            #Value update
            V .= V .+ (α[n] * δ_t) .* e_t
            s = s_new
            i += 1
        end
    end
    return V
end