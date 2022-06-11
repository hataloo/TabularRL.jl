include("TabularRL.jl")


hallwayMDP = getHallwayMDP(10, 0.9)

V_vi = ValueIteration(hallwayMDP, 1000)
Q_vi = ActionValueIteration(hallwayMDP, 1000)

π_ϵ = EpsilonGreedyPolicy(hallwayMDP, 1.0, 0.01, Int64(2e6))
π_ϵ.ϵ_current_iteration
sample(π_ϵ, 10, Q)

N_episodes = 200000
T_max = 100
Q_sarsa = SARSA(π_ϵ, hallwayMDP, N_episodes, T_max)
Q_qlearning = QLearning(π_ϵ, hallwayMDP, N_episodes, T_max)
Q_double1, Q_double2 = DoubleQLearning(π_ϵ, hallwayMDP, N_episodes, T_max)

Q_results = OrderedDict{String, Any}([
    ("sarsa", Q_sarsa), 
    ("Q-learning", Q_qlearning), 
    ("DoubleQ", Q_double1)])
Q_resultsSummary = OrderedDict{String, Any}()
for (k,Q_res) in Q_results
    Qdeviation = abs.(Q_res - Q_vi)
    Vdeviation = abs.(Q_res[argmax(Q_res, dims = 2)] - V_vi)
    PolicyDeviation = argmax(Q_res, dims = 2) .!= argmax(Q_vi, dims = 2)
    Q_resultsSummary[k] = OrderedDict([
        ("meanQDev", mean(Qdeviation)),
        ("minQDev", minimum(Qdeviation)),
        ("maxQDev", maximum(Qdeviation)),
        ("meanVDev", mean(Qdeviation)),
        ("minVDev", minimum(Qdeviation)),
        ("maxVDev", maximum(Qdeviation)),
        ("meanPolicyDeviation", mean(PolicyDeviation)),
    ])
end
Q_resultsSummary