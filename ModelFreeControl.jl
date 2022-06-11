include("TabularRL.jl")


hallwayMDP = getHallwayMDP(10, 0.9)


Q = ActionValueIteration(hallwayMDP, 500)

π_ϵ = EpsilonGreedyPolicy(hallwayMDP, 1.0, 0.01, Int64(2e6))
π_ϵ.ϵ_current_iteration
sample(π_ϵ, 10, Q)

Q_sarsa = SARSA(π_ϵ, hallwayMDP, 3000, 1000)
Q_qlearning = QLearning(π_ϵ, hallwayMDP, 3000, 1000)
Q_double1, Q_double2 = DoubleQLearning(π_ϵ, hallwayMDP, 1000, 2000)