using TabularRL
height, width = 8,8
mdp = getFrozenLakeMDP("$(height)x$(width)", 0.5, γ = 0.95)

episode = sampleEpisode(mdp, getUniformPolicy(mdp), 1000)
print(length(episode))

π_opt, V_opt, _ = PolicyIteration(mdp, 100)

actionArrows = [:⬅,  :➡, :⬆, :⬇]
π_optArrows = reshape([actionArrows[argmax(π_opt[i,:])] for i in 1:size(π_opt,1)], height, width)
display(π_optArrows)
V_optMatrix = reshape(V_opt, height, width)
display(V_optMatrix)

π_ϵ = EpsilonGreedyPolicy(mdp, 1.0, 0.1, Int64(3e6))
N_episodes, T_max = 10000, 1000
Q_double1, Q_double2 = DoubleQLearning(π_ϵ, mdp, N_episodes, T_max)

Q_double = (Q_double1 + Q_double2)/2
display(reshape(Q_double, height, width, 4))

π_QDoubleArrow = reshape([actionArrows[argmax(Q_double[i,:])] for i in 1:size(Q_double,1)], height, width)
display(π_QDoubleArrow)