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