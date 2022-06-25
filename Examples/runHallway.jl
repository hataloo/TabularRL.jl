using TabularRL

hallway = getHallwayMDP(10, 0.9)
ep = sampleEpisode(hallway, getUniformPolicy(hallway).π_vals, 100)
println(ep.states)
println(ep.actions)
println(ep.rewards)
