include("TabularRL.jl")

hallway = getHallwayMDP(10, 0.9)
ep = sampleEpisode(hallway, createUniformPolicy(hallway).π_vals, 100)
println(ep.s)
println(ep.a)
println(ep.r)
