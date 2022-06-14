include("TabularRL.jl")

cliffWalkingMDP = getCliffWalkingMDP(7, 10, 0.9, 0.05)
P, S, A = cliffWalkingMDP.P, cliffWalkingMDP.S, cliffWalkingMDP.A
println("Maximum P: $(maximum(sum(P, dims = [1])))")
println("Minimum P: $(minimum(sum(P, dims = [1])))")
println("Max deviation from 1: $(maximum(abs.(sum(P, dims = [1]) .-1)))")
println("$(argmax(P))")

sampleEpisode(cliffWalkingMDP, createUniformPolicy(cliffWalkingMDP), 500)