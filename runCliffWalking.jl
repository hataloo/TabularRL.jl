include("TabularRL.jl")

height = 6
width = 12
cliffWalkingMDP = getCliffWalkingMDP(height, width, 0.99, 0.4)
P, S, A = cliffWalkingMDP.P, cliffWalkingMDP.S, cliffWalkingMDP.A
println("Maximum P: $(maximum(sum(P, dims = [1])))")
println("Minimum P: $(minimum(sum(P, dims = [1])))")
println("Max deviation from 1: $(maximum(abs.(sum(P, dims = [1]) .-1)))")
println("$(argmax(P))")

ep = sampleEpisode(cliffWalkingMDP, getUniformPolicy(cliffWalkingMDP), 5000)


π_ϵ = EpsilonGreedyPolicy(cliffWalkingMDP, 1.0, 0.05, Int64(2e6))


Q_vi = ActionValueIteration(cliffWalkingMDP, 1000)
Q_double1, Q_double2 = DoubleQLearning(π_ϵ, cliffWalkingMDP, 5000, 300)

optimalPolicy = [argmax(Q_vi[s, :]) for s in cliffWalkingMDP.S]
doubleQPolicy = [argmax((Q_double1 + Q_double2)[s, :]) for s in cliffWalkingMDP.S]
print(sum(optimalPolicy .== doubleQPolicy))
display(reshape(Q_vi, height, width, 4))

display(reshape(optimalPolicy, height, width))