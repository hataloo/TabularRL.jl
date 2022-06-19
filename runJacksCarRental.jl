include("TabularRL.jl")

P, mdp, stateMap = getJacksCarRentalMDP(carsMax = 20, moveMax = 5)


P_sum = sum(P, dims = [1])
println("Maximum transition probability: $(maximum(P_sum))")
println("Minimum transitition probability: $(minimum(P_sum))") 

pi_opt, V_opt, P_opt = PolicyIteration(mdp, 100)

reshape(V_opt, 21, 21)

reshape([argmax(pi_opt[i,:])-6 for i in 1:size(pi_opt,1)], 21, 21)