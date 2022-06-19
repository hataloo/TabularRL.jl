include("TabularRL.jl")
carsMax, moveMax = 12, 5
P, mdp, stateMap = getJacksCarRentalMDP(carsMax = carsMax, moveMax = moveMax)


P_sum = sum(P, dims = [1])
println("Maximum transition probability: $(maximum(P_sum))")
println("Minimum transitition probability: $(minimum(P_sum))") 

pi_opt, V_opt, P_opt = PolicyIteration(mdp, 100)

reshape(V_opt, carsMax+1, carsMax+1)

reshape([argmax(pi_opt[i,:])-(moveMax+1) for i in 1:size(pi_opt,1)], carsMax+1, carsMax+1)

