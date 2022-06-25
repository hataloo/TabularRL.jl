using TabularRL

carsMax, moveMax = 20, 5
t = @elapsed begin
mdp, stateMap = getJacksCarRentalMDP(carsMax = carsMax, moveMax = moveMax)
end
println("Time to build MDP: $t")

P_sum = sum(mdp.P, dims = [1])
# Validate the transition probabilities
println("Maximum transition probability: $(maximum(P_sum))")
println("Minimum transitition probability: $(minimum(P_sum))") 

pi_opt, V_opt, P_opt = PolicyIteration(mdp, 100)

valueMatrix = reshape(V_opt, carsMax+1, carsMax+1)
display(valueMatrix)
optimalActionMatrix = reshape([argmax(pi_opt[i,:])-(moveMax+1) for i in 1:size(pi_opt,1)], carsMax+1, carsMax+1)
display(optimalActionMatrix)

