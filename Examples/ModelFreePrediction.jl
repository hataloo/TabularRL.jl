using TabularRL, OrderedCollections, StatsBase

hallwayMDP = getHallwayMDP(10, 0.999, true)
#S, A, P, R = hallwayMDP.S, hallwayMDP.A, hallwayMDP.P, hallwayMDP.R

t_v = @elapsed begin
V = ValueIteration(hallwayMDP, 100)
end
t_p = @elapsed begin
p_p, V_p, P_p = PolicyIteration(hallwayMDP, 30)
end
println("Time value iteration: ", t_v)
println("Time policy iteration: ", t_p)
println("Maximum difference: ", maximum(V_p - V))

step(hallwayMDP, 3, 1)

ep = sampleEpisode(hallwayMDP, p_p, 10)
N_episodes = 30000
T_max = 500
α = LinRange(1, 1e-6, N_episodes)
t_mc = @elapsed V_mc, n_mc = MonteCarlo(p_p, hallwayMDP, N_episodes, T_max)
t_td0 = @elapsed V_td0 = TD0(p_p, hallwayMDP, N_episodes, T_max, α)
t_tdn = @elapsed V_tdn = TDnStep(p_p, hallwayMDP, 5, N_episodes, T_max, α)
t_lambda = @elapsed V_lambda = TDλ(p_p, hallwayMDP, 0.6, N_episodes, T_max, α)
t_sarsa = @elapsed Q_sarsa = SARSA(p_p, hallwayMDP, N_episodes, T_max, α)
V_sarsa = maximum(Q_sarsa, dims = 2)

maximum(abs.(V_lambda - V_p))
V_results = OrderedDict{String, Any}([("mc", V_mc), ("td0", V_td0), ("tdn", V_tdn), ("λ", V_lambda), ("sarsa", V_sarsa)])
meanMinMaxResults = OrderedDict{String, Any}()
for (k,V) in V_results
    dev = abs.(V - V_p)
    meanMinMaxResults[k] = [minimum(dev), mean(dev), maximum(dev)]
end
meanMinMaxResults