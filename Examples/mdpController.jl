using TabularRL
mdp = getHallwayMDP(10, 0.95)
controller = MDPController(mdp, Int64)