using TabularRL
using Distributions

mdp = getHallwayMDP(3, .95, true, true)
hvc = HallwayVisualController(mdp)
hvc.fig
