using GLMakie
using TabularRL
using Distributions

mdp = getHallwayMDP(7, .95, true, false)
hvc = HallwayVisualController(mdp)
hvc.fig
