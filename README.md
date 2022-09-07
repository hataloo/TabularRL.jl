# TabularRL
[![CI action](https://github.com/hataloo/TabularRL.jl/actions/workflows/RunTests.yml/badge.svg)](https://github.com/hataloo/TabularRL.jl/actions/workflows/RunTests.yml)
[![codecov](https://codecov.io/gh/hataloo/TabularRL.jl/branch/master/graph/badge.svg?token=SxWjW2RqJz)](https://codecov.io/gh/hataloo/TabularRL.jl)
## 🚧 Work in progress 🚧 

An experimental library implementing many classical reinforcement learning algorithms such as Q-learning, temporal difference learning, value iteration and policy iteration.

The library has a general struct called TabularMDP to define an MDP with a discrete state and action space. Several MDPs from Sutton & Barto have been implemented such as HallwayMDP, CliffWalking and JacksCarRental. There exist helpful functions to define arbitrary GridWorld-like MDPs, see FrozenLake.

Finally, I am in the process of creating visualizations of MDPs that can be interacted with. Currently, only HallwayMDP has a visualization. See the example below: 

![Example of HallwayVisualizationController](./Figures/HallwayVisualizationExample.gif)

