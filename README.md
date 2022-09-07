# TabularRL
[![CI action](https://github.com/hataloo/TabularRL.jl/actions/workflows/RunTests.yml/badge.svg)](https://github.com/hataloo/TabularRL.jl/actions/workflows/RunTests.yml)
[![codecov](https://codecov.io/gh/hataloo/TabularRL.jl/branch/master/graph/badge.svg?token=SxWjW2RqJz)](https://codecov.io/gh/hataloo/TabularRL.jl)
## 🚧 Work in progress 🚧 

An experimental library implementing many classical reinforcement learning algorithms such as Q-learning, temporal difference learning, value iteration and policy iteration. The repository has also been a testing ground for using CI on github and creating a package with PkgTemplates.jl.

The library supports arbitrary MDPs with a discrete state and action space. Several MDPs from Sutton & Barto have been implemented such as HallwayMDP, CliffWalking and JacksCarRental. In addition, there are a couple of helpful functions to define arbitrary GridWorld-like MDPs such as FrozenLake.

Finally, I am in the process of creating visualizations of MDPs that can be interacted with. The idea is to allow the user to interact with the MDP but also visualize the algorithms and the optimal policy. 

An example visualization of HallwayMDP with 7 states is shown below. The MDP has two actions: 'go left' and 'go right'. Reaching state 1 or 7 terminates the episode and grants a reward of 5 and 10 respectively. Reaching any other state yields a reward of -1. The transitions are not deterministic. The actions 'go left' and 'go right' have a 90% and 70%, respectively, chance of moving in the requested direction.

![Example of HallwayVisualizationController](https://github.com/hataloo/TabularRL.jl/blob/master/Figures/HallwayVisualisationExample.gif)

