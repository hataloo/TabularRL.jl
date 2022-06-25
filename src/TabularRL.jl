module TabularRL

using StatsBase, Distributions, DataStructures, LinearAlgebra, OrderedCollections
using LogExpFunctions: softmax
import Distributions:convolve
import Base.length, Base.step

include("MDP.jl")
include("Policy.jl")
include("Utils.jl")
include("Algorithms/ValuePolicyIteration.jl")
include("Algorithms/MonteCarlo.jl")
include("Algorithms/TDLearning.jl")
include("Algorithms/SARSA.jl")
include("Algorithms/QLearning.jl")
include("Envs/HallwayMDP.jl")
include("Envs/GridWorlds.jl")
include("Envs/CliffWalking.jl")
include("Envs/JacksCarRental.jl")


export 
    #MDP.jl
    TabularMDP, 
    sampleInitialState,
    sampleNextState,
    sampleState,
    step,
    Episode,
    sampleEpisode,
    #Policy.jl
    AbstractPolicy,
    GLIEPolicy,
    Policy,
    getUniformPolicy,
    sample,
    EpsilonGreedyPolicy,
    sample,
    BoltzmannPolicy,
    getUniformPolicy,
    #Algorithms:
        #ValuePolicyIteration
        ValueIteration,
        PolicyIteration,
        ActionValueIteration,
        #MonteCarlo
        MonteCarlo,
        #TDLearning
        TD0,
        TDnStep,
        TDλ,
        #SARSA
        SARSA,
        ExpectedSARSA,
        #QLearning,
        QLearning,
        DoubleQLearning,
    #Envs:
        #HallwayMDP.jl
        getHallwayMDP,
        #GridWorlds.jl
        getMovementFunctions,
        getWrappingMovementFunctions,
        slipperyMovementProbabilities,
        buildGridWalkTransitionProbabilities,
        buildSlipperyGridTransitionProbabilities,
        addTerminalState!,
        addResettingState!,
        #CliffWalking.jl
        getCliffWalkingMDP,
        #JacksCarRental.jl
        getJacksCarRentalMDP
end
