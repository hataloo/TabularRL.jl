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
include("Envs/GridWorldUtility.jl")
include("Envs/CliffWalking.jl")
include("Envs/JacksCarRental.jl")
include("Envs/FrozenLake.jl")


export 
    #MDP.jl
    # Abstract types
    Space,
    DiscreteSpace,
    ContinuousSpace,
    AbstractMDP,
    AbstractTabularMDP,
    AbstractDiscreteMDP,
    # MDPController
    MDPController,
    getMDP,
    getState,
    hasTerminated,
    reset,
    reset!,
    # TabularMDP
    TabularMDP,
    getStates,
    getActions,
    getTransitionProbabilities,
    getTransitionDistributions,
    getRewardDistributions,
    getInitialStateDistribution,
    getTerminalMask,
    getDiscountFactor,
    sampleInitialState,
    isTerminalState,
    meanReward,
    sampleNextState,
    sampleState,
    sampleReward,
    step,
    # Episode
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
        buildGridWorldTabularMDP,
        getMovementFunctions,
        getWrappingMovementFunctions,
        slipperyMovementProbabilities,
        buildGridWalkTransitionProbabilities,
        buildSlipperyGridTransitionProbabilities,
        addTerminalState!,
        addResettingState!,
        findCharacterCoordinateInLayout,
        findAllCharacterCoordinatesInLayout,
        #CliffWalking.jl
        getCliffWalkingMDP,
        #JacksCarRental.jl
        getJacksCarRentalMDP,
        #FrozenLake.jl
        getFrozenLakeMDP,
        buildFrozenLakeFromLayout
end
