module TabularRL

using StatsBase, Distributions, DataStructures, LinearAlgebra, OrderedCollections
using Makie, Graphs, SimpleWeightedGraphs, GraphMakie
using LogExpFunctions: softmax
import Distributions:convolve
import Base.length, Base.step, Base.reset, Base.showerror, Base.maximum, Base.iterate
import Distributions.sample

include("Space.jl")
include("MDP.jl")
include("MDPWrapper.jl")
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
    # Space.jl
    Space,
    getSpaceType,
    DiscreteSpace,
    ContinuousSpace,
    DiscreteContiguousSpace,
    ContinuousContiguousSpace,
    # MDP.jl
    # Abstract types
    TupleUnion,
    AbstractMDP,
    AbstractTabularMDP,
    AbstractDiscreteMDP,
    hasTerminated,
    reset,
    reset!,
    step,
    # MDPWrapper 
    AbstractMDPWrapper,
    VerboseWrapper,
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
    getCurrentState,
    hasTerminated,
    sampleInitialState,
    isTerminalState,
    getMeanRewards,
    sampleNextState,
    sampleState,
    sampleReward,
    step,
    # Episode
    Episode,
    sampleEpisode,
    # Policy.jl
    AbstractPolicy,
    GLIEPolicy,
    TabularPolicy,
    getUniformPolicy,
    sample,
    EpsilonGreedyPolicy,
    sample,
    BoltzmannPolicy,
    getUniformPolicy,
    # Algorithms:
        # ValuePolicyIteration
        ValueIteration,
        PolicyIteration,
        ActionValueIteration,
        # MonteCarlo
        MonteCarlo,
        # TDLearning
        TD0,
        TDnStep,
        TDλ,
        # SARSA
        SARSA,
        ExpectedSARSA,
        # QLearning,
        QLearning,
        DoubleQLearning,
    # Envs:
        # HallwayMDP.jl
        getHallwayMDP,
        HallwayVisualizationWrapper,
        HallwayVisualController,
        # GridWorlds.jl
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
        # CliffWalking.jl
        getCliffWalkingMDP,
        # JacksCarRental.jl
        getJacksCarRentalMDP,
        # FrozenLake.jl
        getFrozenLakeMDP,
        buildFrozenLakeFromLayout,
    # Utils:
    NotImplementedError
end
