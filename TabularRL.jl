#module TabularRL

include("MDP.jl")
include("Policy.jl")
include("Utils.jl")
include("Algorithms/ValuePolicyIteration.jl")
include("Algorithms/MonteCarlo.jl")
include("Algorithms/TDLearning.jl")
include("Algorithms/SARSA.jl")
include("Algorithms/QLearning.jl")
include("Envs/HallwayMDP.jl")
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
        #QLearning,
        QLearning,
        DoubleQLearning,
    #Envs:
        #HallwayMDP.jl
        getHallwayMDP,
        #CliffWalking.jl
        getCliffWalkingMDP,
        buildGridWalkTransitionProbabilities
        getJacksCarRentalMDP
#end