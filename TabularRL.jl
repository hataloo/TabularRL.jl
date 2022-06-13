#module TabularRL

include("MDP.jl")
include("Policy.jl")
include("Envs/HallwayMDP.jl")
include("Algorithms/ValuePolicyIteration.jl")
include("Algorithms/MonteCarlo.jl")
include("Algorithms/TDLearning.jl")
include("Algorithms/SARSA.jl")
include("Algorithms/QLearning.jl")
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
    createUniformPolicy,
    sample,
    EpsilonGreedyPolicy,
    sample,
    BoltzmannPolicy,
    #HallwayMDP.jl
    getHallwayMDP,
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
    DoubleQLearning
    

#end