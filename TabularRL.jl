#module TabularRL

    include("MDP.jl")
    include("Policy.jl")
    include("HallwayMDP.jl")
    include("ValuePolicyIteration.jl")
    include("MonteCarlo.jl")
    include("TDLearning.jl")
    include("SARSA.jl")
    include("QLearning.jl")
    export 
        #MDP.jl
        TabularMDP, 
        sampleInitialState,
        sampleNextState,
        sampleStateFromDist,
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