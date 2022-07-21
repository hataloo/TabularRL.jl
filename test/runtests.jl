using TabularRL
using Distributions
using Test

@testset verbose = true "TabularRL.jl" begin
    
detect_unbound_args(TabularRL)
@testset "Algorithms" begin
    #TODO: Add correctness tests
    local mdp = getHallwayMDP(5, 0.95)
    local π = getUniformPolicy(mdp)
    @testset "ValuePolicyIteration" begin
        # Test if ValueIteration runs.  
        @test isa(ValueIteration(mdp, 2), Any)
        # Test if PolicyIteration runs.
        @test isa(PolicyIteration(mdp, 2), Any)
        # Test if ActionValueIteration runs.
        @test isa(ActionValueIteration(mdp, 2), Any)
    end
    @testset "MonteCarlo" begin
        # Test if MonteCarlo algorithm runs.
        @test isa(MonteCarlo(π.π_vals, mdp, 2, 10), Any)
    end
    @testset "QLearning" begin
        local π = EpsilonGreedyPolicy(mdp, 1.0, 0.0, 10)
        # Test if QLearning runs.
        @test isa(QLearning(π, mdp, 2, 10), Any)
        # Test if DoubleQLearning runs.
        @test isa(DoubleQLearning(π, mdp, 2, 10), Any)
    end
    @testset "SARSA" begin
        # Test if SARSA runs using array of policy.
        @test isa(SARSA(π.π_vals, mdp, 2, 10), Any)
        local πGLIE = EpsilonGreedyPolicy(mdp, 1.0, 0.0, 10)
        # Test if SARSA runs using GLIE policy.
        @test isa(SARSA(πGLIE, mdp, 2, 10), Any)
        # Test if ExpectedSARSA runs using GLIE policy.
        @test isa(ExpectedSARSA(πGLIE, mdp, 2, 10), Any)
    end
    @testset "TDLearning" begin
        @test isa(TD0(π.π_vals, mdp, 2, 10), Any)
        for n = 1:3
            @test isa(TDnStep(π.π_vals, mdp, n, 2, 10), Any)
        end
        for λ in [0.0, 0.5, 1.0]
            @test isa(TDλ(π.π_vals, mdp, λ, 2, 10), Any)
        end
    end
end

@testset "Policy" begin
    # Test sampling an episode using a uniform policy.
    @test begin
        mdp = getHallwayMDP(10, 0.9)
        π = getUniformPolicy(mdp)
        isa(sampleEpisode(mdp, π, 100), Episode)
    end
    # Test that EpsilonGreedyPolicy converges to a = argmax Q
    @test begin 
        mdp = getHallwayMDP(10, 0.9)
        ϵ_iter = 5
        π = EpsilonGreedyPolicy(mdp, 1.0, 0.0, ϵ_iter)
        Q = fill(0.0, length(mdp.S), length(mdp.A))
        Q[1,1] = 1.0
        for _ = 1:ϵ_iter sample(π, 1, Q) end
        sample(π, 1, Q) == 1
    end
    # Test that BoltzmannPolicy converges to a = argmax Q for Q[1,1] = 1.0
    @test begin 
        mdp = getHallwayMDP(3, 0.9)
        β_iter = 5
        π = BoltzmannPolicy(mdp, 0.0, 1e2, β_iter)
        Q = fill(0.0, length(mdp.S), length(mdp.A))
        Q[1,1] = 1.0
        for _ = 1:β_iter sample(π, 1, Q) end
        #Probability of returning 1: exp(1e2) / (exp(1e2) + 1), i.e. essentially 1.
        sample(π, 1, Q) == 1
    end
    # Test that BoltzmannPolicy does not converge to a = 1 when Q[1,1] = -1.0 and Q[1,a] = 0.0 for a != 1.
    @test begin 
        mdp = getHallwayMDP(3, 0.9)
        β_iter = 5
        π = BoltzmannPolicy(mdp, 0.0, 1e2, β_iter)
        Q = fill(0.0, length(mdp.S), length(mdp.A))
        Q[1,1] = -1.0
        for _ = 1:β_iter sample(π, 1, Q) end
        #Probability of returning 1: exp(-1e2) / (exp(-1e2) + 1), i.e. essentially 0.
        sample(π, 1, Q) != 1
    end
end

@testset "MDPWrapper" begin
    struct TempMDPWrapper <: AbstractMDPWrapper{
        Int64, DiscreteSpace{Int64},
        Int32, DiscreteSpace{Int32},
        DiscreteNonParametric,
        AbstractMDP
    } end
    local tempMDPWrapper = TempMDPWrapper()
    @test_throws NotImplementedError hasTerminated(tempMDPWrapper)
    @test_throws NotImplementedError reset(tempMDPWrapper)
    @test_throws NotImplementedError reset!(tempMDPWrapper)
    @test_throws MethodError step(tempMDPWrapper, 1)

    local mdp = getHallwayMDP(5, 0.95, true, true)
    local wrappedMDP = VerboseWrapper(mdp)
    @test reset(wrappedMDP) == 3
    @test begin 
        reset(wrappedMDP)
        true
    end
    @test hasTerminated(wrappedMDP) == false
    @test step(wrappedMDP, 1) == (2, -1.0, false)
end

@testset "Envs" begin
#TODO: Test functions in GridWorldUtility.jl

@testset "HallwayMDP" begin
@test typeof(getHallwayMDP(10, 0.9)) <: TabularMDP
@test isa(getHallwayMDP(3, 0.5, true), TabularMDP)
@test_throws DomainError getHallwayMDP(2, 0.5, true)
    @testset "HallwayVisualization" begin
        @test isa(HallwayVisualizationWrapper(getHallwayMDP(3, 0.9, true, true)), 
            AbstractTabularMDP)
        local hvw = HallwayVisualizationWrapper(getHallwayMDP(3, 0.9, true, true))
        @test reset!(hvw) === nothing
        @test reset(hvw) == 2
        @test step(hvw, 1) == (1, 5.0, true)
        reset!(hvw)
        @test step(hvw, 2) == (3, 10.0, true)
    end
end

@testset "CliffWalkingMDP" begin
@test typeof(getCliffWalkingMDP(3, 4, 0.9, 0.1, false)) <: TabularMDP
@test typeof(getCliffWalkingMDP(9, 2, 0.9, 0.1, true)) <: TabularMDP
@test_throws DomainError getCliffWalkingMDP(0, 2, 0.9, 0.0, false)
@test_throws DomainError getCliffWalkingMDP(2, 0, 0.9, 0.0, false)
@test_throws DomainError getCliffWalkingMDP(2, 2, 0.9, 2.0, false)
@test_throws DomainError getCliffWalkingMDP(2, 2, 0.9, -0.1, false)
end

@testset "FrozenLakeMDP" begin   
@test isa(getFrozenLakeMDP(γ = 0.5), TabularMDP)
@test_throws KeyError getFrozenLakeMDP("notAValidName", 0.3, γ = 0.5)
@test_throws DomainError getFrozenLakeMDP("4x4", 1.5, γ = 0.5)
@test_throws AssertionError getFrozenLakeMDP(["SH", "FF"], 0.5, γ = 0.5)
@test_throws AssertionError getFrozenLakeMDP(["FH", "FG"], 0.5, γ = 0.5)
@test_throws AssertionError getFrozenLakeMDP(["FHF", "FG"], 0.5, γ = 0.5)
@test isa(getFrozenLakeMDP(["SH", "FG"], 0.5, γ = 0.5), TabularMDP)
end

@testset "getJacksCarRentalMDP" begin
@test isa(getJacksCarRentalMDP(0.5; carsMax = 4, moveMax = 3), Tuple{TabularMDP, Vector{NTuple{2, Int64}}})
@test_throws DomainError getJacksCarRentalMDP(0.5; carsMax = 2, moveMax = 0)
@test_throws DomainError getJacksCarRentalMDP(0.5; carsMax = 1, moveMax = 1)
@test_throws AssertionError getJacksCarRentalMDP(0.5; λ_requests = (0,-1), carsMax = 1, moveMax = 1)
@test_throws AssertionError getJacksCarRentalMDP(0.5; λ_returns = (0,-1), carsMax = 1, moveMax = 1)
end

end
end
