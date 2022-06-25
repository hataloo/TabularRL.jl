using TabularRL
using Test

@testset verbose = true "TabularRL.jl" begin
    # Write your tests here.
    detect_unbound_args(TabularRL)
    @testset "Envs" begin
        @test typeof(getHallwayMDP(10, 0.9)) <: TabularMDP
        
        @test typeof(getCliffWalkingMDP(3, 4, 0.9, 0.1, false)) <: TabularMDP
        @test typeof(getCliffWalkingMDP(9, 2, 0.9, 0.1, true)) <: TabularMDP
        @test_throws DomainError getCliffWalkingMDP(0, 2, 0.9, 0.0, false)
        @test_throws DomainError getCliffWalkingMDP(2, 0, 0.9, 0.0, false)
        @test_throws DomainError getCliffWalkingMDP(2, 2, 0.9, 2.0, false)
        @test_throws DomainError getCliffWalkingMDP(2, 2, 0.9, -0.1, false)
    
        @test typeof(getFrozenLakeMDP(γ = 0.5)) <: TabularMDP
        @test_throws KeyError getFrozenLakeMDP("notAValidName", 0.3, γ = 0.5)
        @test_throws DomainError getFrozenLakeMDP("4x4", 1.5, γ = 0.5)
    end
end
