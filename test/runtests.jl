using TabularRL
using Test

@testset verbose = true "TabularRL.jl" begin
    
detect_unbound_args(TabularRL)
@testset "Envs" begin

@testset "HallwayMDP" begin
@test typeof(getHallwayMDP(10, 0.9)) <: TabularMDP
@test isa(getHallwayMDP(2, 0.5, true), TabularMDP)
@test_throws DomainError getHallwayMDP(1, 0.5, true)
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
