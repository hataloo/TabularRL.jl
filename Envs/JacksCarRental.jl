using Distributions, DataStructures

function getJacksCarRentalMDP(γ = 0.9; λ_requests::Vector{Int64} = [3, 4], λ_returns::Vector{Int64} = [3, 2], carsMax::Int = 20, moveMax::Int = 5)
    @assert length(λ_requests) == length(λ_returns) "Length of requests and returns must be equal, given $(length(λ_requests)) and $(length(λ_returns))"
    @assert length(λ_requests) == 2 "Number of locations (requests and returns) must equal 2"
    numRentalLocations = length(λ_requests)
    S = Vector{Int64}(1:((carsMax+1) ^ numRentalLocations))
    A = Vector{Int64}(-moveMax:moveMax)

    #P[s', r' | s, a]
    stateRewardTransitionProbs = fill(0.0, repeat([carsMax+1], 3 * numRentalLocations)..., 2 *moveMax + 1)
    for s_old1 = 0:carsMax, s_old2 = 0:carsMax
        for s_new1 = 0:carsMax, s_new2 = 0:carsMax
            for (i_a, a) in enumerate(A)
                movedCars = clamp(a, -minimum([s_old1, carsMax - s_old2]), minimum([s_old2, carsMax - s_old1]))
                stock1 = s_old1 + movedCars
                stock2 = s_old2 - movedCars
                #println("$s_old1, $s_old2, $stock1, $stock2, $a, $movedCars")
                reqDist1 = censored(Poisson(λ_requests[1]), 0, stock1)
                reqDist2 = censored(Poisson(λ_requests[2]), 0, stock2)
                retDist1 = censored(Poisson(λ_returns[1]), 0, carsMax - stock1)
                retDist2 = censored(Poisson(λ_returns[2]), 0, carsMax - stock2)
                for req1 in 0:stock1, req2 in 0:stock2
                    ret1 = req1 + s_new1 - stock1
                    ret2 = req2 + s_new2 - stock2
                    stateRewardTransitionProbs[
                    s_new1+1, s_new2+1, 
                    s_old1+1, s_old2+1,
                    req1+1, req2+1, i_a] = pdf(reqDist1, req1) * pdf(reqDist2, req2) *
                        pdf(retDist1, ret1) * pdf(retDist2, ret2)
                end
            end
        end
    end 
    rewardDistributions = Array{DiscreteNonParametric, numRentalLocations * 2 + 1}(undef, repeat([carsMax + 1], 2 * numRentalLocations) ..., 2 * moveMax + 1)
    stateTransitionProbs = fill(0.0, repeat([carsMax+1], 2 * numRentalLocations)..., 2 *moveMax + 1)
    for s_old1 = 0:carsMax, s_old2 = 0:carsMax
        for s_new1 = 0:carsMax, s_new2 = 0:carsMax
            for (i_a, a) in enumerate(A)
                movedCars = clamp(a, -minimum([s_old1, carsMax - s_old2]), minimum([s_old2, carsMax - s_old1]))
                stock1 = s_old1 + movedCars
                stock2 = s_old2 - movedCars
                
                stateTransProb = sum(stateRewardTransitionProbs[s_new1+1, s_new2+1, s_old1+1, s_old2+1, :, :, i_a])
                stateTransitionProbs[s_new1+1, s_new2+1, s_old1+1, s_old2+1, i_a] = stateTransProb

                outcomes = (-2*abs(a)):10:(10 * (stock1 + stock2) - 2 * abs(a))
                probs = OrderedDict{Int64, Float64}(k => 0.0 for k in outcomes)
                for req1 in 0:stock1, req2 in 0:stock2
                    reward = 10 * (req1 + req2) - 2 * abs(a)
                    probs[reward] = probs[reward] + stateRewardTransitionProbs[s_new1+1, s_new2+1, s_old1+1, s_old2+1, req1+1, req2+1, i_a] / stateTransProb
                end
                probVector = [v for v in values(probs)]
                rewardDistributions[s_new1+1, s_new2+1, s_old1+1, s_old2+1, i_a] = DiscreteNonParametric(outcomes, probVector)
            end
        end
    end 

    stateMap = Vector{Tuple{Int64, Int64}}(undef, length(S))
    count = 0
    for s_1 = 1:carsMax+1, s_2 = 1:carsMax+1
        count += 1
        stateMap[count] = (s_1, s_2)
    end

    μ = fill(1/length(S), length(S))
    P = reshape(stateTransitionProbs, (carsMax + 1)^2, (carsMax + 1)^2, 2 * moveMax + 1)
    R = reshape(rewardDistributions, (carsMax + 1)^2, (carsMax + 1)^2, 2 * moveMax + 1)
    JacksCarRentalMDP = TabularMDP(P, R, μ, γ)
    return P, JacksCarRentalMDP, stateMap

end
