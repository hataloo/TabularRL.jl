using Distributions, DataStructures

function getJacksCarRentalMDP(γ = 0.9; λ_requests::Tuple{Int64, Int64} = (3, 4), λ_returns::Tuple{Int64,Int64} = (3, 2), carsMax::Int = 20, moveMax::Int = 5, verbose::Bool = false)
    @assert all(λ_requests .>= 0.0) && all(λ_returns .>= 0.0) "Expected requests and returns must be positive, given λ_requests = $λ_requests, λ_returns = $λ_returns"
    
    numRentalLocations = length(λ_requests)
    S = Vector{Int64}(1:((carsMax+1) ^ numRentalLocations))
    A = Vector{Int64}(-moveMax:moveMax)
    #P[s', r' | s, a]
    stateRewardTransitionProbs = fill(0.0, repeat([carsMax+1], 2 * numRentalLocations)..., 2 *moveMax + 1, carsMax+1, carsMax+1)
    
    poissonpdf(x::Int64, λ::Real) =  λ^x * exp(-λ)/factorial(x) 
    censoredpoissonpdf(x::Int64, λ::Real, max::Int64) = begin
        @assert x <= max "Censoredpoissonpdf does not accept arguments above max = $max, given x = $x, λ = $λ"
        @assert x >= 0 "Censoredpoissonpdf does not accept negative arguments for x, given x = $x, max = $max, λ = $λ"
        if x == max 
            return 1 - sum([poissonpdf(i, λ) for i in 0:(max-1)])
        else
            return poissonpdf(x, λ)
        end
    end

    t = @elapsed begin
    @Threads.threads for s_new1 = 0:carsMax
    @Threads.threads for s_new2 = 0:carsMax
    for s_old1 = 0:carsMax 
    for s_old2 = 0:carsMax
            for (i_a, a) in enumerate(A)
                movedCars = clamp(a, -minimum([s_old1, carsMax - s_old2]), minimum([s_old2, carsMax - s_old1]))
                stock1 = s_old1 + movedCars
                stock2 = s_old2 - movedCars
                lower1 = maximum([-s_new1 + stock1, 0])
                lower2 = maximum([-s_new2 + stock2, 0])
                for req1 in lower1:stock1, req2 in lower2:stock2
                    ret1 = req1 + s_new1 - stock1
                    ret2 = req2 + s_new2 - stock2
                    stateRewardTransitionProbs[
                        s_new1+1, s_new2+1, 
                        s_old1+1, s_old2+1,
                        i_a,
                        req1+1, req2+1] = 
                        censoredpoissonpdf(req1, λ_requests[1], stock1) * 
                        censoredpoissonpdf(req2, λ_requests[2], stock2) * 
                        censoredpoissonpdf(ret1, λ_returns[1], carsMax - stock1 + req1) * 
                        censoredpoissonpdf(ret2, λ_returns[2], carsMax - stock2 + req2)
                end
            end
        end 
    end
    end
    end 
    end
    if verbose
        println("Time to build stateRewardTransitionProbs: $t")
        println(maximum(abs.(sum(stateRewardTransitionProbs, dims = [1,2,6,7]) .- 1.0)))
    end
    rewardDistributions = Array{DiscreteNonParametric, numRentalLocations * 2 + 1}(undef, repeat([carsMax + 1], 2 * numRentalLocations) ..., 2 * moveMax + 1)
    stateTransitionProbs = fill(0.0, repeat([carsMax+1], 2 * numRentalLocations)..., 2 *moveMax + 1)
    t = @elapsed begin
    @Threads.threads for s_new1 = 0:carsMax
    @Threads.threads for s_new2 = 0:carsMax
    for s_old1 = 0:carsMax
    for s_old2 = 0:carsMax
            for (i_a, a) in enumerate(A)
                movedCars = clamp(a, -minimum([s_old1, carsMax - s_old2]), minimum([s_old2, carsMax - s_old1]))
                stock1 = s_old1 + movedCars
                stock2 = s_old2 - movedCars
                
                stateTransProb = sum(stateRewardTransitionProbs[s_new1+1, s_new2+1, s_old1+1, s_old2+1, i_a, :, :])
                stateTransitionProbs[s_new1+1, s_new2+1, s_old1+1, s_old2+1, i_a] = stateTransProb

                outcomes = (-2*abs(a)):10:(10 * (stock1 + stock2) - 2 * abs(a))
                probs = OrderedDict{Int64, Float64}(k => 0.0 for k in outcomes)
                lower1 = maximum([-s_new1 + stock1, 0])
                lower2 = maximum([-s_new2 + stock2, 0])
                for req1 in lower1:stock1, req2 in lower2:stock2
                    reward = 10 * (req1 + req2) - 2 * abs(a)
                    probs[reward] = probs[reward] + stateRewardTransitionProbs[s_new1+1, s_new2+1, s_old1+1, s_old2+1, i_a, req1+1, req2+1, ] / stateTransProb
                end
                probVector = [v for v in values(probs)]
                rewardDistributions[s_new1+1, s_new2+1, s_old1+1, s_old2+1, i_a] = DiscreteNonParametric(outcomes, probVector)
            end
        end 
    end
    end
    end 
    end
    if verbose
        println("Time to build reward & stateTransitionsProbs: $t")
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
    return JacksCarRentalMDP, stateMap

end
