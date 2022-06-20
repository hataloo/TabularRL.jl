include("TabularRL.jl")

height, width = 10, 15

P, S, A = buildSlipperyGridTransitionProbabilities(height, width, false, 0.1)
@assert maximum(abs.(sum(P, dims = [1,2]) .- 1.0)) <= 1e-12 "sum(P[:,:, w_0, h_0]) is not equal to one for all w_0, h_0. maximum(sum(P, dims = [1,2]) .- 1.0) = $(maximum(sum(P, dims = [1,2]) .- 1.0))"

R = fill(-0.1, height, width, height, width, length(A))

addTerminalState!(P, [height, height], [1, width], R, Float64[5, 10])
@assert maximum(abs.(sum(P, dims = [1,2]) .- 1.0)) <= 1e-12 "sum(P[:,:, w_0, h_0]) is not equal to one for all w_0, h_0. maximum(sum(P, dims = [1,2]) .- 1.0) = $(maximum(sum(P, dims = [1,2]) .- 1.0))"

resettingStates = [[(i,j) for i in 2:2:(height-1), j in 2:2:(width-1)]...]
startStates = repeat([(1,1)], length(resettingStates))
resetRewards = repeat([-1.0], length(resettingStates))
addResettingState!(P, resettingStates, startStates, R, resetRewards)
@assert maximum(abs.(sum(P, dims = [1,2]) .- 1.0)) <= 1e-12 "sum(P[:,:, w_0, h_0]) is not equal to one for all w_0, h_0. maximum(sum(P, dims = [1,2]) .- 1.0) = $(maximum(sum(P, dims = [1,2]) .- 1.0))"

μ = zeros(length(S))
μ[1] = 1.0
P_flatten = reshape(P, height * width, height * width, 4)
R_flatten = reshape(R, height * width, height * width, 4)
@assert maximum(sum(P_flatten, dims = 1) .- 1.0) <= 1e-12 "sum(P[:,:, w_0, h_0]) is not equal to one for all w_0, h_0. maximum(sum(P, dims = [1,2]) .- 1.0) = $(maximum(sum(P, dims = [1,2]) .- 1.0))"

mdp = TabularMDP(P_flatten, R_flatten, μ, 0.9)
println()