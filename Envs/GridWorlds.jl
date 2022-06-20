function getMovementFunctions(height::Integer, width::Integer)
    getLeftIndex(i,j) = i, maximum([j-1,1])
    getRightIndex(i,j) = i, minimum([j+1,width])
    getUpIndex(i,j) = maximum([i-1,1]), j
    getDownIndex(i,j) = minimum([i+1,height]), j
    return NTuple{4,Function}((getLeftIndex, getRightIndex, getUpIndex, getDownIndex))
end
function getWrappingMovementFunctions(height::Integer, width::Integer)
    getLeftIndex(i,j) = i, j > 1 ? j-1 : 1
    getRightIndex(i,j) = i, j < width ? j+1 : width
    getUpIndex(i,j) = i > 1 ? i-1 : 1, j
    getDownIndex(i,j) = i < height ? i+1 : 1, j
    return NTuple{4,Function}((getLeftIndex, getRightIndex, getUpIndex, getDownIndex))
end

function slipperyMovementProbabilities(randomDirectionProbability::Float64 = 0.3)
    r = randomDirectionProbability / 3
    getLeftProbabilities(i,j) = [1 - randomDirectionProbability, r, r, r]
    getRightProbabilities(i,j) = [r, 1 - randomDirectionProbability, r, r]
    getUpProbabilities(i,j) = [r, r, 1 - randomDirectionProbability, r]
    getDownProbabilities(i,j) = [r, r, r, 1 - randomDirectionProbability]
    return NTuple{4,Function}((getLeftProbabilities, getRightProbabilities, getUpProbabilities, getDownProbabilities))
end

function buildGridWalkTransitionProbabilities(
        height::Integer, width::Integer, 
        getNextIndexFunctions::Union{Tuple{Vararg{Function,4}},
            Tuple{Vararg{Function,5}}}, 
        getMovementProbs::Union{Tuple{Vararg{Function,4}},
            Tuple{Vararg{Function,5}}})
    @assert length(getNextIndexFunctions) == length(getMovementProbs) "Number of indexFunctions and movementProbs functions must be the same, given $(length(getNextIndexFunctions)) and $(length(getMovementProbs))" 
    S = [(i-1)*width + j for i in 1:height, j in 1:width]
    A = 1:length(getNextIndexFunctions) #left, right, up, down, (optionally stay)
    #P[h_1, w_1, h_0, w_0, a]
    P = Array{Float64,5}(undef, height, width, height, width, length(A))
    P .= 0.0
    for i in 1:(height), j in 1:width, a in A
        i_1, j_1 = getNextIndexFunctions[a](i,j)
        P[ i_1,j_1, i,j, a] += getMovementProbs[a](i,j)[a]
        for a_prim in A 
            if a_prim != a
                i_1, j_1 = getNextIndexFunctions[a_prim](i,j)
                P[i_1,j_1, i,j, a] += getMovementProbs[a](i,j)[a_prim]
            end
        end
    end
    @assert all(abs.(sum(P, dims = [1,2]) .- 1) .< 1e-14) "All transition probs does not sum to 1"
    return P, S, A
end

function buildSlipperyGridTransitionProbabilities(height::Integer, width::Integer, wrapsAround::Bool = false, randomDirectionProbability::Float64 = 0.3)
    movementProbs = slipperyMovementProbabilities(randomDirectionProbability)
    if wrapsAround
        return buildGridWalkTransitionProbabilities(height, width,
            getWrappingMovementFunctions(height, width), movementProbs)
    else
        return buildGridWalkTransitionProbabilities(height, width,
            getMovementFunctions(height, width), movementProbs)
    end
end

function addTerminalState!(P::Array{Float64, 5}, height::Int64, width::Int64)
    P[:,:, height, width, :] .= 0.0
    P[height, width, height, width, :] .= 1.0
    @assert all(abs.(sum(P, dims = [1,2]) .- 1) .< 1e-14) "All transition probs does not sum to 1, max-deviation: $(maximum(abs.(sum(P, dims = [1,2]) .- 1)))"
end

function addTerminalState!(P::Array{Float64, 5}, height::Int64, width::Int64, R::Array{T,5}, terminationReward::T) where {T <: Union{Float64, Distribution}}
    terminalIndices = findall(P[height, width, :, :, :] .> 0.0)
    R[height:height, width:width, terminalIndices] .= terminationReward
    P[:,:, height, width, :] .= 0.0
    P[height, width, height, width, :] .= 1.0
    @assert all(abs.(sum(P, dims = [1,2]) .- 1) .< 1e-14) "All transition probs does not sum to 1, max-deviation: $(maximum(abs.(sum(P, dims = [1,2]) .- 1)))"

end

function addTerminalState!(P::Array{Float64, 5}, height::Vector{Int64}, width::Vector{Int64})
    @assert length(height) == length(width) "Length of width must equal length of height, given lengths; $(length(height)), $(length(width))"
    for (w, h) in zip(height, width)
        addTerminalState!(P, w, h)
    end
end

function addTerminalState!(P::Array{Float64, 5}, height::Vector{Int64}, width::Vector{Int64}, R::Array{T,5}, terminationReward::Vector{T}) where {T <: Union{Float64, Distribution}}
    @assert length(height) == length(width) && length(height) == length(terminationReward) "Length of width, height and terminationReward must be equal, given lengths; $(length(height)), $(length(width)) and $(length(terminationReward))"
    for (w, h, r) in zip(height, width, terminationReward)
        addTerminalState!(P, w, h, R, r)
    end
end

"""
Modify P such that every transition to resettingState is redirected to startState.
"""
function addResettingState!(P::Array{Float64,5}, resettingState::Tuple{Int64, Int64}, startState::Tuple{Int64, Int64})
    @assert resettingState != startState "resettingState and startState must differ, given states: $resettingState, $startState"
    widths, heights = 1:size(P,1), 1:size(P,2)
    A = 1:size(P,5)
    for w in widths, h in heights, a in A
        if P[resettingState[1], resettingState[2], h, w, a] > 0.0
            P[startState[1], startState[2], w, h, a] = P[resettingState[1], resettingState[2], w, h, a]
            P[resettingState[1], resettingState[2], h, w, a] = 0.0
        end
    end
    P[:,:, resettingState[1], resettingState[2], :] .= 0.0
    P[startState[1],startState[2], resettingState[1], resettingState[2], :] .= 1.0
    @assert maximum(abs.(sum(P, dims = [1,2]) .- 1.0)) <= 1e-12 "sum(P[:,:, w_0, h_0]) is not equal to one for all w_0, h_0. maximum(sum(P, dims = [1,2]) .- 1.0) = $(maximum(sum(P, dims = [1,2]) .- 1.0))"
end
function addResettingState!(P::Array{Float64,5}, resettingStates::Vector{Tuple{Int64, Int64}}, startStates::Vector{Tuple{Int64, Int64}})
    @assert length(resettingStates) == length(startStates) "Length of resettingStates and startStates must be equal, given lengths; $(length(resettingStates)) and $(length(startStates))"
    for (r, s) in zip(resettingStates, startStates) addResettingState!(P, r, s) end
end
"""
Modify P such that every transition to resettingState is redirected to startState and receives resetReward.
"""
function addResettingState!(P::Array{Float64,5}, resettingState::Tuple{Int64, Int64}, startState::Tuple{Int64, Int64},
        R::Array{T,5}, resetReward::T) where {T <: Union{Float64, Distribution}}
    heights, widths = 1:size(P,1), 1:size(P,2)
    A = 1:size(P,5)
    for h in heights, w in widths, a in A
        if P[resettingState[1], resettingState[2], h, w, a] > 0.0
            P[startState[1], startState[2], h, w, a] += P[resettingState[1], resettingState[2], h, w, a]
            P[resettingState[1], resettingState[2], h, w, a] = 0.0
            R[startState[1], startState[2], h, w, a] = resetReward
        end
    end
    P[:,:, resettingState[1], resettingState[2], :] .= 0.0
    P[startState[1],startState[2], resettingState[1], resettingState[2], :] .= 1.0
    R[startState[1],startState[2], resettingState[1], resettingState[2], :] .= resetReward
    @assert maximum(abs.(sum(P, dims = [1,2]) .- 1.0)) <= 1e-12 "sum(P[:,:, w_0, h_0]) is not equal to one for all w_0, h_0. maximum(abs.(sum(P, dims = [1,2]) .- 1.0)) = $(maximum(abs.(sum(P, dims = [1,2]) .- 1.0))), argmax = $(argmax(abs.(sum(P, dims = [1,2]) .- 1.0)))"
end

function addResettingState!(P::Array{Float64,5}, resettingStates::Vector{Tuple{Int64, Int64}}, startStates::Vector{Tuple{Int64, Int64}},
    R::Array{T,5}, resetRewards::Vector{T}) where {T <: Union{Float64, Distribution}}   
    @assert length(resettingStates) == length(startStates) && length(resettingStates) == length(resetRewards) "Length of resettingStates, startStates, resetRewards must be equal, given lengths; $(length(resettingStates)), $(length(startStates)) and $(length(resetRewards))"
    for (res, start, ret) in zip(resettingStates, startStates, resetRewards) 
        addResettingState!(P, res, start, R, ret) 
    end
end