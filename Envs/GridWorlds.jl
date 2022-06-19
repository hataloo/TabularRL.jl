function getMovementFunctions(height::Integer, width::Integer)
    getLeftIndex(i,j) = i, maximum([j-1,1])
    getRightIndex(i,j) = i, minimum([j+1,width])
    getUpIndex(i,j) = maximum([i-1,1]), j
    getDownIndex(i,j) = minimum([i+1,height]), j
    return [getLeftIndex, getRightIndex, getUpIndex, getDownIndex]
end
function getWrappingMovementFunctions(height::Integer, width::Integer)
    getLeftIndex(i,j) = i, j > 1 ? j-1 : 1
    getRightIndex(i,j) = i, j < width ? j+1 : width
    getUpIndex(i,j) = i > 1 ? i-1 : 1, j
    getDownIndex(i,j) = i < height ? i+1 : 1, j
    return [getLeftIndex, getRightIndex, getUpIndex, getDownIndex]
end

function slipperyMovementProbabilities(randomDirectionProbability::Float64 = 0.3)
    r = randomDirectionProbability / 3
    getLeftProbabilities(i,j) = [1 - randomDirectionProbability, r, r, r]
    getRightProbabilities(i,j) = [r, 1 - randomDirectionProbability, r, r]
    getUpProbabilities(i,j) = [r, r, 1 - randomDirectionProbability, r]
    getDownProbabilities(i,j) = [r, r, r, 1 - randomDirectionProbability]
    return [getLeftProbabilities, getRightProbabilities, getUpProbabilities, getDownProbabilities]
end

function buildGridWalkTransitionProbabilities(
        height::Integer, width::Integer, 
        getNextIndexFunctions::Tuple{Function,Function,Function,Function}, 
        getMovementProbs::Tuple{Function,Function,Function,Function})
    S = [(i-1)*width + j for i in 1:height, j in 1:width]
    A = 1:4 #left, right, up, down
    #P[h_1, w_1, h_0, w_0, a]
    P = Array{Float64,5}(undef, height, width, height, width, length(A))
    P .= 0.0
    for i in 1:(height), j in 1:width, a in A
        i_1, j_1 = getNextIndexFunctions[a](i,j)
        P[ i_1,j_1, i,j, a] += getMovementProbs[a](i,j)
        for a_prim in A 
            if a_prim != a
                i_1, j_1 = getNextIndexFunctions[a_prim](i,j)
                P[i_1,j_1, i,j, a] += getMovementProbs[a_prim](i,j)
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

function addTerminalState!(P::Array{Float64, 5}, width::Int64, height::Int64)
    P[:,:, width, height, :] .= 0.0
    P[width, height, width, height, :] .= 0.0
end

function addTerminalState!(P::Array{Float64, 5}, width::Vector{Int64}, height::Vector{Int64})
    @assert length(width) == length(height) "Length of width must equal length of height, given lengths; $(length(width)), $(length(height))"
    for (w, h) in zip(width, height)
        addTerminalState!(P, w, h)
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
    @assert maximum(sum(P, dims = [1,2]) .- 1.0) <= 1e-12 "sum(P[:,:, w_0, h_0]) is not equal to one for all w_0, h_0. maximum(sum(P, dims = [1,2]) .- 1.0) = $(maximum(sum(P, dims = [1,2]) .- 1.0))"
end

