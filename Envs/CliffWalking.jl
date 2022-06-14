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

function buildGridWalkTransitionProbabilities(height::Integer, width::Integer, randomDirectionProbability::Float64 = 0.3, wrapsAround::Bool = false)
    S = [(i-1)*width + j for i in 1:height, j in 1:width]
    A = 1:4 #left, right, up, down
    if wrapsAround
        getNextIndexFunctions = getWrappingMovementFunctions(height, width)        
    else
        getNextIndexFunctions = getMovementFunctions(height, width)
    end
    #P[h_1, w_1, h_0, w_0, a]
    P = Array{Float64,5}(undef, height, width, height, width, length(A))
    P .= 0.0
    for i in 1:(height), j in 1:width, a in A
        i_1, j_1 = getNextIndexFunctions[a](i,j)
        P[ i_1,j_1, i,j, a] += 1.0 - randomDirectionProbability
        for a_prim in A 
            if a_prim != a
                i_1, j_1 = getNextIndexFunctions[a_prim](i,j)
                P[ i_1,j_1, i,j, a] += randomDirectionProbability / (length(A) - 1)
            end
        end
    end
    @assert all(abs.(sum(P, dims = [1,2]) .- 1) .< 1e-14) "All transition probs does not sum to 1"
    return P, S, A
end

function getCliffWalkingMDP(height::Integer, width::Integer, γ::Float64, 
        randomDirectionProbability::Float64 = 0.3, wrapsAround::Bool = false)
    left, right, up, down = 1:4
    P, S, A = buildGridWalkTransitionProbabilities(height, width, randomDirectionProbability, wrapsAround)
    R = fill(-1.0, height, width, height, width, length(A))
    # If transition to cliff, ie move from (height-1:height, 2:width-1) to (height, 1),
    # get reward of -100.
    R[height, 1, height-1:height, 2:width-1, :] .= -100
    R[height, width, height, width, :] .= 0
    #Adjust transitions around cliff
    #P[h_1, w_1, h_0, w_0, a]
    P[:, :, height-1, 2:(width-1), :] .= 0.0
    P[:, :, height, 2:(width-1), :] .= 0.0
    P[height, 1, height, 2:(width-1), :] .= 1.0
    getLeftIndex(i,j) = i, maximum([j-1,1])
    getRightIndex(i,j) = i, minimum([j+1,width])
    getUpIndex(i,j) = maximum([i-1,1]), j
    getDownIndex(i,j) = height, 1
    getNextIndexFunctions = [getLeftIndex, getRightIndex, getUpIndex, getDownIndex]
    for w_0 in 2:(width-1), a in A
        i_1, j_1 = getNextIndexFunctions[a](height-1,w_0)
        P[ i_1,j_1, height-1,w_0, a] += 1.0 - randomDirectionProbability
        for a_prim in A 
            if a_prim != a
                i_1, j_1 = getNextIndexFunctions[a_prim](height-1,w_0)
                P[ i_1,j_1, height-1,w_0, a] += randomDirectionProbability / (length(A) - 1)
            end
        end
    end
    # Add goal
    P[:, :, height, width, :] .= 0.0
    P[height, width, height, width, :] .= 1.0
    μ = zeros(height, width); μ[height, 1] = 1.0
    P = reshape(P, height * width, height * width, 4)
    R = reshape(R, height * width, height * width, 4)
    μ = reshape(μ, height * width)
    cliffWalkingMDP = TabularMDP(P, R, μ, γ)

    return cliffWalkingMDP
end    
