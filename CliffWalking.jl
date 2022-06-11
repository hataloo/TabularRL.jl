function getCliffWalkingMDP(height::Integer, width::Integer, γ::Float64)
    
    S = [(i-1)*width + j for i in 1:height, j in 1:width]
    A = 1:4 #left, right, up, down
    getLeftIndex(i,j) = i, maximum([j-1,1])
    getRightIndex(i,j) = i, minimum([j+1,width])
    getUpIndex(i,j) = maximum([i-1,1]), j
    getDownIndex(i,j) = minimum([i+1,1]), j
    getNextIndexFunctions = [getLeftIndex, getRightIndex, getUpIndex, getDownIndex]
    P = Array{Float64,5}(undef, height, width, height, width, length(A))
    P .= 0.0
    p_random_direction = 0.1
    for i in 1:(height), j in 1:width, a in A
        i_1, j_1 = getNextIndexFunctions[a](i,j)
        P[ i_1,j_1, i,j, a] += 1.0 - 3 * p_random_direction
        for a_prim in A 
            if a_prim != a
                i_1, j_1 = getNextIndexFunctions[a_prim](i,j)
                P[ i_1,j_1, i,j, a] += p_random_direction
            end
        end
    end
    P[:, :, 1, width, :] .= 0.0
    P[1, width, 1, width, :] .= 1.0
    @assert all(abs.(sum(P, dims = [1,2]) .- 1) .< 1e-14) "All transition probs does not sum to 1"
    return P
end

P = getCliffWalkingMDP(3, 5, 0.9)

maximum(abs.(sum(P, dims = [1,2]) .-1))