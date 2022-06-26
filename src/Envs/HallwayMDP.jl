
function getHallwayMDP(N::Integer, γ, startInMiddle::Bool = true)::TabularMDP
    if N < 2 throw(DomainError("N must be greater than 1, given N = $N.")) end
    S = [i for i in 1:N]
    A = [1,2] #left, right
    μ = zeros(N)
    if startInMiddle
        μ[N ÷ 2] = 1.0
    else
        μ .= 1/length(S)
    end
    #μ[Int64(floor(N/2))] = 1
    P = Array{Float64,3}(undef, length(S), length(S), length(A))
    R = Array{Float64,2}(undef, length(S), length(A))
    R .= -1
    R[1, :] .= 0
    R[N, :] .= 0
    R[2, 1] = 5
    R[N-1, 2] = 10
    P .= 0.0
    p_left = 0.9 # Probability of going left when choosing left
    p_right = 0.7 # Probability of going right when choosing right
    for s¹ in S, s⁰ in S, a in A
        if (s¹ - s⁰ == -1) && a == 1 # P(go left, if choosing left)
            P[s¹, s⁰, a] = p_left
        elseif (s¹ - s⁰ == 1) && a == 1 #P(go right, if choosing left) 
            P[s¹, s⁰, a] = 1-p_left
        elseif (s¹ - s⁰ == -1) && a == 2 # P(go left, if choosing right)
            P[s¹, s⁰, a] = 1 - p_right
        elseif (s¹ - s⁰ == 1) && a == 2 #P(go right, if choosing left) 
            P[s¹, s⁰, a] = p_right
        elseif (s¹ == 1) && (s⁰ == 1) && a == 1 #At left-most point, choosing left
            P[s¹, s⁰, a] = p_left
        elseif (s¹ == 1) && (s⁰ == 1) && a == 2 #At left-most point, choosing right
            P[s¹, s⁰, a] = 1 - p_right
        elseif (s¹ == N) && (s⁰ == N) && a == 1 #At right-most point, choosing left
            P[s¹, s⁰, a] = 1 - p_left
        elseif (s¹ == N) && (s⁰ == N) && a == 2 #At right-most point, choosing right
            P[s¹, s⁰, a] = p_right
        end
    end 
    # Make endpoints terminal
    P[:, 1, :] .= 0.0
    P[:, N, :] .= 0.0
    P[1, 1, :] .= 1.0
    P[N, N, :] .= 1.0
    return TabularMDP(S, A, P, R, μ, γ)
    #return S, A, P, R
end