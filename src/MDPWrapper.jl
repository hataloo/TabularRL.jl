abstract type AbstractMDPWrapper{
    S_type, S_space <: Space, 
    A_type, A_space <: Space, 
    R_dist <: Distribution,
    MDP_type <: AbstractMDP
    } <: AbstractMDP{
        S_type, S_space, 
        A_type, A_space, 
        R_dist} 
end

struct VerboseWrapper{
    S_type, S_space <: Space, 
    A_type, A_space <: Space, 
    R_dist <: Distribution,
    MDP_type <: AbstractMDP
    } <: AbstractMDPWrapper{
            S_type, S_space, 
            A_type, A_space, 
            R_dist, MDP_type}
    MDP::MDP_type
    VerboseWrapper(mdp::AbstractMDP{
        S_type, S_space, 
        A_type, A_space, 
        R_dist}) where {
            S_type, S_space <: Space,
            A_type, A_space <: Space,
            R_dist} = 
        begin
            new{S_type, S_space, A_type, A_space, R_dist,
                AbstractMDP{
                    S_type, S_space, 
                    A_type, A_space, 
                    R_dist}}(mdp)
        end 
end

function hasTerminated(mdpWrapper::VerboseWrapper)
    return hasTerminated(mdpWrapper.MDP)
end

function reset(mdpWrapper::VerboseWrapper)
    s_new = reset(mdpWrapper.MDP)
    println("Initial state: s_new")
    return s_new
end

function reset!(mdpWrapper::VerboseWrapper)
    s_new = reset(mdpWrapper.MDP)
    println("Initial state: $s_new")
end

function step(mdpWrapper::VerboseWrapper{
    S_type, S_space,
    A_type, A_space, R_dist}, a::A_type) where 
    {S_type, S_space <: Space, A_type, A_space <: Space, R_dist <: Distribution} 
    s_new, r, done =  step(mdpWrapper.MDP, a)
    println("Action: $a, State: $s_new, Reward: $r, Done: $done")
    return s_new, r, done
end
