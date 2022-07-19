abstract type AbstractMDPController{MDPType <: AbstractMDP, S} end

function getMDP(mdpController::AbstractMDPController) 
    throw(NotImplementedError("getMDP not implemented for $(typeof(mdpController))")) 
end
function getState(mdpController::AbstractMDPController) 
    throw(NotImplementedError("getState not implemented for $(typeof(mdpController))")) 
end
function hasTerminated(mdpController::AbstractMDPController) 
    throw(NotImplementedError("hasTerminated not implemented for $(typeof(mdpController))")) 
end



mutable struct MDPController{MDPType <: AbstractMDP, S} <: AbstractMDPController{MDPType, S}
    mdp::MDPType
    state::S
    terminated::Bool
    MDPController(mdp::AbstractMDP, S::Type) = 
        begin 
            startState = sampleInitialState(mdp)
            if !(typeof(startState) <: S)
                throw(ArgumentError("Type of initial state does not match argument S, $(typeof(startState)) <: $S: $(typeof(startState) <: S)"))
            else
                new{AbstractMDP, S}(mdp, sampleInitialState(mdp), false)
            end
        end
end

function getMDP(mdpController::MDPController) return mdpController.mdp end
function getState(mdpController::MDPController) return mdpController.state end
function hasTerminated(mdpController::MDPController) return mdpController.terminated end 

function reset(mdpController::MDPController)
    mdp = getMDP(mdpController)
    mdpController.state = s = sampleInitialState(mdp)
    return s
end

function reset!(mdpController::MDPController)
    mdp = getMDP(mdpController)
    mdpController.state = sampleInitialState(mdp)
end

function step(mdpController::MDPController{M, S}, a::A)::Tuple{S, A, Bool} where 
        {M<:AbstractMDP, S, A}
    mdp = getMDP(mdpController)
    mdpController.state, r, mdpController.terminated = step(mdp, getState(mdpController), a)
    return getState(mdpController), r, hasTerminated(mdpController)
end