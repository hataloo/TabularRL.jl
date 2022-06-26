function getFrozenLakeMDP(layout::Union{Vector{String},String} = "4x4", slipProbability::Float64 = 0.3; γ::Float64)
    if (slipProbability > 1.0 || slipProbability < 0.0) 
        DomainError("slipProbability must be in the range [0, 1] but was given $slipProbability")
    end
    predefinedLayouts = Dict{String, Vector{String}}(
    "4x4" => [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
        ],
    "8x8" => [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ]
    )
    
    if isa(layout, String)
        try layout = predefinedLayouts[layout] 
        catch e
            throw(KeyError("Layout name $layout not found in predefinedLayouts, available keys are: $(keys(predefinedLayouts)). $e"))
        end
    end
    P, _, _, R, μ = buildFrozenLakeFromLayout(layout, slipProbability)
    mdp = buildGridWorldTabularMDP(P, R, μ, γ)
    return mdp
end

function buildFrozenLakeFromLayout(layout::Vector{String}, slipProbability::Float64)
    height = length(layout)
    width = length(layout[1])
    @assert all(length.(layout) .== width) "All entries in layout must have equal length, given length.(layout) == $(length.(layout))" 
    P, S, A = buildSlipperyGridTransitionProbabilities(height, width, false, slipProbability)
    R = fill(0.0, height, width, height, width, length(A))
    startCoords = findCharacterCoordinateInLayout(layout, 'S')
    goalCoords = findCharacterCoordinateInLayout(layout, 'G')

    addTerminalState!(P, goalCoords[1], goalCoords[2], R, 1.0)

    holeCoords = findAllCharacterCoordinatesInLayout(layout, 'H')
    startCoordsVector = repeat([startCoords], length(holeCoords))
    resetRewards = repeat([0.0], length(holeCoords))
    addResettingState!(P, holeCoords, startCoordsVector, R, resetRewards)
    μ = fill(0.0, height, width)
    μ[startCoords[1], startCoords[2]] = 1.0
    return P, S, A, R, μ
end 