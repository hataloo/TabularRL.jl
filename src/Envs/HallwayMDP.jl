
function getHallwayMDP(N::Integer, γ, startInMiddle::Bool = true, deterministic::Bool = false)::TabularMDP
    if N < 3 throw(DomainError("N must be greater than 2, given N = $N.")) end
    S = [i for i in 1:N]
    A = [1,2] #left, right
    μ = zeros(N)
    if startInMiddle && isodd(N)
        μ[N ÷ 2 + 1] = 1.0
    elseif startInMiddle && iseven(N)
        μ[N ÷ 2] = 1/2
        μ[N ÷ 2 + 1] = 1/2
    else
        μ[2:(N-1)] .= 1/(length(S)-2)
    end
    #μ[Int64(floor(N/2))] = 1
    P = Array{Float64,3}(undef, length(S), length(S), length(A))
    P .= 0.0
    R = Array{Float64,3}(undef, length(S), length(S), length(A))
    R .= -1
    R[1, :,:] .= 5
    R[N, :,:] .= 10
    if deterministic
        p_left = 1.0
        p_right = 1.0
    else
        p_left = 0.9 # Probability of going left when choosing left
        p_right = 0.7 # Probability of going right when choosing right
    end
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
    return TabularMDP(P, R, μ, γ)
    #return S, A, P, R
end

mutable struct HallwayVisualizationWrapper <: AbstractMDPWrapper{
        DiscreteContiguousSpace,
        DiscreteContiguousSpace,
        Dirac{Float64}, 
        TabularMDP{Dirac{Float64}}}
    MDP::TabularMDP{Dirac{Float64}}
    fig::Figure
    graphPlots::Vector{Any}
    state::Int64
    HallwayVisualizationWrapper(mdp::TabularMDP{Dirac{Float64}}) = begin
        numStates = length(getStates(mdp))
        fig = Figure(resolution = (minimum([160 * numStates, 1800]), 500))
        graphPlots = initHallwayVisualization(fig, mdp)
        newHVW = new(mdp, fig, graphPlots, getCurrentState(mdp))
        reset!(newHVW)
        newHVW
    end
    HallwayVisualizationWrapper(gridPos::GridPosition, mdp::TabularMDP{Dirac{Float64}}) = begin
        fig = gridPos.layout.parent
        graphPlots = initHallwayVisualization(gridPos, mdp)
        newHVW = new(mdp, fig, graphPlots, getCurrentState(mdp))
        reset!(newHVW)
        newHVW
    end
end



function initHallwayVisualization(fig::Union{Figure, GridPosition}, mdp::TabularMDP{Dirac{Float64}})
    numStates = length(getStates(mdp))
    numActions = length(getActions(mdp))
    meanRewards = mean.(getRewardDistributions(mdp))
    edgeLabels(g, a) = [
            "p=" * repr(round(weight(e), digits = 3)) *
            ",r=" *
            repr(round(meanRewards[dst(e), src(e), a], digits = 3))
        for e in edges(g)]
    flatLayout(g) = [Point2f(i, 0.0) for i = 1:nv(g)]
    axes = []
    graphPlots = []
    titles = ["⬅", "➡"]
    for a in 1:numActions
        ax = Axis(fig[a,1], title = "Action: $(titles[a])")
        push!(axes, ax)
        g = SimpleWeightedDiGraph(getTransitionProbabilities(mdp)[:,:,a]')
        graphPlot = graphplot!(ax, g, 
            nlabels_color = repeat([:black], nv(g)),
            node_size = repeat([10.0], nv(g)),
            node_color = repeat([:black], nv(g)),
            nlabels = repr.(1:numStates),
            nlabels_align = (:center, :center),
            nlabels_offset = Point2f(.0, .05),
            elabels = edgeLabels(g, a),
            elabels_textsize = 16,
            elabels_color = repeat([:black], ne(g)),
            edge_color = repeat([:black], ne(g)),
            edge_width = 1.8,
            selfedge_width = 1.2,
            layout = flatLayout
            )
        push!(graphPlots, graphPlot)
        hidedecorations!(ax)
    end
    state = getCurrentState(mdp)
    for (gp, ax) in zip(graphPlots, axes)
        gp[:node_color][][state] = hasTerminated(mdp) ? :red : :blue 
        gp[:node_size][][state] = 20.0
        gp[:nlabels_color][][state] = hasTerminated(mdp) ? :red : :blue 
        notify(gp[:node_color])
        notify(gp[:node_size])
        notify(gp[:nlabels_color])

        gp.elabels_distance[] = 15
        gp.elabels_side[] = Dict(1 => :right, ne(gp.graph[]) => :right)
        gp.selfedge_direction[] = Dict(
            [(i, Point2f(-1.0 + (src(e) > (numStates/2)), 0.0))  
            for (i, e) in enumerate(edges(gp.graph[])) if src(e) == dst(e)])
        autolimits!(ax)
    end
    return graphPlots
end

function updateState(hvw::HallwayVisualizationWrapper, newState::Int64)
    oldState = hvw.state
    for gp in hvw.graphPlots
        newColor = hasTerminated(hvw.MDP) ? :red : :blue
        gp[:node_color][][oldState] = :black
        gp[:node_color][][newState] = newColor
        notify(gp[:node_color])
        gp[:node_size][][oldState] = 10.0
        gp[:node_size][][newState] = 20.0
        notify(gp[:node_size])
        gp[:nlabels_color][][oldState] = :black
        gp[:nlabels_color][][newState] = newColor 
        notify(gp[:nlabels_color])
        for (i, e) in enumerate(edges(gp.graph[]))
            if src(e) == oldState
                gp.edge_color[][i] = :black
            end
            if src(e) == newState
                gp.edge_color[][i] = newColor
            end
        end
        notify(gp.edge_color)
    end
    hvw.state = newState
    return
end

function hasTerminated(hvw::HallwayVisualizationWrapper)
    return hasTerminated(hvw.MDP)
end

function reset(hvw::HallwayVisualizationWrapper)
    s_new = reset(hvw.MDP)
    updateState(hvw, s_new)
    return s_new
end

function reset!(hvw::HallwayVisualizationWrapper)
    s_new = reset(hvw.MDP)
    updateState(hvw, s_new)
end

function step(hvw::HallwayVisualizationWrapper, a::Int64)
    s_new, r, done = step(hvw.MDP, a)
    updateState(hvw, s_new)
    return s_new, r, done
end

struct HallwayVisualController 
    fig::Figure
    hvw::HallwayVisualizationWrapper
    HallwayVisualController(mdp::TabularMDP) = begin
        numStates = length(getStates(mdp))
        width = 160 * numStates
        fig = Figure(resolution = (minimum([width, 1200]), 550))
        hvw = HallwayVisualizationWrapper(fig[1:2, 1], mdp)
        addHVCControls(fig, hvw)
        new(hvw.fig, hvw)
    end
end

function addHVCControls(fig::Figure, hvw::HallwayVisualizationWrapper)
    buttonLabels = ["⬅", "Reset", "➡"] 
    fig[3, 1] = buttonGrid = GridLayout(tellwidth = false)
    buttonGrid[1, 1:3] = buttons = [Button(fig, label = l) for l in buttonLabels]

    for (action, button) in zip([1,2], buttons[[1, 3]])
        on(button.clicks) do n 
            if !hasTerminated(hvw) 
                _, _, done = step(hvw, action)
                if done
                    buttons[1].buttoncolor = RGBf(0.95, 0.1, 0.1)
                    buttons[2].buttoncolor = RGBf(0.1, 0.95, 0.1)
                    buttons[3].buttoncolor = RGBf(0.95, 0.1, 0.1)
                end
            end
        end
    end
    on(buttons[2].clicks) do n
        reset!(hvw)
        buttons[1].buttoncolor = RGBf(0.94, 0.94, 0.94)
        buttons[2].buttoncolor = RGBf(0.94, 0.94, 0.94)
        buttons[3].buttoncolor = RGBf(0.94, 0.94, 0.94)
    end
end
