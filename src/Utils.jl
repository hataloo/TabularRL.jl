import Distributions:convolve

# Added from https://github.com/JuliaStats/Distributions.jl/pull/1523
function convolve(d1::DiscreteNonParametric, d2::DiscreteNonParametric)
    support_conv = collect(Set(s1 + s2 for s1 in support(d1), s2 in support(d2)))
    sort!(support_conv) #for fast index finding below
    probs1 = probs(d1)
    probs2 = probs(d2)
    p_conv = zeros(Base.promote_eltype(probs1, probs2), length(support_conv)) 
    for (s1, p1) in zip(support(d1), probs(d1)), (s2, p2) in zip(support(d2), probs(d2))
            idx = searchsortedfirst(support_conv, s1+s2)
            p_conv[idx] += p1*p2
    end
    return DiscreteNonParametric(support_conv, p_conv,check_args=false) 
end