

## Calculate variance covariances

momf_jac(mdp::MinDivProb) = momf_jac(mdp::MinDivProb, :weighted)
momf_var(mdp::MinDivProb) = momf_var(mdp::MinDivProb, :weighted)


## momf_grad -> G
## momf_var -> Ω

## V = (G'Ω⁻¹G)⁻¹
##
## G = ∑ ∂gᵢ(θ₀)/∂θ₀/n
## G = ∑ pᵢ ∂gᵢ(θ₀)/∂θ₀/n
## if smoothed
## G = c×∑ ∂g̃ᵢ(θ₀)/∂θ₀/n
## G = c×∑ pᵢ ∂g̃ᵢ(θ₀)/∂θ₀/n

## Ω = another 4 versions

## Sandwhich
## H⁻¹(G'Ω⁻¹G)H⁻¹


##vcov(mdp::MinDivProb, version::Symbol)

##ss(theta) = reshape(wsum(mf.sᵢ(theta), pw, 1)', 6)

## This returns Gn(θ₀) with optimal weights
## which are taken from MinimumDivergence.__p
## this is however subject to change....
## Return a (m×k) matrix
## TODO: Should I consider the case in which <(>) present
function momf_jac(mdp::MinDivProb, ver::Symbol)
    if ver==:weighted
        mdp.mdnlpe.momf.∂∑pᵢsᵢ(coef(mdp))
    elseif ver==:unweighted
        mdp.mdnlpe.momf.∂∑sᵢ(coef(mdp))
    elseif ver==:unsmoothed
        mdp.mdnlpe.momf.∂∑pᵢgᵢ(coef(mdp))
    elseif ver==:unweightedUnsmoothed
        mdp.mdnlpe.momf.∂∑gᵢ(coef(mdp))
    end
end

function momf_var(mdp::MinDivProb, ver::Symbol)
    gn = mdp.mdnlpe.momf.sᵢ(coef(mdp))
    if ver==:weighted     
        try
            V = (getmdweights(mdp).*gn)'*gn
            PDMat(V)
        catch
            V = gn'*gn
            PDMat(V)
        end 
    elseif ver==:unweighted
        PDMat(gn'*gn)
    end
end


## macro forwardrule(x, e)
##     x, e = esc(x), esc(e)
##     quote
##         $e = sqrt(eps(eltype($x))) * max(one(eltype($x)), abs($x))
##     end
## end

## macro centralrule(x, e)
##     x, e = esc(x), esc(e)
##     quote
##         $e = cbrt(eps(eltype($x))) * max(one(eltype($x)), abs($x))
##     end
## end

## macro hessianrule(x, e)
##     x, e = esc(x), esc(e)
##     quote
##         $e = eps(eltype($x))^(1/4) * max(one(eltype($x)), abs($x))
##     end
## end

## function md_finite_difference_hessian!{S <: Number,
##                                        T <: Number}(f::Function,
##                                                     f0::S,
##                                                     x::Vector{S},
##                                                     H::Array{T})
##     # What is the dimension of x?
##     n = length(x)

##     epsilon = NaN
##     # TODO: Remove all these copies
##     xpp, xpm, xmp, xmm = copy(x), copy(x), copy(x), copy(x)
##     for i = 1:n
##         xi = x[i]
##         @hessianrule x[i] epsilon
##         xpp[i], xmm[i] = xi + epsilon, xi - epsilon
##         H[i, i] = (f(xpp) - 2*f0 + f(xmm)) / epsilon^2
##         @centralrule x[i] epsiloni
##         xp = xi + epsiloni
##         xm = xi - epsiloni
##         xpp[i], xpm[i], xmp[i], xmm[i] = xp, xp, xm, xm
##         for j = i+1:n
##             xj = x[j]
##             @centralrule x[j] epsilonj
##             xp = xj + epsilonj
##             xm = xj - epsilonj
##             xpp[j], xpm[j], xmp[j], xmm[j] = xp, xm, xp, xm
##             H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm))/(4*epsiloni*epsilonj)
##             xpp[j], xpm[j], xmp[j], xmm[j] = xj, xj, xj, xj
##         end
##         xpp[i], xpm[i], xmp[i], xmm[i] = xi, xi, xi, xi
##     end
##     Base.LinAlg.copytri!(H,'U')
## end

## function md_finite_difference_hessian{T <: Number}(f::Function,
##                                                    f0::T,
##                                                    x::Vector{T})
##     # What is the dimension of x?
##     n = length(x)

##     # Allocate an empty Hessian
##     H = Array(Float64, n, n)

##     # Mutate the allocated Hessian
##     md_finite_difference_hessian!(f, f0, x, H)

##     # Return the Hessian
##     return H
## end

function mdobj_hessian(mdp::MinDivProb)
    mdobj_hessian(mdp, coef(mdp))
end

function mdobj_hessian(mdp::MinDivProb, θ::Vector)
    n, m, k = size(mdp)    
    λ = zeros(m)
    p  = ones(n)    
    g  = MomentMatrix(Array(Float64, n, m))
    smd = MinDivProb(g, divergence(mdp), solver = mdp.mdnlpe.solver)
    function f(theta)
        @inbounds g.g[:] = mdp.mdnlpe.momf.sᵢ(theta)
        getobjval(solve(smd))/2.0
    end
    Calculus.finite_difference_hessian(f, θ)
end 

## function mdobj_hessian(mdp::MinDivProb, θ::Vector)
##     # 1. Initialize SMinDivProb
##     n, m, k = size(mdp)    
##     λ = zeros(m)
##     p = ones(n)    
##     g  = MomentMatrix(Array(Float64, n, m))
##     smd = MinDivProb(g, divergence(mdp), solver = mdp.mdnlpe.solver)
    
##     h(θ, λ, p) = (p'*mdp.mdnlpe.momf.sᵢ(θ)*λ)[1]
##     h_closure!(θ, gg) = @inbounds gg[:] = h(θ, λ, p)
##     ∂h = ForwardDiff.forwarddiff_jacobian(h_closure!, Float64, fadtype=:dual, n = k, m = 1)
    
##     H = zeros(k, k)
    
##     for i = 1:k
##         @forwardrule θ[i] epsilon
##         @inbounds oldx = θ[i]
##         @inbounds θ[i] = oldx + epsilon
##         @inbounds g.g[:]  = mdp.mdnlpe.momf.sᵢ(θ)
##         solve(smd)
##         λ = getlambda(smd)
##         p = getmdweights(smd)
##         f_xplusdx = ∂h(θ)
##         @inbounds θ[i] = oldx
##         @inbounds H[i, :] = f_xplusdx / epsilon
##     end
##     return H
## end

function getobjhess!(mdp::MinDivProb)
    mdp.H = try
        PDMat(mdobj_hessian(mdp, coef(mdp)))
    catch
        inv(vcov(mdp, :weighted))
    end            
    return mdp.H
end

getobjhess(mdp::MinDivProb)  = mdp.H

stderr(mdp::MinDivProb) = sqrt(diag(mdp.Vʷ))
stderr(mdp::MinDivProb, ver::Symbol) = ver==:hessian ? sqrt(diag(mdp.Vᴴ)) : sqrt(diag(mdp.Vᵂ))

function vcov!(mdp::MinDivProb, ver::Symbol)
    if ver==:weighted 
        Ω = momf_var(mdp)
        G = momf_jac(mdp)
        mdp.Vʷ = inv(PDMat(Xt_invA_X(Ω, G)))
        return(mdp.Vʷ)
    end

    if ver==:hessian        
        getobjhess!(mdp)
        Ω = momf_var(mdp)
        G = momf_jac(mdp)
        V = PDMat(Xt_invA_X(Ω, G))
        mdp.Vᴴ = PDMat(Xt_A_X(V, full(inv(mdp.H))))
        return(mdp.Vᴴ)
    end
end

function vcov(mdp::MinDivProb, ver::Symbol)
    if typeof(mdp.Vʷ) <: Nothing
        vcov!(mdp, :weighted)
        mdp.Vʷ
    else
        if ver==:weighted
            return mdp.Vʷ
        elseif ver==:hessian
            if typeof(mdp.H) <: Nothing
                getobjhess!(mdp)
            end
            PDMat(Xt_invA_X(mdp.Vʷ, full(inv(mdp.H))))
        end
    end
end

vcov(mdp::MinDivProb) = vcov(mdp, :weighted)
vcov!(mdp::MinDivProb) = vcov!(mdp, :weighted)









