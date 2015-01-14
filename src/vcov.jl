

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
	if ver==:weighted
 		gn = mdp.mdnlpe.momf.sᵢ(coef(mdp))
 		V = (getmdweights(mdp).*gn)'*gn
 	elseif ver==:unweighted
 		gn = mdp.mdnlpe.momf.sᵢ(coef(mdp))
 		V = gn'*gn
	end
 	return PDMat(V)
end


## macro forwardrule(x, e)
##     x, e = esc(x), esc(e)
##     quote
##         $e = sqrt(eps(eltype($x))) * max(one(eltype($x)), abs($x))
##     end
## end


function mdobj_hessian(mdp::MinDivProb, θ::Vector)
    n, m, k = size(mdp)    
    λ = zeros(m)
    p = ones(n)    
    g2  = MomentMatrix(Array(Float64, n, m))
    smd = MinDivProb(g2, divergence(mdp), solver = IpoptSolver(linear_solver = "ma27", print_level=0))

    function f(theta)
        @inbounds g2.g[:]  = mdp.mdnlpe.momf.sᵢ(theta)
        solve(smd).model.inner.obj_val
    end 

    Calculus.second_derivative(f, coef(mdp))

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
    mdp.H = PDMat(mdobj_hessian(mdp, coef(mdp)))
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









