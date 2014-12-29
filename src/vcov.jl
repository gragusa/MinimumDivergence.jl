

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


macro forwardrule(x, e)
    x, e = esc(x), esc(e)
    quote
        $e = sqrt(eps(eltype($x))) * max(one(eltype($x)), abs($x))
    end
end


function mdobj_hessian(mdp::MinDivProb, θ::Vector)
	# 1. Initialize SMinDivProb
	n, m, k = size(mdp)

	λ = zeros(m)
	p = ones(n)

    g  = MomentMatrix(Array(Float64, n, m))
	smd = MinDivProb(g, divergence(mdp),
					solver = IpoptSolver(print_level=0, linear_solver = "ma27"))

	h(θ, λ, p) = (p'*mdp.mdnlpe.momf.sᵢ(θ)*λ)[1]
	h_closure!(θ, gg) = gg[:] = h(θ, λ, p)
	∂h = ForwardDiff.forwarddiff_jacobian(h_closure!, Float64, fadtype=:dual, n = k, m = 1)

	H = zeros(k, k)

    for i = 1:k
      @forwardrule θ[i] epsilon
            oldx = θ[i]
            θ[i] = oldx + epsilon
            @inbounds g.g[:]  = mdp.mdnlpe.momf.sᵢ(θ)
			solve(smd)
			λ = getlambda(smd)
			p = getmdweights(smd)
            f_xplusdx = ∂h(θ)
            θ[i] = oldx
            @inbounds H[i, :] = f_xplusdx / epsilon
    end
    return H
end

function getobjhess!(mdp::MinDivProb)
	mdp.H = PDMat(mdobj_hessian(mdp, coef(mdp)))
	return mdp.H
end

getobjhess(mdp::MinDivProb)  = mdp.H

stderr(mdp::MinDivProb) = sqrt(diag(mdp.Vʷ))
stderr(mdp::MinDivProb, ver::Symbol) = ver==:hessian ? sqrt(diag(mdp.Vᴴ)) : sqrt(diag(mdp.Vᵂ))

function vcov!(mdp::MinDivProb, ver::Symbol)
	if ver==:weighted || (ver==:hessian && typeof(mdp.Vʷ) <: Nothing)
		Ω = momf_var(mdp)
		G = momf_jac(mdp)
		mdp.Vʷ = inv(PDMat(Xt_invA_X(Ω, G)))
	end

	if ver==:hessian
		if typeof(mdp.H) <: Nothing
			getobjhess!(mdp)
		end
		mdp.Vᴴ = PDMat(Xt_invA_X(mdp.Vʷ, full(inv(mdp.H))))
		mdp.Vᴴ
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









