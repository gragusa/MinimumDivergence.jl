## Calculate variance covariances

momf_jac(md::MinimumDivergenceEstimator) = momf_jac(md::MinDivProb, :weighted)
momf_var(md::MinimumDivergenceEstimator) = momf_var(md::MinDivProb, :weighted)


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


##vcov(md::MDEstimator, version::Symbol)

##ss(theta) = reshape(wsum(mf.sᵢ(theta), pw, 1)', 6)

## This returns Gn(θ₀) with optimal weights
## which are taken from MinimumDivergence.__p
## this is however subject to change....
## Return a (m×k) matrix
## TODO: Should I consider the case in which <(>) present
function momf_jac(md::MDEstimator, ver::Symbol)
    if ver==:weighted
        md.e.momf.Dws(coef(md), getmdweights(md))
    elseif ver==:unweighted
        md.e.momf.Dsn(coef(md))
    ## elseif ver==:unsmoothed
    ##     md.e.momf.Dgn(coef(md))
    ## elseif ver==:unweighted_unsmoothed
    ##     md.e.momf.Dg(coef(md))
    end
end

function momf_var(md::MDEstimator, ver::Symbol)
    sn = md.e.momf.s(coef(md))
    if ver==:weighted     
        V = (getmdweights(md).*sn)'*sn
    elseif ver==:unweighted
        V = sn'*sn
    end
end

function hessian(md::MDEstimator, θ::Vector)
    n, m, k = size(md)    
    λ = zeros(m)
    p = ones(n)    
#   g = MomentMatrix(Array(Float64, n, m), IdentitySmoother())
    r = MDProblem(Array(Float64, n, m), zeros(m),
                        div = divergence(md), solver = md.e.solver)
    function f(theta)
        @inbounds r.e.mm.S[:] = md.e.momf.s(theta)
        getobjval(solve(r))/2.0
    end
    hessian(f, θ)
end

function hessian!(md::MDEstimator)
    if is(md.H, Nothing())
        md.H = hessian(md::MDEstimator, coef(md))
    end
    return md.H
end

stderr(md::MDEstimator) = sqrt(diag(vcov(md, :weighted)))
stderr(md::MDEstimator, ver::Symbol) = sqrt(diag(vcov(md, ver)))

function vcov(md::MDEstimator, ver::Symbol)
    if ver==:hessian
        Ω = momf_var(md, :weighted)
        G = momf_jac(md, :weighted)
        V = G'pinv(Ω)*G
        Hinv = inv(hessian!(md))
        Hinv'*V*Hinv
    else
        Ω = momf_var(md, ver)
        G = momf_jac(md, ver)
        pinv(G'pinv(Ω)*G)
    end
end 
   
vcov(md::MDEstimator) = vcov(md, :weighted)
vcov!(md::MDEstimator) = vcov!(md, :weighted)









