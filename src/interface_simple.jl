function initialize(d::SMDNLPE, rf::Vector{Symbol})
    for feat in rf
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

features_available(d::SMDNLPE) = [:Grad, :Jac, :Hess]
eval_f(d::SMDNLPE, u) = Divergences.evaluate(d.div, u[1:d.nobs])

MathProgBase.isobjlinear(d::SMDNLPE) = false
MathProgBase.isobjquadratic(d::SMDNLPE) = false
MathProgBase.isconstrlinear(d::SMDNLPE, i::Int64) = false


function eval_g(d::SMDNLPE, g, u)
    n = d.nobs
    m = d.nmom
    p   = u[1:n]
    @inbounds StatsBase.wsum!(view(g, 1:m), d.mm.g, p, 1)
    @inbounds g[m+1]  = sum(p)
end

function eval_grad_f(d::SMDNLPE, grad_f, u)
    n = d.nobs
    m = d.nmom
    for j=1:n
        @inbounds grad_f[j] = Divergences.gradient(d.div, u[j])
    end
end

function jac_structure(d::SMDNLPE)
    n = d.nobs
    m = d.nmom
    rows = Array(Int64, d.gele)
    cols = Array(Int64, d.gele)
    for j = 1:m+1, r = 1:n
        @inbounds rows[r+(j-1)*n] = j
        @inbounds cols[r+(j-1)*n] = r
    end
  rows, cols
end

function hesslag_structure(d::SMDNLPE)
    n = d.nobs
    m = d.nmom
    rows = Array(Int64, d.hele)
    cols = Array(Int64, d.hele)
    for j = 1:n
        @inbounds rows[j] = j
        @inbounds cols[j] = j
    end
  rows, cols
end

function eval_jac_g(d::SMDNLPE, J, u)
    n = d.nobs
    m = d.nmom
    for j=1:m+1, i=1:n
        if(j<=m)
            @inbounds J[i+(j-1)*n] = d.mm.g[i+(j-1)*n]
        else
    @inbounds J[i+(j-1)*n] = 1.0
        end
    end
end

function eval_hesslag(d::SMDNLPE, H, u, σ, λ)
    n = d.nobs
    m = d.nmom    
    if σ==0
        for j=1:n
            @inbounds H[j] = 0.0
        end
    else
      for j=1:n
          @inbounds H[j] = σ*Divergences.hessian(d.div, u[j])
      end
    end
end
