function initialize(e::SMDNLPE, rf::Vector{Symbol})
    for feat in rf
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

features_available(e::SMDNLPE) = [:Grad, :Jac, :Hess]
eval_f(e::SMDNLPE, u) = Divergences.evaluate(e.div, u[1:e.mm.n])

MathProgBase.isobjlinear(e::SMDNLPE) = false
MathProgBase.isobjquadratic(e::SMDNLPE) = false
MathProgBase.isconstrlinear(e::SMDNLPE, i::Int64) = false


function eval_g(e::SMDNLPE, g, u)
    n, m, m_eq, m_ineq = size(e)
    p   = u[1:n]
    @inbounds StatsBase.wsum!(view(g, 1:m), e.mm.S, p, 1)
    @inbounds g[m+1]  = sum(p)
end

function eval_grad_f(e::SMDNLPE, grad_f, u)
    n, m, m_eq, m_ineq = size(e)
    for j=1:n
        @inbounds grad_f[j] = Divergences.gradient(e.div, u[j])
    end
end

function jac_structure(e::SMDNLPE)
    n, m, m_eq, m_ineq = size(e)
    rows = Array(Int64, e.gele)
    cols = Array(Int64, e.gele)
    for j = 1:m+1, r = 1:n
        @inbounds rows[r+(j-1)*n] = j
        @inbounds cols[r+(j-1)*n] = r
    end
  rows, cols
end

function hesslag_structure(e::SMDNLPE)
    n, m, m_eq, m_ineq = size(e)
    rows = Array(Int64, e.hele)
    cols = Array(Int64, e.hele)
    for j = 1:n
        @inbounds rows[j] = j
        @inbounds cols[j] = j
    end
  rows, cols
end

function eval_jac_g(e::SMDNLPE, J, u)
    n, m, m_eq, m_ineq = size(e)
    for j=1:m+1, i=1:n
        if(j<=m)
            @inbounds J[i+(j-1)*n] = e.mm.S[i+(j-1)*n]
        else
            @inbounds J[i+(j-1)*n] = 1.0
        end
    end
end

function eval_hesslag(e::SMDNLPE, H, u, σ, λ)
    n, m, m_eq, m_ineq = size(e)
    if σ==0
        for j=1:n
            @inbounds H[j] = 0.0
        end
    else
      for j=1:n
          @inbounds H[j] = σ*Divergences.hessian(e.div, u[j])
      end
    end
end
