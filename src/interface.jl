function initialize(d::MDNLPE, rf::Vector{Symbol})
    for feat in rf
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

features_available(d::MDNLPE) = [:Grad, :Jac, :Hess]
eval_f(d::MDNLPE, u) = Divergences.evaluate(d.div, u[1:d.nobs])

function eval_g(d::MDNLPE, g, u)
  n = d.nobs
  k = d.npar
  m = d.nmom
  θ   = u[(n+1):(n+k)]
  p   = u[1:n]
  @inbounds g[1:m]  = d.momf.wsn(θ, p)
  @inbounds g[m+1]  = sum(p)
end

function eval_grad_f(d::MDNLPE, grad_f, u)
  n = d.nobs
  k = d.npar
  m = d.nmom

  for j=1:n
    @inbounds grad_f[j] = Divergences.gradient(d.div, u[j])
  end
  for j=(n+1):(n+k)
    @inbounds grad_f[j] = 0.0
  end
end

function jac_structure(d::MDNLPE)
  n = d.nobs
  k = d.npar
  m = d.nmom

  rows = Array(Int64, d.gele)
  cols = Array(Int64, d.gele)
  for j = 1:m+1, r = 1:n+k
        if !((r > n) && (j==m+1))
          @inbounds rows[r+(j-1)*(n+k)] = j
          @inbounds cols[r+(j-1)*(n+k)] = r
        end
      end
  rows, cols
end

function hesslag_structure(d::MDNLPE)
  n = d.nobs
  k = d.npar
  m = d.nmom

  rows = Array(Int64, d.hele)
  cols = Array(Int64, d.hele)
  for j = 1:n
    @inbounds rows[j] = j
    @inbounds cols[j] = j
  end
  idx = n+1

  for s = 1:n
    for j = 1:k
      @inbounds rows[idx] = n+j
      @inbounds cols[idx] = s
      idx += 1
    end
  end

  for j = 1:k
    for s = 1:j
      @inbounds rows[idx] = n+j
      @inbounds cols[idx] = n+s
      idx += 1
    end
  end
  rows, cols
end

function eval_jac_g(d::MDNLPE, J, u)
 n = d.nobs
 k = d.npar
 m = d.nmom

 global __p    = u[1:n]
 θ      = u[(n+1):(n+k)]
 g      = d.momf.sᵢ(θ)
 ∂∑pᵢsᵢ = d.momf.∂∑pᵢsᵢ(θ)

 for j=1:m+1, i=1:n+k
  if(j<=m && i<=n)
    @inbounds J[i+(j-1)*(n+k)] = g[i+(j-1)*n]
  elseif (j<=m && i>n)
    @inbounds J[i+(j-1)*(n+k)] = ∂∑pᵢsᵢ[j, i-n]
  elseif (j>m && i<=n)
    @inbounds J[i+(j-1)*(n+k)] = 1.0
  end
 end
end

function eval_hesslag(d::MDNLPE, H, u, σ, λ)
  n = d.nobs
  k = d.npar
  m = d.nmom

  global __p  = u[1:n]
  global __λ  = λ[1:m]
  θ           = u[(n+1):(n+k)]

  ∂sᵢλ = transpose(d.momf.∂sᵢλ(θ))

  if σ==0
    for j=1:n
      @inbounds H[j] = 0.0
    end
  else
    for j=1:n
      @inbounds H[j] = σ*Divergences.hessian(d.div, u[j])
    end
  end
  @inbounds H[n+1:n*k+n] = ∂sᵢλ[:]
  @inbounds H[n*k+n+1:d.hele] = gettril(d.momf.∂²sᵢλ(θ))
end
