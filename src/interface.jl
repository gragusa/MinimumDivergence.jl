function MathProgSolverInterface.initialize(d::MDP, rf::Vector{Symbol})
    for feat in rf
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
            # TODO: implement Jac-vec and Hess-vec products
            # for solvers that need them
        end
    end
end

MathProgSolverInterface.features_available(d::MDP) = [:Grad, :Jac, :Hess]
MathProgSolverInterface.eval_f(d::MDP, u) = Divergences.evaluate(d.div, u[1:d.nobs])

function MathProgSolverInterface.eval_g(d::MDP, g, u)
  n = d.nobs
  k = d.npar
  m = d.nmom
  θ   = u[(n+1):(n+k)]
  p   = u[1:n]
  @inbounds g[1:m]  = d.momf.sn(θ, p)
  @inbounds g[m+1]  = sum(p)
end

function MathProgSolverInterface.eval_grad_f(d::MDP, grad_f, u)
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

function MathProgSolverInterface.jac_structure(d::MDP)   
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

function MathProgSolverInterface.hesslag_structure(d::MDP) 
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

function MathProgSolverInterface.eval_jac_g(d::MDP, J, u)  
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

function MathProgSolverInterface.eval_hesslag(d::MDP, H, u, σ, λ)
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


function mdtest(mf::MomentFunction,
                div::Divergence, 
                θ₀::Vector, 
                lb::Vector, ub::Vector;
                solver=IpoptSolver())

    model = MathProgSolverInterface.model(solver)

    m = mf.nmom
    n = mf.nobs 
    k = mf.npar

    u₀ = [ones(n), θ₀]

    gele = int((n+k)*(m+1)-k)
    hele = int(n*k + n + (k+1)*k/2)

    # lb = [-2., -2]
    # ub = [2, 2.]
    g_L = [zeros(m), n];
    g_U = [zeros(m), n];

    u_L = [zeros(n),  lb];
    u_U = [ones(n)*n, ub];
    # l = [1,1,1,1]
    # u = [5,5,5,5]
    # lb = [25, 40]
    # ub = [Inf, 40]
    MdProb = MinimumDivergenceProblem(mf, div, n, m, k, gele, hele, Array(Float64, n), Array(Float64, m+1)) 
    

    MathProgSolverInterface.loadnonlinearproblem!(model, n+k, m+1, u_L, u_U, g_L, g_U, :Min, MdProb)
    
    MathProgSolverInterface.setwarmstart!(model, u₀)

    MathProgSolverInterface.optimize!(model)
    stat = MathProgSolverInterface.status(model)

    # @test stat == :Optimal
    uᵒ = MathProgSolverInterface.getsolution(model)
    Qᵒ = MathProgSolverInterface.getobjval(model) 
    (MathProgSolverInterface, uᵒ, Qᵒ)

end






