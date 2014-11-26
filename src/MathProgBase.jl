import MathProgBase.MathProgSolverInterface

type MinimumDivergenceProblem <: MathProgSolverInterface.AbstractNLPEvaluator
end

typealias MDP MinimumDivergenceProblem

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
MathProgSolverInterface.eval_f(d::MDP, u) = Divergences.evaluate(divergence, u[1:n])

function MathProgSolverInterface.eval_g(d::MDP, g, u)
    θ   = u[(nobs+1):(nobs+npar)]
    p   = u[1:nobs]
    @inbounds g[1:nmom]  = mf.gn(θ)
    @inbounds g[nmom+1]  = sum(p)
end

function MathProgSolverInterface.eval_grad_f(d::MDP, grad_f, u)
    for j=1:nobs
      @inbounds grad_f[j] = Divergences.gradient(divergence, u[j])
    end
    for j=(nobs+1):(n+npar)
      @inbounds grad_f[j] = 0.0
    end
end

function MathProgSolverInterface.jac_structure(d::MDP) 
  rows = Array(Float64, gele)
  cols = Array(Float64, gele)
  for j = 1:nmom+1, r = 1:nobs+npar
        if !((r > n) && (j==m+1))
          @inbounds rows[r+(j-1)*(nobs+npar)] = j
          @inbounds cols[r+(j-1)*(nobs+npar)] = r
        end
      end
  rows, cols
end

function MathProgSolverInterface.hesslag_structure(d::MDP) 
  rows = Array(Float64, hele)
  cols = Array(Float64, hele)
  for j = 1:nobs
    @inbounds rows[j] = j
    @inbounds cols[j] = j
  end
  idx = nobs+1

  for d = 1:nobs
    for j = 1:npar
      @inbounds rows[idx] = nobs+j
      @inbounds cols[idx] = d
      idx += 1
    end
  end

  for j = 1:npar
    for d = 1:j
      @inbounds rows[idx] = nobs+j
      @inbounds cols[idx] = nobs+d
      idx += 1
    end
  end
  rows, cols
end 

function MathProgSolverInterface.eval_jac_g(d::MDP, J, u)  
  __p   = u[1:nobs]
  θ     = u[(nobs+1):(nobs+npar)]
  g     = mf.sᵢ(θ)
  ∂gn    = mf.∂gn(theta)
  
  for j=1:m+1, i=1:nobs+npar
    if(j<=m && i<=nobs)
      @inbounds J[i+(j-1)*(nobs+npar)] = g[i+(j-1)*nobs]
    elseif (j<=m && i>n)
      @inbounds J[i+(j-1)*(nobs+npar)] = ∂gn[j, i-nobs]
    elseif (j>m && i<=n)
      @inbounds J[i+(j-1)*(nobs+npar)] = 1.0
    end
  end
end

function MathProgSolverInterface.eval_hesslag(d::MDP, H, u, σ, λ)
  __p  = u[1:nobs]
  θ    = u[(nobs+1):(nobs+npar)]      
  __λ  = λ[1:nmom]

  ∂g1  = transpose(mf.∂g1(θ))

  if σ==0
    for j=1:nobs
      @inbounds H[j] = 0.0
    end
  else
    for j=1:n
      @inbounds H[j] = σ*Divergences.hessian(divergence, u[j])
    end
  end
  @inbounds H[nobs+1:nobs*npar+nobs] = ∂g1[:]
  @inbounds H[nobs*npar+nobs+1:hele] = gettril(mf.∂²g2(θ))
end


function mdtest(mf::MomentFunction,
                div::Divergence, 
                θ₀::Vector, 
                lb::Vector, ub::Vector;
                solver=IpoptSolver())

    model = MathProgSolverInterface.model(solver)

    npar = length(θ₀)
    nobs, nmom = size(mf.sᵢ(θ₀))

    u₀ = [ones(nobs), θ₀]

    gele::Int64 = int((nobs+npar)*(nmom+1)-npar)
    hele::Int64 = int(nobs*npar + nobs + (npar+1)*npar/2)


    g_L = [zeros(nmom), nobs];
    g_U = [zeros(nmom), nobs];

    u_L = [zeros(nobs),  lb];
    u_U = [ones(nobs)*nobs, ub];
    # l = [1,1,1,1]
    # u = [5,5,5,5]
    # lb = [25, 40]
    # ub = [Inf, 40]
    MathProgSolverInterface.loadnonlinearproblem!(model, nmom+1, npar, 
                                                  g_L, g_U, u_L, u_U, :Min, MDP())
    
    MathProgSolverInterface.setwarmstart!(model, u₀)

    MathProgSolverInterface.optimize!(model)
    stat = MathProgSolverInterface.status(model)

    # @test stat == :Optimal
    uᵒ = MathProgSolverInterface.getsolution(model)
    Qᵒ = MathProgSolverInterface.getobjval(model) 

end




