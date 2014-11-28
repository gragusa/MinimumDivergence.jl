abstract SmoothingKernel

type MomentFunction
  gᵢ::Function      ## Moment Function
  sᵢ::Function      ## Smoothed moment function
  sn::Function      ## (m×1) ∑pᵢ sᵢ
  ∂∑pᵢsᵢ::Function  ## (k×m)
  ∂sᵢλ::Function    ## (n×k) 
  ∂²sᵢλ::Function   ## (kxk)
  kern::SmoothingKernel
  nobs::Int64
  nmom::Int64
  npar::Int64
end  

type MinimumDivergenceProblem <: MathProgSolverInterface.AbstractNLPEvaluator
  momf::MomentFunction
  div::Divergence
  nobs::Int64
  nmom::Int64
  npar::Int64  
  gele::Int64
  hele::Int64
  lmbd::Array{Float64, 1}
  wght::Array{Float64, 1}
end

typealias MDP MinimumDivergenceProblem

type MinimumDivergenceProblem <: MathProgSolverInterface.AbstractNLPEvaluator
  momf::MomentFunction
  div::Divergence
  nobs::Int64
  nmom::Int64
  npar::Int64  
  gele::Int64
  hele::Int64
  lmbd::Array{Float64, 1}
  wght::Array{Float64, 1}
end


