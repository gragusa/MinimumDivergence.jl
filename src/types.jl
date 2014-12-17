abstract SmoothingKernel

type MomentFunction
  gᵢ::Function        ## Moment Function
  sᵢ::Function        ## Smoothed moment function
  wsn::Function       ## (m×1) ∑pᵢ sᵢ
  sn::Function        ## ∑sᵢ(θ)
  ∂∑pᵢsᵢ::Function    ## (k×m)
  ∂∑sᵢ::Function      ## (k×m)
  ∂sᵢλ::Function      ## (n×k)
  ∑pᵢ∂sᵢλ::Function   ## (k×1)
  ∂²sᵢλ::Function     ## (kxk)
  kern::SmoothingKernel
  nobs::Int64
  nmom::Int64
  npar::Int64
end

type MomentMatrix
  g::AbstractMatrix
  g_L::Vector
  g_U::Vector
  kern::SmoothingKernel
  m_eq::Int64
  m_ineq::Int64
end

type MinDivNLPEvaluator <: AbstractNLPEvaluator
  momf::MomentFunction
  div::Divergence
  nobs::Int64
  nmom::Int64
  npar::Int64
  gele::Int64
  hele::Int64
  solver::AbstractMathProgSolver
  lmbd::Array{Float64, 1}
  wght::Array{Float64, 1}
end

type SMinDivNLPEvaluator <: AbstractNLPEvaluator
  mm::MomentMatrix
  div::Divergence
  nobs::Int64
  nmeq::Int64
  nmineq::Int64
  nmom::Int64
  gele::Int64
  hele::Int64
  solver::AbstractMathProgSolver
end

typealias MDNLPE MinDivNLPEvaluator
typealias SMDNLPE SMinDivNLPEvaluator