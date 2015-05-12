abstract SmoothingKernel

type MomentFunction
    g::Function            ## Moment Function
    s::Function            ## Smoothed moment function
    ws::Function           ## (m×1) ∑pᵢ sᵢ
    sn::Function           ## ∑sᵢ(θ)
    Dws::Function          ## (k×m)
    Dsn::Function          ## (k×m)
    Dsl::Function          ## (n×k)
    Dwsl::Function         ## (k×1)
    Hwsl::Function         ## (kxk)
    kern::SmoothingKernel
    nobs::Int64
    nmom::Int64
    npar::Int64
end

type MomentMatrix
    X::AbstractMatrix ## Unsmoothed
    S::AbstractMatrix ## Smoothed
    g_L::Vector
    g_U::Vector
    kern::SmoothingKernel
    n::Int64        ## Rows of X
    m::Int64        ## Cols of X 
    m_eq::Int64     ## Cols of X[:,1:m_eq] => G
    m_ineq::Int64   ## Cols of X[:,m_eq+1:end] => H
end

type MinimumDivergenceNLPEvaluator <: AbstractNLPEvaluator
    momf::MomentFunction
    div::Divergence
    gele::Int64
    hele::Int64
    solver::AbstractMathProgSolver
    solved::Int64   ## This takes value 0 is model only init and not solved (=1)
    ## TODO: I don't think lmbd and wght are used
    ## lmbd::Array{Float64, 1}
    ## wght::Array{Float64, 1}
end

type SMinimumDivergenceNLPEvaluator <: AbstractNLPEvaluator
  mm::MomentMatrix
  div::Divergence
  ## nobs::Int64
  ## nmeq::Int64
  ## nmineq::Int64
  ## nmom::Int64
  gele::Int64
  hele::Int64
  solver::AbstractMathProgSolver
end

typealias MDNLPE MinimumDivergenceNLPEvaluator
typealias SMDNLPE SMinimumDivergenceNLPEvaluator

## type MinimumDivergenceResult <: MomentEstimatorResult
##     status::Symbol
##     objval::Real
##     coef::Array{Float64, 1}
##     nmom::Integer
##     npar::Integer
##     nobs::Integer
## end

