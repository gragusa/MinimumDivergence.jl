module MinimumDivergence

##############################################################################
##
## Dependencies
##
##############################################################################


using Calculus
using Divergences
using Ipopt
using ArrayViews
using PDMats
using StatsBase
using MathProgBase
using ForwardDiff


##############################################################################
##
## Extend methods 
##
##############################################################################

import Base.show
import Calculus: gradient, hessian
import Divergences: Divergence
import StatsBase: coef, coeftable, confint, loglikelihood, nobs, stderr, vcov
import MathProgBase.MathProgSolverInterface

##############################################################################
##
## Exported methods
##
##############################################################################




##############################################################################
##
## Load file
##
##############################################################################


# include("types.jl")
# include("interface.jl")
# include("api.jl")
# include("smoothing.jl")
# include("methods.jl")
# include("utils.jl")



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


global __λ 
global __p

include("MathProgBase.jl")
include("smoothing.jl")
include("momfun.jl")
include("md.jl")
include("methods.jl")
include("utils.jl")


export MomentFunction,
       MinimumDivergenceProblem,
       md,
       ivmd,
       nobs,
       ncond,
       lambda,
       weights,
       obj_val,
       mfjacobian!,
       meat!,
       vcov!,
       vcov,
       coef,
       jacobian,
       hessian,
       stderr,
       show


end # module

