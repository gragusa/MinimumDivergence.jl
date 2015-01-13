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
using Distributions
try
    using KNITRO
end 

global const isknitro::Bool = isdefined(:KNITRO)

import MathProgBase: getobjval
import MathProgBase.MathProgSolverInterface: AbstractMathProgSolver,
                                             AbstractNLPEvaluator,
                                             model,
                                             loadnonlinearproblem!,
                                             initialize,
                                             setwarmstart!,
                                             status,
                                             eval_f,
                                             eval_g,
                                             jac_structure,
                                             hesslag_structure,
                                             eval_grad_f,
                                             eval_jac_g,
                                             eval_hesslag,
                                             eval_hesslag_prod,
                                             optimize!

##############################################################################
##
## Extend methods
##
##############################################################################
import Base: show, size
import Calculus: gradient, hessian
import Divergences: Divergence
import StatsBase: coef, coeftable, confint, loglikelihood, nobs, stderr, vcov

##############################################################################
##
## Load file
##
##############################################################################
include("types.jl")
include("smoothing.jl")
include("momfun.jl")
include("interface.jl")
include("interface_simple.jl")
include("api.jl")
include("methods.jl")
include("utils.jl")
include("vcov.jl")

global __λ
global __p

##############################################################################
##
## Exported methods
##
##############################################################################
export TruncatedKernel,
       BartlettKernel,
       IdentityKernel,
       MomentFunction,
       MomentMatrix,
       MinDivProb,
       solve,
       getobjval,
       getlambda,
       geteta,
       getmdweights,
       status,
       nobs,
       npar,
       nmom,
       coef,
       momf_jac,
       momf_var,
       vcov,
       vcov!,
       stderr,
       size,
       divergence,
       getobjhess

end # module
