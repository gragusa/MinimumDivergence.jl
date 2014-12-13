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
## Exported methods
##
##############################################################################




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

# include("MathProgBase.jl")
# include("smoothing.jl")
# include("momfun.jl")
# include("md.jl")
# include("methods.jl")
# include("utils.jl")

export MomentFunction,
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
       size,
       divergence,
       getobjhess


end # module

