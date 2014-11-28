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


include("types.jl")
include("smoothing.jl")
include("momfun.jl")
include("interface.jl")
include("api.jl")
include("methods.jl")
include("utils.jl")



global __λ 
global __p

# include("MathProgBase.jl")
# include("smoothing.jl")
# include("momfun.jl")
# include("md.jl")
# include("methods.jl")
# include("utils.jl")


export MomentFunction, 
       MinimumDivergenceProblem, 
       nobs, nmom, npar
       
end # module

