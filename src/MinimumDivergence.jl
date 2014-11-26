module MinimumDivergence


using Calculus
using Divergences

using Ipopt
using ArrayViews
using PDMats
using StatsBase
using MathProgBase
using ForwardDiff

import Calculus: gradient
import Divergences: Divergence, hessian
import Base.show
import StatsBase: coef, coeftable, confint, deviance, loglikelihood, nobs, stderr, vcov


abstract SmoothingKernel

__λ = Array(Float64, 10)
__ω = Array(Float64, 100)
__p = Array(Float64, 100)

include("smoothing.jl")
include("momfun.jl")
include("md.jl")
include("methods.jl")

include("utils.jl")
include("MathProgBase.jl")

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

