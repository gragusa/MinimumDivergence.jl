module MinimumDivergence


using Calculus
using Divergences

using Ipopt
using ArrayViews
using PDMats
using StatsBase



import Calculus: gradient
import Divergences: Divergence, hessian
import Base.show
import StatsBase: coef, coeftable, confint, deviance, loglikelihood, nobs, stderr, vcov



include("md.jl")
include("methods.jl")

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

