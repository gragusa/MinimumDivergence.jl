module MinimumDivergence

#import Calculus: gradient, hessian
importall Divergences

#using ForwardDiff
#import ForwardDiff.typed_fad_hessian
using Ipopt
using ArrayViews
using ForwardDiff
using PDMats
using StatsBase
using Calculus



include("md.jl")
include("methods.jl")

export MomentFunction,
       MinimumDivergenceProblem,
       md,
       obs,
       ncond,
       lambda,
       weights,
       obj_val,
       mfjacobian!,
       meat!,
       vcov!,
       coef,
       jacobian,
       hessian

end # module

