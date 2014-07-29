module MinimumDivergence


using ForwardDiff
using Ipopt
using ArrayViews
using ForwardDiff
using PDMats
using StatsBase
using Calculus
importall Divergences


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

