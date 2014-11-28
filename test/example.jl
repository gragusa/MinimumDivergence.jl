using MinimumDivergence
using Ipopt
using KNITRO
using ModelsGenerators
using ForwardDiff
using Divergences


y, x, z = randiv(k=2)
g(θ) = z.*(y-x*θ)
mf = MomentFunction(g, :dual, nobs = 100, nmom = 6, npar = 2)
solver = IpoptSolver()
θ₀ = [1.,0]
div = KullbackLeibler()
lb = [-20, -20]
ub = [20, 20]

out1 = MinimumDivergence.mdtest(mf, div, θ₀, lb, ub, solver=IpoptSolver())
out2 = MinimumDivergence.mdtest(mf, div, θ₀, lb, ub, solver=KnitroSolver())
import MathProgBase.MathProgSolverInterface
MinimumDivergence.mdtest(mf, KullbackLeibler(), [.1,.1], [-2, -2.], [2.,2]; solver = IpoptSolver())

out3 = MinimumDivergence.mdtest(mf, div, θ₀, lb, ub, solver=KnitroSolver(ms_enable=1))
out4 = MinimumDivergence.mdtest(mf, div, θ₀, lb, ub, solver=KnitroSolver(KTR_PARAM_MULTISTART=1))


# ## Time series
# ## Small Monte-Carlo
# sim = 1000
# Β = zeros(sim, 2)

# @time y, x, z  = randiv_ts(128);

# g_i(theta)  = z.*(y-x*theta)
# mf = MomentFunction(g_i, MinimumDivergence.BartlettKernel(1.))
# div = ReverseKullbackLeibler()
# n, k = size(x);
# n, m = size(z);

# srand(2)
# sim = 1000
# for j=1:sim
#   y, x, z  = randiv_ts(128, m=1, σ₁₂=0.0);
#   out=md(mf,
#       div,
#       [0., 1.],
#       -18.0*ones(k),
#       18.0*ones(k),
#       0,
#       "mumps",
#       "exact")
#   Β[j, :] = coef(out)
# end
# @time vcov!(out)
# @time MinimumDivergence.hessian!(out)
# mf = MomentFunction(g_i, MinimumDivergence.TruncatedKernel(0.))

