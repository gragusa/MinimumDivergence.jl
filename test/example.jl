using MinimumDivergence
#using Ipopt
#conditional using/testing a package?
#using KNITRO
using ModelsGenerators
#using ForwardDiff
#using Divergences
using Base.Test

srand(2)
y, x, z = randiv(100, 5)
x = [x randn(100)];
z = [z randn(100)];
g(θ) = z.*(y-x*θ)
mf_dual = MomentFunction(g, :dual,  nobs = 100, nmom = 6, npar = 2)
mf_typd = MomentFunction(g, :typed, nobs = 100, nmom = 6, npar = 2)
mf_diff = MomentFunction(g, :diff,  nobs = 100, nmom = 6, npar = 2)

theta = [.1, .1];

mf_dual.sn(theta)
mf_typd.sn(theta)
mf_diff.sn(theta)

mf_dual.Dsn(theta)
mf_typd.Dsn(theta)
mf_diff.Dsn(theta)

mf_dual.Dws(theta, ones(100))
mf_typd.Dws(theta, ones(100))
mf_diff.Dws(theta, ones(100))

mf_dual.Dsl(theta, ones(6))
mf_typd.Dsl(theta, ones(6))
mf_diff.Dsl(theta, ones(6))

mf_dual.Dwsl(theta, ones(100), ones(6))
mf_typd.Dwsl(theta, ones(100), ones(6))
mf_diff.Dwsl(theta, ones(100), ones(6))

mf_dual.Hwsl(theta, ones(100), ones(6))
mf_typd.Hwsl(theta, ones(100), ones(6))
mf_diff.Hwsl(theta, ones(100), ones(6))


solver = IpoptSolver()
θ₀ = [0.,0]
div = KullbackLeibler()
lb = [-20, -20]
ub = [20, 20]
p = MinimumDivergenceEstimator(mf_dual, div, θ₀, lb, ub, solver=solver)
#MathProgBase.MathProgSolverInterface.setwarmstart!(p.m, ones(102))
#MathProgBase.MathProgSolverInterface.optimize!(p.m)
solve(p)
hessian!(p)
vcov(p, :hessian)








## Test MDProblem
G = g([.06,0.1])
c = zeros(6)

r  = MinimumDivergenceProblem(G, c)

H   = float(x.>1.96)
lwr = n*[0.03, 0.03]
upp = n*[0.06, 0.06]

r  = MinimumDivergenceProblem(G, c, H, lwr, upp)











@test abs(maximum(coef(minkl) - [-0.13187044493542183,0.11713620496711936]))<1e-09
vcov!(minkl)




srand(2)
y, x, z = randiv(100, k=2, CP = 100)
solve(minkl)
@test abs(maximum(coef(minkl) - [-0.05304299084504518,0.10735722297685665]))<1e-09
vcov!(minkl)
vcov(minkl)


minkl = MinDivProb(mf, div, θ₀, lb, ub, solver=IpoptSolver(linear_solver = "ma27"))
solve(minkl)






##out1 = MinimumDivergence.mdtest(mf, div, θ₀, lb, ub, solver=IpoptSolver())
##out2 = MinimumDivergence.mdtest(mf, div, θ₀, lb, ub, solver=KnitroSolver())

## New interface





# import MathProgBase.MathProgSolverInterface
# MinimumDivergence.mdtest(mf, KullbackLeibler(), [.1,.1], [-2, -2.], [2.,2]; solver = IpoptSolver())

# out3 = MinimumDivergence.mdtest(mf, div, θ₀, lb, ub, solver=KnitroSolver(ms_enable=1))

# out4 = MinimumDivergence.mdtest(mf, div, θ₀, lb, ub, solver=KnitroSolver(KTR_PARAM_MULTISTART=1))


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

# @time out=md(mf,
#       div,
#       zeros(k),
#       -10*ones(k),
#       10*ones(k),
#       0,
#       "ma27",
#       "exact")

# @time vcov!(out);

# mf = MomentFunction(g_i, MinimumDivergence.BartlettKernel(0.))

# @time out=md(mf,
#       div,
#       zeros(k),
#       -10*ones(k),
#       10*ones(k),
#       0,
#       "ma27",
#       "exact")

MDEstimator: ModifiedKullbackLeibler(), 1200 parameter(s) with 1146 moment(s)
