using MinimumDivergence
#using Ipopt
#conditional using/testing a package?
#using KNITRO
using ModelsGenerators
#using ForwardDiff
#using Divergences
using FactCheck

srand(4)
obs = 10000
y, x, z = randiv(obs, 5);
ex = randn(obs);
x = [x ex];
z = [z ex];

m = size(z, 2);
k = size(x, 2);

g(θ) = z.*(y-x*θ);
mf_dual = MomentFunction(g, :dual,  nobs = obs, nmom = m, npar = k)
mf_typd = MomentFunction(g, :typed, nobs = obs, nmom = m, npar = k);
mf_diff = MomentFunction(g, :diff,  nobs = obs, nmom = m, npar = k);

θ = [.1, .1];
λ = ones(m);
π = ones(obs);
lb = [-2, -2];
ub = [2, 2];

facts("Check basic interface for momf and derivatives") do
    context("∑πs(θ) is a (m x 1)") do
        @fact length(mf_dual.sn(theta)) => m
        @fact mf_dual.sn(theta) => roughly(mf_typd.sn(theta))
        @fact mf_dual.sn(theta) => roughly(mf_diff.sn(theta))
    end
    context("∂∑s(θ)/∂θ' is a (m x k)") do
        @fact size(mf_dual.Dsn(theta)) => (m, k)
        @fact mf_dual.Dsn(theta) => roughly(mf_typd.Dsn(theta))
        @fact mf_typd.Dsn(theta) => roughly(mf_diff.Dsn(theta))
    end

    context("∂∑πs(θ)/∂θ' is a (m x k)") do
        @fact size(mf_dual.Dws(theta, π)) => (m, k)
        @fact mf_dual.Dws(theta, π) => roughly(mf_typd.Dws(theta, π))
        @fact mf_dual.Dws(theta, π) => roughly(mf_diff.Dws(theta, π))
    end
    context("∂∑s(θ)'λ/∂θ' is (n x k)") do
        @fact size(mf_typd.Dsl(theta, λ)) => (obs, k)
        @fact mf_typd.Dsl(theta, λ) => roughly(mf_dual.Dsl(theta, λ))
        @fact mf_diff.Dsl(theta, λ) => roughly(mf_dual.Dsl(theta, λ))
    end
    context("∂∑πs(θ)'λ/∂θ' is (n x k)") do
        @fact size(mf_typd.Dwsl(theta, π, λ)) => (obs, k)
        @fact mf_typd.Dwsl(theta, π, λ) => roughly(mf_dual.Dwsl(theta, π, λ))
        @fact mf_diff.Dwsl(theta, π, λ) => roughly(mf_dual.Dwsl(theta, π, λ))
    end
    context("∂²∑πs(θ)'λ/∂θ∂θ' is (k x k)") do
        @fact size(mf_typd.Hwsl(theta, π, λ)) => (k, k)
        @fact mf_typd.Hwsl(theta, π, λ) => roughly(mf_dual.Hwsl(theta, π, λ))
        @fact mf_diff.Hwsl(theta, π, λ) => roughly(mf_dual.Hwsl(theta, π, λ))
    end
end

facts("Solve MD problems") do
    context("Construct MDEstimator problems") do
        solver = IpoptSolver(print_level = 1)
        div = KullbackLeibler()

        p_iid = MDEstimator(mf_dual, div, θ, lb, ub, solver=solver)

        kern = MinimumDivergence.TruncatedSmoother(1)
        mf_dual = MomentFunction(g, :dual,  kernel = kern, nobs = obs, nmom = m, npar = k)
        p_truncated = MDEstimator(mf_dual, div, θ, lb, ub, solver=solver)


        kern = MinimumDivergence.BartlettSmoother(1)
        mf_dual = MomentFunction(g, :dual, kernel = kern, nobs = obs, nmom = m, npar = k)
        p_bartlett = MDEstimator(mf_dual, div, θ, lb, ub, solver=solver)

        solve(p_iid);
        solve(p_truncated);
        solve(p_bartlett);

        @fact coef(p_iid) => roughly(coef(p_truncated), atol = 0.01)
        @fact coef(p_iid) => roughly(coef(p_bartlett), atol = 0.01)

        @fact stderr(p_iid) => roughly(stderr(p_bartlett), atol = 0.01)
        @fact stderr(p_iid) => roughly(stderr(p_truncated), atol = 0.01)

        @fact stderr(p_iid, :hessian) => roughly(stderr(p_iid), atol = 0.02)
        @fact stderr(p_iid, :hessian) => roughly(stderr(p_truncated, :hessian), atol = 0.02)
        @fact stderr(p_iid, :hessian) => roughly(stderr(p_bartlett, :hessian), atol = 0.02)
        
    end
end














## ## Test MDProblem
## G = g([.06,0.1])
## c = zeros(6)

## r  = MinimumDivergenceProblem(G, c)

## H   = float(x.>1.96)
## lwr = n*[0.03, 0.03]
## upp = n*[0.06, 0.06]

## r  = MinimumDivergenceProblem(G, c, H, lwr, upp)











## @test abs(maximum(coef(minkl) - [-0.13187044493542183,0.11713620496711936]))<1e-09
## vcov!(minkl)




## srand(2)
## y, x, z = randiv(100, k=2, CP = 100)
## solve(minkl)
## @test abs(maximum(coef(minkl) - [-0.05304299084504518,0.10735722297685665]))<1e-09
## vcov!(minkl)
## vcov(minkl)


## minkl = MinDivProb(mf, div, θ₀, lb, ub, solver=IpoptSolver(linear_solver = "ma27"))
## solve(minkl)






## ##out1 = MinimumDivergence.mdtest(mf, div, θ₀, lb, ub, solver=IpoptSolver())
## ##out2 = MinimumDivergence.mdtest(mf, div, θ₀, lb, ub, solver=KnitroSolver())

## ## New interface





## # import MathProgBase.MathProgSolverInterface
## # MinimumDivergence.mdtest(mf, KullbackLeibler(), [.1,.1], [-2, -2.], [2.,2]; solver = IpoptSolver())

## # out3 = MinimumDivergence.mdtest(mf, div, θ₀, lb, ub, solver=KnitroSolver(ms_enable=1))

## # out4 = MinimumDivergence.mdtest(mf, div, θ₀, lb, ub, solver=KnitroSolver(KTR_PARAM_MULTISTART=1))


## # ## Time series
## # ## Small Monte-Carlo
## # sim = 1000
## # Β = zeros(sim, 2)

## # @time y, x, z  = randiv_ts(128);

## # g_i(theta)  = z.*(y-x*theta)
## # mf = MomentFunction(g_i, MinimumDivergence.BartlettKernel(1.))
## # div = ReverseKullbackLeibler()
## # n, k = size(x);
## # n, m = size(z);

## # srand(2)
## # sim = 1000
## # for j=1:sim
## #   y, x, z  = randiv_ts(128, m=1, σ₁₂=0.0);
## #   out=md(mf,
## #       div,
## #       [0., 1.],
## #       -18.0*ones(k),
## #       18.0*ones(k),
## #       0,
## #       "mumps",
## #       "exact")
## #   Β[j, :] = coef(out)
## # end




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

##MDEstimator: ModifiedKullbackLeibler(), 1200 parameter(s) with 1146 moment(s)
