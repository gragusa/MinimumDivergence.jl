using MinimumDivergence
using FactCheck

function dgp_iv(n::Int64        = 100,
                m::Int64         = 5,
                theta0::Float64  = 0.0,
                rho::Float64     = 0.9,
                CP::Int64        = 20)
    
    ## Generate IV Model with CP
    tau     = fill(sqrt(CP/(m*n)), m)
    z       = randn(n, m)
    vi      = randn(n, 1)
    eta     = randn(n, 1)
    epsilon = rho*eta+sqrt(1-rho^2)*vi
    x       = z*tau + eta    
    y       = x[:,1]*theta0 +  epsilon
    return y, [ones(n) x], [ones(n) z]
end

srand(1)
y, x, z = dgp_iv(100)
g(θ) = z.*(y-x*θ)

solver = IpoptSolver(print_level = 1)

KL = KullbackLeibler
RKL = ReverseKullbackLeibler
MKL = ModifiedKullbackLeibler
MRKL = ModifiedReverseKullbackLeibler

θ₀ = [  0.00,   0.00]
lb = [-20.00, -20.00]
ub = [ 20.00,  20.00]

facts("Check basic interface for estimating θ") do
    context("i.i.d - MD with different divergences") do
        mf = MomentFunction(g, :dual, nobs = size(z,1), nmom = size(z,2), npar = size(x, 2))
        p = MDEstimator(mf, KL(), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
        solve(p)
        @fact status(p) => :Optimal
        @fact coef(p) => roughly([0.10103176863782486,-0.09024762288640839])
        @fact vcov(p, :weighted) => roughly([0.014530624012613189 -0.019653655298360027
                                            -0.019653655298360027 0.10690939757999675])
        @fact stderr(p) => roughly([0.1205430380097216, 0.3269700255069213])        
        @fact getobjval(p)  => greater_than(0)
        
        @fact vcov(p, :hessian) => anything
        @fact stderr(p, :hessian) => roughly([0.14617065585307037, 0.3981458334239717])

        mf = MomentFunction(g, :dual, nobs = size(z,1), nmom = size(z,2), npar = size(x, 2))
        p = MDEstimator(mf, RKL(), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
        solve(p)
        @fact status(p) => :Optimal
        
        mf = MomentFunction(g, :dual, nobs = size(z,1), nmom = size(z,2), npar = size(x, 2))
        p = MDEstimator(mf, CressieRead(.5), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
        solve(p)
        @fact status(p) => :Optimal        
        
        mf = MomentFunction(g, :dual, nobs = size(z,1), nmom = size(z,2), npar = size(x, 2))
        p = MDEstimator(mf, CressieRead(-.5), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
        solve(p)
        @fact status(p) => :Optimal

        mf = MomentFunction(g, :dual, nobs = size(z,1), nmom = size(z,2), npar = size(x, 2))
        p = MDEstimator(mf, MKL(.1), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
        solve(p)
        @fact status(p) => :Optimal

        mf = MomentFunction(g, :dual, nobs = size(z,1), nmom = size(z,2), npar = size(x, 2))
        p = MDEstimator(mf, MRKL(.1), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
        solve(p)
        @fact status(p) => :Optimal
	end

    context("time series - MD with Truncated Kernel") do
        mf = MomentFunction(g, :dual, MinimumDivergence.TruncatedSmoother(1), nobs = size(z,1), nmom = size(z,2), npar = size(x, 2))
        minkl = MDEstimator(mf, KL(), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
        solve(minkl)
        @fact status(minkl) => :Optimal
        
        mf = MomentFunction(g, :dual, TruncatedSmoother(1), nobs = size(z,1), nmom = size(z,2), npar = size(x, 2))
        minkl = MDEstimator(mf, RKL(), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
        solve(minkl)
        @fact status(minkl) => :Optimal
        
        mf = MomentFunction(g, :dual, TruncatedSmoother(1), nobs = size(z,1), nmom = size(z,2), npar = size(x, 2))
        minkl = MDEstimator(mf, MRKL(.1), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
        solve(minkl)
        @fact status(minkl) => :Optimal
        
        mf = MomentFunction(g, :dual, TruncatedSmoother(1), nobs = size(z,1), nmom = size(z,2), npar = size(x, 2))
        minkl = MDEstimator(mf, MKL(.1), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
        solve(minkl)
        @fact status(minkl) => :Optimal
 end

    context("time series - MD with Bartlett Kernel") do
        mf = MomentFunction(g, :dual, BartlettKernel(1), nobs = size(z,1), nmom = size(z,2), npar = size(x, 2))
        minkl = MinDivProb(mf, KL(), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
        solve(minkl)
        @fact status(minkl) => :Optimal
        
        mf = MomentFunction(g, :dual, BartlettKernel(1), nobs = size(z,1), nmom = size(z,2), npar = size(x, 2))
        minkl = MinDivProb(mf, RKL(), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
        solve(minkl)
        @fact status(minkl) => :Optimal
        
        mf = MomentFunction(g, :dual, BartlettKernel(1), nobs = size(z,1), nmom = size(z,2), npar = size(x, 2))
        minkl = MinDivProb(mf, MRKL(.1), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
        solve(minkl)
        @fact status(minkl) => :Optimal
        
        mf = MomentFunction(g, :dual, BartlettKernel(1), nobs = size(z,1), nmom = size(z,2), npar = size(x, 2))
        minkl = MinDivProb(mf, MKL(.1), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
        solve(minkl)
        @fact status(minkl) => :Optimal
    end
    
end

facts("Check MomentMatrix interface") do
    context("Equality only") do
        gg = z.*(y-x*[.1,.1])
        mm = MomentMatrix(gg, IdentityKernel())
        mp = MinDivProb(mm, KullbackLeibler())
        solve(mp)
        @fact status(mp) => :Optimal
    end
    context("Equality and inequality") do
        gg  = z.*(y-x*[.1,.1])
        ggi = reshape(float(x[:,2].>1.96), length(y), 1)
        mm = MomentMatrix(gg, ggi, [0.04].*100, [0.06].*100, IdentityKernel())
        mp = MinDivProb(mm, KullbackLeibler())
        solve(mp)
        @fact status(mp) => :Optimal
    end
end



facts("Check Simplified API - IV") do
    context("IdentityKernel") do
        md = MinDivProb(IV(y,x,z), KullbackLeibler())
        solve(md)
        @fact status(md) => :Optimal
        coef(md)
        vcov(md)
        resolve(md, θ₀)
        @fact status(md) => :Optimal
        coef(md)
        vcov(md, :hessian)
        @fact status(md) => :Optimal
    end
    context("Equality and inequality") do
        gg  = z.*(y-x*[.1,.1])
        ggi = reshape(float(x[:,2].>1.96), length(y), 1)
        mm = MomentMatrix(gg, ggi, [0.04].*100, [0.06].*100, IdentityKernel())
        mp = MinDivProb(mm, KullbackLeibler())
        solve(mp)
        @fact status(mp) => :Optimal
    end
end




FactCheck.exitstatus()


## sim = 1000

## θ₀ = [  0.00]
## lb = [-20.00]
## ub = [ 20.00]


## coeff = Array(Float64, sim, 4)
## g(θ) = z.*(y-x*θ)
## mf = MomentFunction(g, :dual, nobs = 100, nmom = 3, npar = 1)
## rkl = MinDivProb(mf, RKL(), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
## kl = MinDivProb(mf, KL(), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
## mrkl = MinDivProb(mf, MRKL(.1), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
## mkl = MinDivProb(mf, MKL(.1), θ₀, lb, ub, solver=IpoptSolver(print_level=0))

## srand(10)
## for j = 1:sim
## 	y, x, z = randiv(CP = 10, m = 3)
## 	solve(kl)
## 	solve(rkl)
## 	solve(mkl)
## 	solve(mrkl)
## 	coeff[j,1] = coef(kl)[1]
## 	coeff[j,2] = coef(rkl)[1]
## 	coeff[j,3] = coef(mkl)[1]
## 	coeff[j,4] = coef(mrkl)[1]
## 	if any(coeff[j,:] .< -19)
## 		break
## 	end
## end



