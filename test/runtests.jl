using MinimumDivergence
using Ipopt
#conditional using/testing a package?
#using KNITRO
using ModelsGenerators
using ForwardDiff
using Divergences
using FactCheck


srand(1)
y, x, z = randiv(100, k=2)

g(θ) = z.*(y-x*θ)

function G(θ)
	n, k, m = (size(x)[1], size(x)[2], size(z)[2])
	U = Array(Float64, n, k, m)
	for j=1:m
		U[:,:,j] = -z[:,j].*x
	end
	U
end

solver = IpoptSolver()

KL = KullbackLeibler
RKL = ReverseKullbackLeibler
MKL = ModifiedKullbackLeibler
MRKL = ModifiedReverseKullbackLeibler

θ₀ = [  0.00,   0.00]
lb = [-20.00, -20.00]
ub = [ 20.00,  20.00]

facts("Check basic interface for estimating θ") do
	context("i.i.d - MD with different divergences") do
		mf = MomentFunction(g, :dual, nobs = 100, nmom = 6, npar = 2)
		minkl = MinDivProb(mf, KL(), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
		solve(minkl)
		@fact coef(minkl) => roughly([0.02003660467780194,0.16286537485821131])

		mf = MomentFunction(g, :dual, nobs = 100, nmom = 6, npar = 2)
		minkl = MinDivProb(mf, RKL(), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
		solve(minkl)
		@fact coef(minkl) => roughly([-0.05349524128174826,0.19915918195684024])

		mf = MomentFunction(g, :dual, nobs = 100, nmom = 6, npar = 2)
		minkl = MinDivProb(mf, CressieRead(.5), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
		solve(minkl)
		@fact coef(minkl) => roughly([0.05852212073265522,0.1396089479703614])

		mf = MomentFunction(g, :dual, nobs = 100, nmom = 6, npar = 2)
		minkl = MinDivProb(mf, CressieRead(-.5), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
		solve(minkl)
		@fact coef(minkl) => roughly([-0.016362778384386892,0.18205290875034516])

		mf = MomentFunction(g, :dual, nobs = 100, nmom = 6, npar = 2)
		minkl = MinDivProb(mf, MKL(.1), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
		solve(minkl)
		@fact coef(minkl) => roughly([0.03227781191888403,0.15709786412788168])

		mf = MomentFunction(g, :dual, nobs = 100, nmom = 6, npar = 2)
		minkl = MinDivProb(mf, MRKL(.1), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
		solve(minkl)
		@fact coef(minkl) => roughly([-0.014073384470336923,0.18463261677504222])
	end

	context("time series - MD with different divergences") do
		mf = MomentFunction(g, :dual, TruncatedKernel(1), nobs = 100, nmom = 6, npar = 2)
		minkl = MinDivProb(mf, KL(), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
		solve(minkl)
		@fact coef(minkl) => roughly([-0.16192622980204185,0.25111191239583974])

		mf = MomentFunction(g, :dual, TruncatedKernel(1), nobs = 100, nmom = 6, npar = 2)
		minkl = MinDivProb(mf, RKL(), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
		solve(minkl)
		@fact coef(minkl) => roughly([-0.2122902620387113,0.2890456685169426])

		mf = MomentFunction(g, :dual, TruncatedKernel(1), nobs = 100, nmom = 6, npar = 2)
		minkl = MinDivProb(mf, MRKL(.1), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
		solve(minkl)
		@fact coef(minkl) => roughly([-0.15704507414384197,0.25898344329988093])

		mf = MomentFunction(g, :dual, TruncatedKernel(1), nobs = 100, nmom = 6, npar = 2)
		minkl = MinDivProb(mf, MKL(.1), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
		solve(minkl)
		@fact coef(minkl) => roughly([-0.13617356187145876,0.23861461177870208])
	end
end

facts("Check MomentMatrix interface") do
	context("Equality only") do
		gg = z.*(y-x*[.1,.1])
		mm = MomentMatrix(gg, IdentityKernel())
		mp = MinDivProb(mm, KullbackLeibler())
		solve(mp)
	end
	context("Equality and inequality") do
		gg  = z.*(y-x*[.1,.1])
		ggi = float(x.>1.96)
		mm = MomentMatrix(gg, ggi, [0.04, 0.04].*100, [0.06, 0.06].*100, IdentityKernel())
		mp = MinDivProb(mm, KullbackLeibler())
		solve(mp)
	end
end

sim = 1000

θ₀ = [  0.00]
lb = [-20.00]
ub = [ 20.00]


coeff = Array(Float64, sim, 4)
g(θ) = z.*(y-x*θ)
mf = MomentFunction(g, :dual, nobs = 100, nmom = 3, npar = 1)
rkl = MinDivProb(mf, RKL(), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
kl = MinDivProb(mf, KL(), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
mrkl = MinDivProb(mf, MRKL(.1), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
mkl = MinDivProb(mf, MKL(.1), θ₀, lb, ub, solver=IpoptSolver(print_level=0))
srand(10)
for j = 1:sim
	y, x, z = randiv(100, CP = 10, m = 3)
	solve(kl)
	solve(rkl)
	solve(mkl)
	solve(mrkl)
	coeff[j,1] = coef(kl)[1]
	coeff[j,2] = coef(rkl)[1]
	coeff[j,3] = coef(mkl)[1]
	coeff[j,4] = coef(mrkl)[1]
	if any(coeff[j,:] .< -19)
		break
	end
end



