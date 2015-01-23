using MinimumDivergence
using ModelsGenerators

y, x, z = randiv(n = 500, k = 3, m = 15);

mdp = MinDivProb(IV(y,x,z), KullbackLeibler(), solver = IpoptSolver(print_level = 0));
@time solve(mdp);
coef(mdp)
vcov(mdp)
@time vcov(mdp, :hessian)

mdp = MinDivProb(IV(y,x,z), ChiSquared(), solver = IpoptSolver(print_level = 0));
@time solve(mdp);




g(θ) = z.*(y-x*θ)
θ₀ = [  0.00]
lb = [-20.00]
ub = [ 20.00]

mf = MomentFunction(g, :dual, nobs = size(z,1), nmom = size(z,2), npar = size(x, 2));
minkl = MinDivProb(mf, KullbackLeibler(), θ₀, lb, ub, solver=IpoptSolver(print_level=0));
@time solve(minkl);
@time vcov(minkl, :hessian)


