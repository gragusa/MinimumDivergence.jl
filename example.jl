using Ipopt
using ModelsGenerators
using ArrayViews
using Divergences
using MinimumDivergence

srand(2)
@time y, x, z  = randiv(100, k = 2);

n, k = size(x);
n, m = size(z);

g_i(theta)  = z.*(y-x*theta)
mf = MomentFunction(g_i)
div = ReverseKullbackLeibler()

out=md(mf,
      div,
      zeros(k),
      -ones(k),
      ones(k),
      0,
      "mumps",
      "exact")

@time out=md(mf,
      div,
      zeros(k),
      -ones(k),
      ones(k),
      0,
      "mumps",
      "exact")

## Time series
## Small Monte-Carlo
sim = 1000
Β = zeros(sim, 2)

@time y, x, z  = randiv_ts(100);

g_i(theta)  = z.*(y-x*theta)
mf = MomentFunction(g_i, MinimumDivergence.BartlettKernel(1.))
div = ReverseKullbackLeibler()
n, k = size(x);
n, m = size(z);

srand(2)
sim = 1000
for j=1:sim
  y, x, z  = randiv_ts(128, m=1, σ₁₂=0.0);
  out=md(mf,
      div,
      [0., 1.],
      -18.0*ones(k),
      18.0*ones(k),
      0,
      "mumps",
      "exact")
  Β[j, :] = coef(out)
end




@time vcov!(out)
@time MinimumDivergence.hessian!(out)



mf = MomentFunction(g_i, MinimumDivergence.TruncatedKernel(0.))

@time out=md(mf,
      div,
      zeros(k),
      -10*ones(k),
      10*ones(k),
      0,
      "ma27",
      "exact")

@time vcov!(out);

mf = MomentFunction(g_i, MinimumDivergence.BartlettKernel(0.))

@time out=md(mf,
      div,
      zeros(k),
      -10*ones(k),
      10*ones(k),
      0,
      "ma27",
      "exact")
