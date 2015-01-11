# MinimumDivergence.jl

[![Build Status](https://travis-ci.org/gragusa/MinimumDivergence.jl.svg?branch=develop)](https://travis-ci.org/gragusa/MinimumDivergence.jl)
[![Coverage Status](https://coveralls.io/repos/gragusa/MinimumDivergence.jl/badge.png?branch=develop)](https://coveralls.io/r/gragusa/MinimumDivergence.jl?branch=develop)

## Examples

```
using Ipopt
using ModelsGenerators
using ArrayViews
using Divergences
using MinimumDivergence

srand(2)
@time y, x, z  = randiv(100, k = 2)

n, k = size(x)
n, m = size(z)

g_i(theta)  = z.*(y-x*theta)
mf = MomentFunction(g_i)
div = ReverseKullbackLeibler()

@time out=md(mf,
      div,
      zeros(k),
      -ones(k),
      ones(k),
      0,
      "mumps",
      "exact")

mf = MomentFunction(g_i, MinimumDivergence.TruncatedKernel(3.))

@time out=md(mf,
      div,
      zeros(k),
      -10*ones(k),
      10*ones(k),
      0,
      "ma27",
      "exact")

```
