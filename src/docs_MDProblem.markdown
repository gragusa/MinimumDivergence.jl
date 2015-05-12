## Minimum Divergence Problem
The minimum divergence problem is defined as follows
\begin{equation*}
\min_{(\pi_1,\ldots,\pi_n)} \sum_{i=1}^n \gamma(n\pi_i)
\end{equation*}
subject to
\begin{align*}
& \sum_{i=1}^n G_{ji} \pi_i = c_j, \quad j = 1,\ldots,m_{g}   \\
& l_s \leq \sum_{i=1}^n H_{si} \pi_i \leq u_s, \quad s = 1,\ldots,m_{h}.
\end{align*}

The basic construct for `MinimumDivergenceProblem` type is
```
MinimumDivergenceProblem(G::AbstractMatrix, c::Vector, H::AbstractMatrix,
                         b::Vector; k::SmoothingKernel)
```





