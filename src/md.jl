using ForwardDiff
using Ipopt
using ArrayViews
using Divergences
using ForwardDiff

## Simplest API
## The only function that is passed is
##
## g_i(theta::Vector) = ...
##
#
## which returns a (n x m) Array{Float64, 2}
##
##
## It remains to be computed
##
##
## Dg_n = function(theta::Vector, p::Vector) ...   m x k
## Dg_j = function(theta::Vector) ...              n x k
## H_n = function(theta::Vector, lambda::Vector, p::Vector) k x x

function gettril(x::Array{Float64, 2})
    n,m = size(x)
    a = zeros(convert(Int, n.*(n+1)/2))
    k::Int = 1
    for i = 1:n
        for j = 1:i
            a[k] = x[i, j]
            k += 1
        end
    end
    return a
end

function ForwardDiff.typed_fad_hessian{T<:Real}(f::Function, ::Type{T})
  g(x::Vector{T}) = mf_hessian(f(FADHessian(x)))
  return g
end

function mf_hessian{T<:Real, n}(x::Array{FADHessian{T, n},1})
  y = Array(T, n, n)
  k = 1
    for i in 1:n
      for j in 1:i
        y[i, j] = x[1].h[k]
        k += 1
      end
    end
  y
end

type MomentFunction
    g_i::Function
    Dg_n::Function    ## return k x m
    Dg_j::Function    ## return n x k
    H_n::Function     ## return kxk (hessian of lambda'\sum p_i g_i
end


immutable MinimumDivergenceVariance
  stab::Array{Float64, 2}
end

immutable MinimumDivergenceProblem
  mf::MomentFunction
  div::Divergence
  status::Symbol
  fprob::IpoptProblem
  lambda::Array{Float64,1}
  x_ul::Array{Float64,1}
  x_lw::Array{Float64,1}
##Sigma::MinimumDivergenceVariance
end





function MomentFunction(g_i::Function)

    g_n(theta::Vector) = g_i(theta)'*__p
    Dg_n = forwarddiff_jacobian(g_n, Float64, fadtype=:typed)

    function constraint_g_theta(theta::Vector)
        __p'*g_i(theta)*__lambda
    end

    function constraint_g_lambda(theta::Vector)
        g_i(theta)*__lambda
    end

    Dg_j = forwarddiff_jacobian(constraint_g_lambda, Float64, fadtype=:typed)
    H_n  = forwarddiff_hessian(constraint_g_theta, Float64, fadtype=:typed)

    MomentFunction(g_i, Dg_n, Dg_j, H_n)
end


function md(mf::MomentFunction,
               divergence::Divergence,
               init_val::Array{Float64,1}, lb::Array{Float64, 1},
               ub::Array{Float64, 1},
               print_level::Int64 = 0,
               linear_solver::ASCIIString = "ma27",
               hessian_approximation = "exact")

    global __p
    global __lambda

    theta0 = copy(init_val)

    _g_i = mf.g_i(theta0)

    k    = length(theta0)
    n, m = size(_g_i)

    ## Hpl = zeros(n,k,m)
    ## for j=1:m
    ##     Hpl[:,:,j] = -x.*z[:,j]
    ## end

    u = [ones(n), theta0]

    gele::Int64 = (n+k)*(m+1)-k
    hele::Int64 = n*k + n + (k+1)*k/2
    ## This are the lower Triangular element of Hessian


    function eval_f(u::Vector{Float64})
        evaluate(divergence, u[1:n])
    end

    function eval_grad_f(u, grad_f)
        for j=1:n
            @inbounds grad_f[j] = gradient(divergence, u[j])
        end
        for j=(n+1):(n+k)
            @inbounds grad_f[j] = 0.0
        end
    end

    function eval_g(u, g)
        theta   = u[(n+1):(n+k)]
        p       = u[1:n]
        @inbounds g[1:m]  = mf.g_i(theta)'p
        @inbounds g[m+1]  = sum(p)
    end

    function eval_jac_g(
                        u,     # Current solution
                        mode,  # Either :Structure or :Values
                        rows,  # Sparsity structure - row indices
                        cols,  # Sparsity structure - column indices
                        val)   # The values of the Jacobian )

        if mode == :Structure
            for j = 1:(m+1), kk = 1:(n+k)
                if !((kk > n) && (j==m+1))
                    @inbounds rows[kk+(j-1)*(n+k)] = j
                    @inbounds cols[kk+(j-1)*(n+k)] = kk
                end
            end

        else
            theta = u[(n+1):(n+k)]
            __p   = u[1:n]
            gg    = mf.g_i(theta)       ## z.*(y-x*theta) n x m
            dg    = -mf.Dg_n(theta)'     ## k x m
            jac   = [ [gg, dg] [ones(n), zeros(k)] ]
            for icol = 1:(n+k), irow = 1:(m+1)
                if !((icol > n) && (irow==m+1))
                    @inbounds val[icol+(irow-1)*(n+k)] = jac[icol, irow]
                end
            end
        end
    end


    function eval_h(u, mode, rows, cols, obj_factor, lambda, values)
        if mode == :Structure
            for j = 1:n
                @inbounds rows[j] = j
                @inbounds cols[j] = j
            end
            idx = n+1

            for d = 1:n
                for j = 1:k
                    @inbounds rows[idx] = n+j
                    @inbounds cols[idx] = d
                    idx += 1
                end
            end

            for j = 1:k
                for d = 1:j
                    @inbounds rows[idx] = n+j
                    @inbounds cols[idx] = n+d
                    idx += 1
                end
            end
        else
            theta    = u[(n+1):(n+k)]
            __p  = u[1:n]
            __lambda = lambda[1:m]

            H = transpose(mf.Dg_j(theta))

            if obj_factor==0
                for j=1:n
                    @inbounds values[j] = 0.0
                end
            else
                for j=1:n
                    @inbounds values[j] = obj_factor*hessian(divergence, u[j])
                end
            end
            @inbounds values[n+1:n*k+n] = H[:]
            @inbounds values[n*k+n+1:hele] = gettril(mf.H_n(theta))
        end
    end

    g_L = [zeros(m), n];
    g_U = [zeros(m), n];

    x_L = [zeros(n),  lb];
    x_U = [ones(n)*n, ub];

    fprob = Ipopt.createProblem(
                                n+k,
                                x_L,
                                x_U,
                                m+1,
                                g_L,
                                g_U,
                                gele,
                                hele,
                                eval_f,
                                eval_g,
                                eval_grad_f,
                                eval_jac_g,
                                eval_h)

    fprob.x = [ones(n), init_val];

    mult_g = zeros(m+1)
    mult_x_U = zeros(n+k)
    mult_x_L = zeros(n+k)

    ## ma27 (use the Harwell routine MA27)
    ## ma57 (use the Harwell routine MA57)
    ## ma77 (use the Harwell routine HSL_MA77)
    ## ma86 (use the Harwell routine HSL_MA86)
    ## ma97 (use the Harwell routine HSL_MA97)
    ## pardiso (use the Pardiso package)
    ## wsmp (use WSMP package)
    ## mumps (use MUMPS package)

    Ipopt.addOption(fprob, "derivative_test", "none");
    Ipopt.addOption(fprob, "print_level", print_level);
    Ipopt.addOption(fprob, "hessian_approximation", hessian_approximation);
    Ipopt.addOption(fprob, "linear_solver", linear_solver);
    status = Ipopt.solveProblem(fprob, mult_g, mult_x_U, mult_x_L);

    ## println(Ipopt.ApplicationReturnStatus[status])
    ## println(fprob.x)
    ## println(fprob.obj_val)

    return MinimumDivergenceProblem(mf, divergence, Ipopt.ApplicationReturnStatus[status], fprob, mult_g, mult_x_U, mult_x_L)
end