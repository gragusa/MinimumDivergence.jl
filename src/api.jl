type MinimumDivergenceEstimator
    m::MathProgBase.AbstractMathProgModel
    e::MDNLPE
    H::Union(Nothing, Array{Float64, 2})
end

typealias MDEstimator MinimumDivergenceEstimator

immutable MinimumDivergenceProblem
    m::MathProgBase.AbstractMathProgModel
    e::SMDNLPE
end

typealias MDProblem MinimumDivergenceProblem
#####################################################################
## MDEstimator
##
#####################################################################

## function MDEstimator(mf::MomentFunction, div::Divergence, θ₀::Vector,
##                      lb::Vector, ub::Vector, π::Vector, A::Array{Float64, 2},
##                      c::Array{Float64, 1}, h::Function; solver=IpoptSolver())
    ## Aθ = c
    ## h(θ) = 0
## end

function MDEstimator(mf::MomentFunction, div::Divergence, θ₀::Vector,
                     lb::Vector, ub::Vector, π₀::Vector; solver=IpoptSolver())
    model = MathProgBase.MathProgSolverInterface.model(solver)
    n, m, k = size(mf)
    u₀ = [π₀, θ₀]
    gele = int((n+k)*(m+1)-k)
    hele = int(n*k + n + (k+1)*k/2)
    g_L = [zeros(m), n];
    g_U = [zeros(m), n];
    ## TODO:
    ## Test whether restricting this makes sense
    ## They should be in principle be (0,n)
    ## and (-infty, +infty) for difergence that takes value in R
    u_L = [ones(n)*(-Inf),  lb]
    u_U = [ones(n)*(+Inf), ub]
    e = MDNLPE(mf, div, gele, hele, solver, 0)
    loadnonlinearproblem!(model, n+k, m+1, u_L, u_U, g_L, g_U, :Min, e)
    setwarmstart!(model, u₀)
    ## Add a flag - problem is loaded, but it has not being solved
    MinimumDivergenceEstimator(model, e, Nothing())
end

function MDEstimator(mf::MomentFunction, div::Divergence, θ₀::Vector,
                     lb::Vector, ub::Vector; solver=IpoptSolver())
    n, m, k = size(mf)
    MDEstimator(mf, div, θ₀, lb, ub, ones(n), solver = solver)
end

function solve(mdp::MDEstimator)
    n, m, k = size(mdp)
    x0 = _initial_x(mdp.m)
    resolve(mdp, x0[n+1:end], x0[1:n])
end

function resolve(mdp::MDEstimator, θ₀::Vector, π₀::Vector, λ₀::Vector)
    n, m, k = size(mdp.e.momf)
    n0, m0, k0 = (length(π₀), length(λ₀), length(θ₀))
    @assert  k0 == k   "Starting point for θ of wrong length. It should be a ($k x 1) array got a ($k0 x 1) array"
    @assert n0 == n   "Starting point for π of wrong length. It should be a ($n x 1) array got a ($n0 x 1) array"
    @assert m0 == m "Starting point for λ of wrong length. It should be a ($n x 1) array got a ($m0 x 1) array"
    lambda0 = [λ₀, 0.]
    stat = _resolve(mdp.m, [π₀, θ₀], lambda0)
    mdp.e.solved = status(mdp) == :Optimal ? 1 : 999
    return mdp
end

function resolve(mdp::MinimumDivergenceEstimator, θ₀::Vector)
    p0 = ones(n)
    resolve(mdp, θ₀, p0)
end

function resolve(mdp::MinimumDivergenceEstimator, θ₀::Vector, π₀::Vector)
    n, m, k = size(mdp.e.momf)
    lambda0 = Array(Float64, m)
    resolve(mdp, θ₀, π₀, lambda0)
end


#####################################################################
## MDProblem
##
#####################################################################

## Low level function
function MinimumDivergenceProblem(X::AbstractMatrix, X_L::Vector, X_U::Vector,
                                  m_eq::Int64, m_ineq::Int64, div::Divergence,
                                  solver::AbstractMathProgSolver, k::SmoothingKernel)
    n, m = size(X)
    model = MathProgBase.MathProgSolverInterface.model(solver)

    gele = int(n*(m+1))
    hele = int(n)
    u_L = [ones(n)*(-Inf)]
    u_U = [ones(n)*(+Inf)]
    mm  = MomentMatrix(X, smooth(X, k), X_L, X_U, k, n, m, m_eq, m_ineq)
    e   = SMDNLPE(mm, div, gele, hele, solver)
    loadnonlinearproblem!(model, n, m+1, u_L, u_U, [X_L, n], [X_U, n], :Min, e)
    setwarmstart!(model, ones(n))
    MinimumDivergenceProblem(model, e)
end

function MinimumDivergenceProblem(G::AbstractMatrix, c::Vector;
                                  div::Divergence = KullbackLeibler(),
                                  solver = IpoptSolver(),
                                  k::SmoothingKernel = IdentitySmoother())
    n, m = size(G)
    m_eq = length(c)
    @assert m == m_eq "Inconsistent dimension"
    MinimumDivergenceProblem(G, c, c, m_eq, 0, div, solver, k)
end

function MinimumDivergenceProblem(G::AbstractMatrix, c::Vector,
                                  H::AbstractMatrix, lwr::Vector, upp::Vector;
                                  div::Divergence = KullbackLeibler(),
                                  solver = IpoptSolver(),
                                  k::SmoothingKernel = IdentitySmoother())

    m_c = length(c); m_lwr = length(lwr); m_upp = length(upp)
    n_g, m_g = size(G)
    n_h, m_h = size(H)
    @assert n_g == n_h "Dimensions of G and H are inconsistent"
    @assert m_lwr == m_upp "Dimensions of lower and upper bounds are inconsistent"

    @assert m_g == m_c "Dimensions of G and c are inconsistent"
    @assert m_h == m_lwr "Dimensions of bounds and H are inconsistent"
    m = m_g + m_lwr
    X_L = [c, lwr]
    X_U = [c, upp]
    MinimumDivergenceProblem([G H], X_L, X_U, m_g, m_h, div, solver, k)
end


function solve(mdp::MDProblem)
    resolve(mdp, _initial_x(mdp.m))
    return mdp
end

function resolve(mdp::MDProblem, π₀::Vector)
    n, m, m_eq, m_ineq = size(mdp)
    lambda0 = Array(Float64, m + 1)
    _resolve(mdp.m, π₀, lambda0)
end

function resolve(mdp::MDProblem, π₀::Vector, lambda0::Vector)
    _resolve(mdp.m, π₀, lambda0)
end


function _resolve(mmi::Ipopt.IpoptMathProgModel, x0::Vector)
    setwarmstart!(mmi, x0)
    optimize!(mmi)
end

function _resolve(mmi::Ipopt.IpoptMathProgModel, x0::Vector, lambda0::Vector)
    setwarmstart!(mmi, x0)
    optimize!(mmi)

end

function _resolve(mm::KNITRO.KnitroMathProgModel, x0::Vector)
    if status(mm) == :Uninitialized
        setwarmstart!(model, x0)
        optimize!(mm)
    else
        restartProblem(mm.inner, x0, mm.inner.numConstr)
        solveProblem(mm.inner)
    end
end

function _resolve(mm::KNITRO.KnitroMathProgModel, x0::Vector, lambda0::Vector)
    if status(mm) == :Uninitialized
        setwarmstart!(mm, x0)
        optimize!(mm)
    else
        restartProblem(mm.inner, x0, lambda0)
        solveProblem(mm.inner)
    end
end


_initial_x(mm::KNITRO.KnitroMathProgModel) =  mm.initial_x
_initial_x(mm::Ipopt.IpoptMathProgModel) =  mm.warmstart

_initial_lambda(mm::KNITRO.KnitroMathProgModel) =  mm.inner.lambda
_initial_lambda(mm::Ipopt.IpoptMathProgModel)   =  mm.warmstart


## function multistart(mdp::MinDivProb, ms_thetas::Array{Array{Float64, 1}, 1})
##     obj = fill!(Array(Float64, length(ms_thetas)), inf(1.0))
##     w = ones(nobs(mdp))
##     for i = 1:length(obj)
##         setwarmstart!(mdp.m, [ms_thetas[i], w])
##         MathProgBase.optimize!(mdp.m)
##         if MathProgBase.status(mdp)==:Optimal
##             obj[i] = getobjval(mdp)
##         end
##     end

##     if length(obj) > 0
##         opt = indmax(obj)
##         MathProgBase.setwarmstart!(mdp.m, [ms_thetas[opt], w])
##         solve(mdp)
##     end
##     return(mdp)
## end





status(mdp::MinimumDivergenceEstimator) = status(mdp.m)
status(mdp::MinimumDivergenceProblem)   = status(mdp.m)

status_plain(mdp::MinimumDivergenceEstimator) = mdp.m.inner.status
status_plain(mdp::MinimumDivergenceProblem)   = mdp.m.inner.status

getobjval(mdp::MinimumDivergenceEstimator) = getobjval(mdp.m)*scale_f(mdp)
getobjval(mdp::MinimumDivergenceProblem)   = getobjval(mdp.m)*scale_f(mdp)


getplainobjval(mdp::MinimumDivergenceEstimator) = getobjval(mdp.m)
getplainobjval(mdp::MinimumDivergenceProblem) = getobjval(mdp.m)

## Scaling for lagrange multipliers
#multscaling(mdp::MDEstimator) = mdp.e.momf.kern.κ₁/mdp.e.momf.kern.κ₂
#multscaling(mdp::MDProblem)   = mdp.e.mm.kern.κ₁/mdp.e.mm.kern.κ₂

kernel(mdp::MDEstimator) = mdp.e.momf.kern
kernel(mdp::MDProblem) = mdp.e.mm.kern

scale_l(mdp::MDEstimator) = κ₂(kernel(mdp))/κ₁(kernel(mdp))
scale_l(mdp::MDProblem)   = κ₂(kernel(mdp))/κ₁(kernel(mdp))

## Scaling for objective function
## 2.0/S*k1^2/k2
## for iid this reduces to 2.0
function scale_f(mdp::MDEstimator)
    k  = kernel(mdp)
    St = bw(k)
    k1 = κ₁(k)
    k2 = κ₂(k)
    2.0/(St*k1^2/k2)
end

function scale_f(mdp::MDProblem)
    k  = kernel(mdp)
    St = bw(k)
    k1 = κ₁(k)
    k2 = κ₂(k)
    2.0/(St*k1^2/k2)
end




npar(mdp::MinimumDivergenceEstimator) = mdp.e.momf.npar
nobs(mdp::MinimumDivergenceEstimator) = mdp.e.momf.nobs
nmom(mdp::MinimumDivergenceEstimator) = mdp.e.momf.nmom

nmom(mdp::MinimumDivergenceProblem) = mdp.m.nmom
nobs(mdp::MinimumDivergenceProblem) = mdp.m.nobs
npar(mdp::MinimumDivergenceProblem) = 0

nmom(m::KNITRO.KnitroMathProgModel) = m.numConstr-1
nmom(m::Ipopt.IpoptMathProgModel)   = m.inner.m-1

coef(mdp::MinimumDivergenceEstimator) = mdp.m.inner.x[nobs(mdp)+1:nobs(mdp)+npar(mdp)]

getmdweights(mdp::MinimumDivergenceEstimator) = mdp.m.inner.x[1:nobs(mdp)]
getmdweights(mdp::MinimumDivergenceProblem)   = mdp.m.inner.x[1:nobs(mdp)]

size(mdp::MinimumDivergenceEstimator)   = size(mdp.e.momf)


size(mdp::MinimumDivergenceProblem)  = size(mdp.e.mm)
size(e::SMDNLPE) = size(e.mm)
size(mm::MomentMatrix)  = (mm.n, mm.m, mm.m_eq, mm.m_ineq)


divergence(mdp::MinimumDivergenceEstimator) = mdp.e.div
divergence(mdp::MinimumDivergenceProblem)   = mdp.e.div

_getlambda(m::Ipopt.IpoptMathProgModel) = m.inner.mult_g[1:nmom(m)]
_geteta(m::Ipopt.IpoptMathProgModel)    = m.inner.mult_g[nmom(m)+1]

try
    _getlambda(m::KNITRO.KnitroMathProgModel) = m.inner.lambda[1:nmom(m)]
    _geteta(m::KNITRO.KnitroMathProgModel)    = m.inner.lambda[nmom(m)+1]
end

getlambda(mdp::MDEstimator) = scale_l(mdp).*_getlambda(mdp.m)
geteta(mdp::MDEstimator)    = scale_l(mdp).*_geteta(mdp.m)

df(mdp::MinimumDivergenceEstimator) = nmom(mdp)-npar(mdp)


##############################################################################
##
## Display results
##
##############################################################################

function Base.writemime{T<:MDEstimator}(io::IO, ::MIME"text/plain", me::T)
    # get coef table and j-test
    # show info for our model
    println(io, "\nMDEstimator: $(me.e.div), $(npar(me)) parameter(s) with $(nmom(me)) moment(s)\n")
    # Show extra information for this type
    stat = status(me)
    if me.e.solved == 1
        if stat == :Optimal
            ct = coeftable(me)
            # print coefficient table
            println(io, "Coefficients:\n")
            # then show coeftable
            show(io, ct)
            println("\nTesting H₀: E[g(x,θ₀)] = 0")
            print(io, show_extra(me))
        else
            println("The previous run of the optimization failed to locate a local maximum - $(stat)")
        end
    else
        println("θ₀ = ", round(me.m.inner.x[nobs(me)+1:end]))
    end
end

function LR_test(me::MDEstimator)
    j = [getobjval(me)]
    p = df(me) > 0 ? ccdf(Chisq(df(me)), j) : NaN
    # sometimes p is garbage, so we clamp it to be within reason
    return j[1], clamp(p, eps(), Inf)
end

function LM_test(me::MDEstimator, ver::Symbol = :weighted)
    l = getlambda(me)
    Ω = momf_var(me, ver)
    n = nobs(me)
    S = bw(kernel(me))
    k1= κ₁(kernel(me))
    k2= κ₂(kernel(me))
    j = (l'*Ω*l)/S^2
    p = df(me) > 0 ? ccdf(Chisq(df(me)), j) : NaN
    # sometimes p is garbage, so we clamp it to be within reason
    return j[1], clamp(p, eps(), Inf)
end

function J_test(me::MDEstimator, ver::Symbol = :weighted)
    g = me.e.momf.sn(coef(me))
    Ω = momf_var(me, ver)
    n = nobs(me)
    k1= κ₁(kernel(me))
    j = (g'*pinv(Ω)*g)/k1^2
    p = df(me) > 0 ? ccdf(Chisq(df(me)), j) : NaN
    # sometimes p is garbage, so we clamp it to be within reason
    return j[1], clamp(p, eps(), Inf)
end

function show_extra(me::MDEstimator)
    j1, p1 = MinimumDivergence.LR_test(me)
    j2, p2 = MinimumDivergence.LM_test(me)
    j3, p3 = MinimumDivergence.J_test(me)
    "\nLR-test: $(round(j1, 3)) (P-value: $(round(p1, 3)))
LM-test: $(round(j2, 3)) (P-value: $(round(p2, 3)))
J-test:  $(round(j3, 3)) (P-value: $(round(p3, 3)))\n"
end

function coeftable(mm::MinimumDivergenceEstimator, se::Vector)
    cc = coef(mm)
    @assert length(se)==length(cc)
    zz = cc ./ se
    CoefTable(hcat(cc,se,zz,2.0 * ccdf(Normal(), abs(zz))),
              ["Estimate","Std.Error","z value", "Pr(>|z|)"],
              ["θ$i" for i = 1:length(cc)], 4)
end

function coeftable(mm::MDEstimator, ver::Symbol)
    cc = coef(mm)
    se = stderr(mm, ver)
    coeftable(mm, se)
end

function coeftable(mm::MDEstimator)
    cc = coef(mm)
    se = stderr(mm, :weighted)
    zz = cc ./ se
    CoefTable(hcat(cc,se,zz,2.0 * ccdf(Normal(), abs(zz))),
              ["Estimate","Std.Error","z value", "Pr(>|z|)"],
              ["θ\_$i" for i = 1:length(cc)], 4)
end

## Dropin for IV only
function InstrumentalVariableMomentFunction(y::Vector, x::Matrix, z::Matrix)
    (nr,_) = size(y)
    yc = reshape(y, nr)
    InstrumentalVariableMomentFunction(yc, x::Matrix, z::Matrix)
end

function InstrumentalVariableMomentFunction(y::Matrix, x::Matrix, z::Matrix)
    nobs, nmom = size(z)
    (_,npar) = size(x)
    _ivderiv = -z'*x
    g(θ) = z.*(y-x*θ)
    s(θ) = z.*(y-x*θ)
    sn(θ) = sum(g(θ), 1)
    sw(θ, p) = g(θ)'*p

    function ∂sw(θ::Vector)
        -(__p.*z)'*x
    end

    ∂sn(θ) = _ivderiv

    ∂sl(θ::Vector) = -(z*__λ).*x
    ∂swl(θ::Vector) = -(z*__λ)'*(__p.*x)
    ∂²swl(θ::Vector) = zeros(npar, npar)

    MomentFunction( g, g, sw, sn, ∂sw, ∂sn, ∂sl, ∂swl, ∂²swl, IdentitySmoother(), nobs, nmom, npar)
end

## Simplified interface for IV
## To be refined........

type InstrumentalVariableModel
    y::Array{Float64, 2}
    x::Array{Float64, 2}
    z::Array{Float64, 2}
    Pz::Array{Float64, 2}
    k::SmoothingKernel
end

typealias IV InstrumentalVariableModel



function IV(y::Array{Float64, 2}, x::Array{Float64, 2} ,z::Array{Float64, 2})
    zz = PDMat(z'z)
    Pz = X_invA_Xt(zz, z)
    IV(y, x, z, Pz, IdentitySmoother())
end

function MinDivProb(iv::InstrumentalVariableModel, div::Divergence,
                    θ₀::Vector, lb::Vector, ub::Vector; solver = IpoptSolver())
    MinDivProb(InstrumentalVariableMomentFunction(iv.y, iv.x, iv.z), div, θ₀, lb, ub, solver = solver)
end

function MinDivProb(iv::InstrumentalVariableModel, div::Divergence; solver = IpoptSolver())
    θ = ivreg(iv.y, iv.x, iv.z)
    lb = ones(Float64, length(θ)).*(θ-20.)
    ub = ones(Float64, length(θ)).*(θ+20.)
    MinDivProb(InstrumentalVariableMomentFunction(iv.y, iv.x, iv.z), div, θ, lb, ub, solver = solver)
end

function MinDivProb(iv::InstrumentalVariableModel, div::Divergence, θ::Vector, π::Vector; solver = IpoptSolver())

    lb = ones(Float64, length(θ)).*(θ-20.)
    ub = ones(Float64, length(θ)).*(θ+20.)
    MinDivProb(InstrumentalVariableMomentFunction(iv.y, iv.x, iv.z), div, θ, lb, ub, π, solver = solver)
end

function MinDivProb(iv::InstrumentalVariableModel, div::Divergence, θ::Vector,
                    lb::Vector, ub::Vector, π::Vector; solver = IpoptSolver())
    MinDivProb(InstrumentalVariableMomentFunction(iv.y, iv.x, iv.z), div, θ, lb, ub, π, solver = solver)
end



function ivreg(y, x, z)
    zz = PDMat(z'z)
    Pz = X_invA_Xt(zz, z)
    xPz= x'*Pz
    reshape(xPz*x\xPz*y, size(x)[2])
end
