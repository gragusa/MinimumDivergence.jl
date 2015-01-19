############################
## To be refined....
##
############################
abstract MinimumDivergenceProblems

typealias MDPS MinimumDivergenceProblems

type MinDivProb <: MinimumDivergenceProblems
    model::MathProgBase.AbstractMathProgModel
    mdnlpe::MinDivNLPEvaluator
    Vʷ::Union(Nothing, PDMat)
    Vᴴ::Union(Nothing, PDMat)
    H::Union(Nothing, PDMat)
end

immutable SMinDivProb <: MinimumDivergenceProblems
    model::MathProgBase.AbstractMathProgModel
    mdnlpe::SMinDivNLPEvaluator
end

function MinDivProb(g_eq::AbstractMatrix, g_ineq::AbstractMatrix,
                    div::Divergence, g_L_ineq::Vector, g_U_ineq::Vector;
                    solver=IpoptSolver())
    n_eq, m_eq  = size(g_eq)
    n_ineq, m_ineq = size(g_ineq)
    @assert n_eq == n_ineq
    @assert length(g_L_ineq) == length(g_U_ineq)
    @assert length(g_L_ineq) == m_ineq
    model = MathProgBase.MathProgSolverInterface.model(solver)

    m = m_eq + m_ineq
    n = n_eq

    gele = int(n*(m + 1))
    hele = int(n)

    g_L = [zeros(m_eq), g_L_ineq, n];
    g_U = [zeros(m_eq), g_U_ineq, n];

    u_L = [zeros(n)];
    u_U = [ones(n)*n];

    mdnlpe = SMDNLPE([g_eq g_ineq], div, n, m_eq, m_ineq, m, gele, hele)
    loadnonlinearproblem!(model, n, m + 1, u_L, u_U, g_L, g_U, :Min, mdnlpe)
    setwarmstart!(model, ones(n))
    SMinDivProb(model, mdnlpe)
end

function MinDivProb(mm::MomentMatrix, div::Divergence; solver = IpoptSolver())
    model = MathProgBase.MathProgSolverInterface.model(solver)
    n, m = size(mm)
    gele = int(n*(m+1))
    hele = int(n)
    g_L = [mm.g_L, n]
    g_U = [mm.g_U, n]
    u_L = [zeros(n)];
    u_U = [ones(n)*n];
    mdnlpe = SMDNLPE(mm, div, n, mm.m_eq, mm.m_ineq, m, gele, hele, solver)
    loadnonlinearproblem!(model, n, m+1, u_L, u_U, g_L, g_U, :Min, mdnlpe)
    setwarmstart!(model, ones(n))
    SMinDivProb(model, mdnlpe)
end

function MinDivProb(mf::MomentFunction, div::Divergence, θ₀::Vector,
                    lb::Vector, ub::Vector; solver=IpoptSolver())
    model = MathProgBase.MathProgSolverInterface.model(solver)
    m = mf.nmom
    n = mf.nobs
    k = mf.npar
    u₀ = [ones(n), θ₀]
    gele = int((n+k)*(m+1)-k)
    hele = int(n*k + n + (k+1)*k/2)
    g_L = [zeros(m), n];
    g_U = [zeros(m), n];
    u_L = [zeros(n),  lb];
    u_U = [ones(n)*n, ub];
    mdnlpe = MDNLPE(mf, div, n, m, k, gele, hele, solver,
                    Array(Float64, n), Array(Float64, m+1))
    loadnonlinearproblem!(model, n+k, m+1, u_L, u_U, g_L, g_U, :Min, mdnlpe)
    setwarmstart!(model, u₀)
    MinDivProb(model, mdnlpe, Nothing(), Nothing(), Nothing())
end


function MinDivProb(mf::MomentFunction, div::Divergence, θ₀::Vector,
                    lb::Vector, ub::Vector, π::Vector; solver=IpoptSolver())
    model = MathProgBase.MathProgSolverInterface.model(solver)
    m = mf.nmom
    n = mf.nobs
    k = mf.npar
    u₀ = [π, θ₀]
    gele = int((n+k)*(m+1)-k)
    hele = int(n*k + n + (k+1)*k/2)
    g_L = [zeros(m), n];
    g_U = [zeros(m), n];
    u_L = [zeros(n),  lb];
    u_U = [ones(n)*n, ub];
    mdnlpe = MDNLPE(mf, div, n, m, k, gele, hele, solver,
                    Array(Float64, n), Array(Float64, m+1))
    loadnonlinearproblem!(model, n+k, m+1, u_L, u_U, g_L, g_U, :Min, mdnlpe)
    setwarmstart!(model, u₀)
    MinDivProb(model, mdnlpe, Nothing(), Nothing(), Nothing())
end

function solve(mdp::MinDivProb)
    resolve(mdp, _initial_x(mdp.model))
    if status(mdp)==:Optimal
        vcov!(mdp)
    end
    return mdp
end

function solve(mdp::SMinDivProb)
    resolve(mdp, _initial_x(mdp.model))    
    return mdp
end

function resolve(mdp::MinimumDivergenceProblems, x0::Array{Float64,1})
    lambda0 = Array(Float64, nobs(mdp) + nmom(mdp) + npar(mdp) + 1)
    if length(x0) == nobs(mdp)+npar(mdp)
        _resolve(mdp.model, x0, lambda0)
    elseif length(x0) > 0 && length(x0) == npar(mdp)
        _resolve(mdp.model, [ones(nobs(mdp)), x0], lambda0)
    else
        error("Cannot resolve the problem. Check dimension of 'x0'")
    end 
end

function resolve(mdp::MinimumDivergenceProblems, x0::Array{Float64,1}, lambda0::Array{Float64,1})
    @assert length(lambda0) == nobs(mdp) + nmom(mdp) + npar(mdp) + 1
    if length(x0) == nobs(mdp)+npar(mdp)
        _resolve(mdp.model, x0, lambda0)
    elseif length(x0) > 0 && length(x0) == npar(mdp)
        _resolve(mdp.model, [ones(nobs(mdp)) x0], lambda0)
    else
        error("Cannot resolve the problem. Check dimension of 'x0'")
    end 
end


function _resolve(mmi::Ipopt.IpoptMathProgModel, x0::Array{Float64,1})
    setwarmstart!(mmi, x0)
    optimize!(mmi)    
end 

function _resolve(mmi::Ipopt.IpoptMathProgModel, x0::Array{Float64,1}, lambda0::Array{Float64,1})
    setwarmstart!(mmi, x0)
    optimize!(mmi)
end 

function _resolve(mm::KNITRO.KnitroMathProgModel, x0::Array{Float64,1})
    if status(mm) == :Uninitialized
        setwarmstart!(model, x)
        optimize!(mm)
    else
        restartProblem(mm.inner, x0, mm.inner.numConstr)
        solveProblem(mm.inner)
    end
end

function _resolve(mm::KNITRO.KnitroMathProgModel, x0::Array{Float64,1}, lambda0::Array{Float64,1})
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
##         setwarmstart!(mdp.model, [ms_thetas[i], w])
##         MathProgBase.optimize!(mdp.model)
##         if MathProgBase.status(mdp)==:Optimal
##             obj[i] = getobjval(mdp)
##         end
##     end

##     if length(obj) > 0
##         opt = indmax(obj)
##         MathProgBase.setwarmstart!(mdp.model, [ms_thetas[opt], w])
##         solve(mdp)
##     end
##     return(mdp)
## end






status(mdp::MDPS)       = status(mdp.model)
status_plain(mdp::MDPS) = mdp.model.inner.status

getobjval(mdp::MDPS)    = getobjval(mdp.model)*objscaling(mdp)

multscaling(mdp::MinDivProb)  = mdp.mdnlpe.momf.kern.κ₁/mdp.mdnlpe.momf.kern.κ₂
multscaling(mdp::SMinDivProb) = mdp.mdnlpe.mm.kern.κ₁/mdp.mdnlpe.mm.kern.κ₂
objscaling(mdp::MinDivProb)   = mdp.mdnlpe.momf.kern.scale
objscaling(mdp::SMinDivProb)  = mdp.mdnlpe.mm.kern.scale

nobs(mdp::MDPS)         = mdp.mdnlpe.nobs
npar(mdp::MinDivProb)   = mdp.mdnlpe.npar
nmom(mdp::MDPS)         = mdp.mdnlpe.nmom

nmom(m::KNITRO.KnitroMathProgModel) = m.numConstr-1
nmom(m::Ipopt.IpoptMathProgModel) = m.inner.m-1

npar(m::SMinDivProb) = 0



coef(mdp::MinDivProb)   = mdp.model.inner.x[nobs(mdp)+1:nobs(mdp)+npar(mdp)]
getmdweights(mdp::MDPS) = mdp.model.inner.x[1:nobs(mdp)]

size(mdp::MinDivProb)   = (mdp.mdnlpe.nobs, mdp.mdnlpe.nmom, mdp.mdnlpe.npar)
size(mdp::SMinDivProb)  = (mdp.mdnlpe.nobs, mdp.mdnlpe.nmom)
size(mm::MomentMatrix)  = size(mm.g)
divergence(mdp::MDPS)   = mdp.mdnlpe.div

_getlambda(m::Ipopt.IpoptMathProgModel)  = m.inner.mult_g[1:nmom(m)]
_geteta(m::Ipopt.IpoptMathProgModel)  = m.inner.mult_g[nmom(m)+1]

try 
    _getlambda(m::KNITRO.KnitroMathProgModel) = m.inner.lambda[1:nmom(m)]
    _geteta(m::KNITRO.KnitroMathProgModel)    = m.inner.lambda[nmom(m)+1]
end 

getlambda(mdp::MDPS) = multscaling(mdp).*_getlambda(mdp.model)
geteta(mdp::MDPS)    = multscaling(mdp).*_geteta(mdp.model)


function coeftable(mm::MinDivProb)
    cc = coef(mm)
    se = stderr(mm)
    zz = cc ./ se
    CoefTable(hcat(cc,se,zz,2.0 * ccdf(Normal(), abs(zz))),
              ["Estimate","Std.Error","z value", "Pr(>|z|)"],
              ["θ\_$i" for i = 1:length(cc)], 4)
end


function coeftable(mm::MinDivProb, se::Vector)
    cc = coef(mm)
    @assert length(se)==length(cc)
    zz = cc ./ se
    CoefTable(hcat(cc,se,zz,2.0 * ccdf(Normal(), abs(zz))),
              ["Estimate","Std.Error","z value", "Pr(>|z|)"],
              ["θ$i" for i = 1:length(cc)], 4)
end

function coeftable(mm::MinDivProb, ver::Symbol)
    cc = coef(mm)
    se = stderr(mm, ver)
    zz = cc ./ se
    CoefTable(hcat(cc,se,zz,2.0 * ccdf(Normal(), abs(zz))),
              ["Estimate","Std.Error","z value", "Pr(>|z|)"],
              ["θ$i" for i = 1:length(cc)], 4)
end



## Simplified API

## type MomentFunction
##   gᵢ::Function        ## Moment Function
##   sᵢ::Function        ## Smoothed moment function
##   wsn::Function       ## (m×1) ∑pᵢ sᵢ
##   sn::Function        ## ∑sᵢ(θ)
##   ∂∑pᵢsᵢ::Function    ## (k×m)
##   ∂∑sᵢ::Function      ## (k×m)
##   ∂sᵢλ::Function      ## (n×k)
##   ∑pᵢ∂sᵢλ::Function   ## (k×1)
##   ∂²sᵢλ::Function     ## (kxk)
##   kern::SmoothingKernel
##   nobs::Int64
##   nmom::Int64
##   npar::Int64
## end



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
    
    MomentFunction( g, g, sw, sn, ∂sw, ∂sn, ∂sl, ∂swl, ∂²swl, IdentityKernel(), nobs, nmom, npar)
end

## Simplified interface for IV
## To be refined........

type InstrumentalVariableModel
    y::Array{Float64, 2}
    x::Array{Float64, 2}
    z::Array{Float64, 2}
    k::SmoothingKernel
end

typealias IV InstrumentalVariableModel 

IV(y::Array{Float64, 2}, x::Array{Float64, 2} ,z::Array{Float64, 2}) = IV(y, x, z, IdentityKernel())

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





    
    
    
    


