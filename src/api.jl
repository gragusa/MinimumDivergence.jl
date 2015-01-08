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

function solve(mdp::MinDivProb)
    optimize!(mdp.model)
    if status(mdp)==:Optimal
        vcov!(mdp)
    end
    return mdp
end

function multistart(mdp::MinDivProb, ms_thetas::Array{Array{Float64, 1}, 1})
    obj = fill!(Array(Float64, length(ms_thetas)), inf(1.0))
    w = ones(nobs(mdp))
    for i = 1:length(obj)
        setwarmstart!(mdp.model, [ms_thetas[i], w])
        MathProgBase.optimize!(mdp.model)
        if MathProgBase.status(mdp)==:Optimal
            obj[i] = getobjval(mdp)
        end
    end

    if length(obj) > 0
        opt = indmax(obj)
        MathProgBase.setwarmstart!(mdp.model, [ms_thetas[opt], w])
        solve(mdp)
    end
    return(mdp)
end

function solve(mdp::SMinDivProb)
    optimize!(mdp.model)
    return mdp
end

status(mdp::MDPS)       = status(mdp.model)
status_plain(mdp::MDPS) = mdp.model.inner.status

getobjval(mdp::MDPS)    = getobjval(mdp.model)*objscaling(mdp)

multscaling(mdp::MinDivProb)  = mdp.mdnlpe.momf.kern.κ₁/mdp.mdnlpe.momf.kern.κ₂
multscaling(mdp::SMinDivProb) = mdp.mdnlpe.mm.kern.κ₁/mdp.mdnlpe.mm.kern.κ₂
objscaling(mdp::MinDivProb)   = mdp.mdnlpe.momf.kern.scale
objscaling(mdp::SMinDivProb)  = mdp.mdnlpe.momf.kern.scale

nobs(mdp::MDPS)         = mdp.mdnlpe.nobs
npar(mdp::MinDivProb)   = mdp.mdnlpe.npar
nmom(mdp::MDPS)         = mdp.mdnlpe.nmom
getlambda(mdp::MDPS)    = multscaling(mdp).*mdp.model.inner.mult_g[1:nmom(mdp)]
geteta(mdp::MDPS)       = multscaling(mdp).*mdp.model.inner.mult_g[nmom(mdp)+1]
coef(mdp::MinDivProb)   = mdp.model.inner.x[nobs(mdp)+1:nobs(mdp)+npar(mdp)]
getmdweights(mdp::MDPS) = mdp.model.inner.x[1:nobs(mdp)]

size(mdp::MinDivProb)   = (mdp.mdnlpe.nobs, mdp.mdnlpe.nmom, mdp.mdnlpe.npar)
size(mdp::SMinDivProb)  = (mdp.mdnlpe.nobs, mdp.mdnlpe.nmom)
size(mm::MomentMatrix)  = size(mm.g)
divergence(mdp::MDPS)   = mdp.mdnlpe.div

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
