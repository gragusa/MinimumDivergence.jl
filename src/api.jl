############################
## To be refined....
##
############################
function mdtest(mf::MomentFunction,
                div::Divergence, 
                θ₀::Vector, 
                lb::Vector, ub::Vector;
                solver=IpoptSolver())

    model = MathProgSolverInterface.model(solver)

    m = mf.nmom
    n = mf.nobs 
    k = mf.npar

    u₀ = [ones(n), θ₀]

    gele = int((n+k)*(m+1)-k)
    hele = int(n*k + n + (k+1)*k/2)

    # lb = [-2., -2]
    # ub = [2, 2.]
    g_L = [zeros(m), n];
    g_U = [zeros(m), n];

    u_L = [zeros(n),  lb];
    u_U = [ones(n)*n, ub];
    # l = [1,1,1,1]
    # u = [5,5,5,5]
    # lb = [25, 40]
    # ub = [Inf, 40]

    MdProb = MinimumDivergenceProblem(mf, div, n, m, k, gele, hele, Array(Float64, n), Array(Float64, m+1)) 
    

    MathProgSolverInterface.loadnonlinearproblem!(model, n+k, m+1, u_L, u_U, g_L, g_U, :Min, MdProb)
    
    MathProgSolverInterface.setwarmstart!(model, u₀)

    MathProgSolverInterface.optimize!(model)
    stat = MathProgSolverInterface.status(model)

    # @test stat == :Optimal
    uᵒ = MathProgSolverInterface.getsolution(model)
    Qᵒ = MathProgSolverInterface.getobjval(model) 
    (model, uᵒ, Qᵒ)

end
