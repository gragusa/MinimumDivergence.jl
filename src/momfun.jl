smooth{T}(g::Array{T, 2}, k::IdentityKernel) = g
smooth{T}(g::Array{T, 2}, k::SmoothingKernel) = k.smoother(g)

nobs(mf::MomentFunction) = mf.nobs
nmom(mf::MomentFunction) = mf.nmom
npar(mf::MomentFunction) = mf.npar

function get_mom_deriv(g::Function, dtype::Symbol, k::SmoothingKernel, nobs, nmom, npar)
    ## Smoothed moment condition
    s(θ::Vector)  = smooth(g(θ), k)
    ## Average moment condition
    sn(θ::Vector)  = mean(s(θ), 1)    
    ## Weighted moment conditions 
    ws(θ::Vector, p::Vector) = s(θ)'*p 
    sl(θ::Vector, λ::Vector) = s(θ)*λ 
    wsl(θ::Vector, p::Vector, λ::Vector) = (p'*s(θ)*λ)[1]
    ## Weighted moment conditions (inplace version)
    ws!(θ::Vector, jac_out, p::Vector) = jac_out[:] = s(θ)'*p 
    sl!(θ::Vector, jac_out, λ::Vector) = jac_out[:] =s(θ)*λ 
    wsl!(θ::Vector, jac_out,  p::Vector, λ::Vector) = jac_out[:] = (p'*s(θ)*λ)[1]        
    ## Smoothed average moment conditions
    sn(θ::Vector) = sum(s(θ), 1)

    if dtype==:typed
        ## First derivative
        Dsn  = ForwardDiff.dual_fad_gradient(sn,  Float64)
        Dws  = args_dual_fad_gradient(sw,  Float64)
        Dsl  = args_dual_fad_gradient(sl,  Float64)
        Dwsl = args_dual_fad_gradient(wsl, Float64)
        ## Second derivative
        Hwsl = args_typed_fad_hessian(wsl, Float64)
    elseif dtype==:typed
        Dsn  = ForwardDiff.typed_fad_gradient(sn,  Float64)
        Dws  = args_typed_fad_gradient(sw,  Float64)
        Dsl  = args_typed_fad_gradient(sl,  Float64)
        Dwsl = args_typed_fad_gradient(wsl, Float64)
        ## Second derivative
        Hwsl = args_typed_fad_hessian(wsl, Float64)
    elseif dtype==:diff
        Dsn(θ::Vector, args...)  = fd_jacobian(sn,  θ, :central, args...)
        Dws(θ::Vector, args...)  = fd_jacobian(ws,  θ, :central, args...)
        Dsl(θ::Vector, args...)  = fd_jacobian(sl,  θ, :central, args...)
        Dwsl(θ::Vector, args...) = fd_jacobian(wsl, θ, :central, args...)
        Hwsl(θ::Vector, args...) = fd_hessian(wsl,  θ, :central, args...)
    end

    return (g, s, ws, sn, Dws, Dsn, Dsl, Dwsl, Hwsl)
end 


function MomentMatrix(g_eq::AbstractMatrix,
                      g_ineq::AbstractMatrix,
                      g_L_ineq::Vector,
                      g_U_ineq::Vector,
                      ker::SmoothingKernel)
    n_eq, m_eq  = size(g_eq)
    n_ineq, m_ineq = size(g_ineq)
    @assert n_eq==n_ineq
    @assert length(g_L_ineq)==length(g_U_ineq)
    @assert length(g_L_ineq)==m_ineq
    g_L = [zeros(m_eq), g_L_ineq];
    g_U = [zeros(m_eq), g_U_ineq];
    MomentMatrix([g_eq g_ineq], g_L, g_U, ker, m_eq, m_ineq)    
end

function MomentMatrix(g_eq::AbstractMatrix, ker::SmoothingKernel)
    n_eq, m_eq  = size(g_eq)
    n_ineq, m_ineq = (n_eq, 0)
    g_L = [zeros(m_eq)];
    g_U = [zeros(m_eq)];
    MomentMatrix(g_eq, g_L, g_U, ker, m_eq, m_ineq)
end

function MomentMatrix(g_eq::AbstractMatrix)
    n_eq, m_eq  = size(g_eq)
    n_ineq, m_ineq = (n_eq, 0)
    g_L = [zeros(m_eq)];
    g_U = [zeros(m_eq)];
    MomentMatrix(g_eq, g_L, g_U, IdentityKernel(), m_eq, m_ineq)
end


function MomentMatrix(g_ineq::AbstractMatrix,
                      g_L_ineq::Vector,
                      g_U_ineq::Vector,
                      ker::SmoothingKernel)
    n_ineq, m_ineq = size(g_ineq)
    n_eq, m_eq  = (n_ineq, 0)
    @assert length(g_L_ineq)==length(g_U_ineq)
    @assert length(g_L_ineq)==m_ineq
    MomentMatrix(g_eq, g_L_ineq, g_U_ineq, ker, m_eq, m_ineq)
end

function MomentFunction(g::Function, dtype::Symbol; nobs = Nothing,
                        npar = Nothing, nmom = Nothing)
    ## Default is no smoothing
    MomentFunction(get_mom_deriv(g, dtype, IdentityKernel(),
                                 nobs, nmom, npar)...,
                   IdentityKernel(), nobs, nmom, npar)
end

function MomentFunction(g::Function, dtype::Symbol, k::SmoothingKernel;
                        nobs = Nothing, npar = Nothing, nmom = Nothing)
  ## Default is no smoothing
    MomentFunction(get_mom_deriv(g, dtype, k, nobs, nmom, npar)...,
                   k, nobs, nmom, npar)
end
