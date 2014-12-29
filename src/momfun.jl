smooth{T}(g::Array{T, 2}, k::IdentityKernel) = g
smooth{T}(g::Array{T, 2}, k::SmoothingKernel) = k.smoother(g)

nobs(mf::MomentFunction) = mf.nobs
nmom(mf::MomentFunction) = mf.nmom
npar(mf::MomentFunction) = mf.npar

function get_mom_deriv(g::Function, dtype::Symbol, k::SmoothingKernel, nobs, nmom, npar)

  s(θ::Vector)  = smooth(g(θ), k)
  sn(θ::Vector) = sum(s(θ), 1)

  sw(θ::Vector, p::Vector) = s(θ)'*p
  sl(θ::Vector, λ::Vector) = s(θ)*λ
  swl(θ::Vector, p::Vector, λ::Vector) = (p'*s(θ)*λ)[1]

  sw_closure(θ::Vector) = sw(θ, __p)
  sl_closure(θ::Vector) = sl(θ, __λ)
  swl_closure(θ::Vector) = swl(θ, __p, __λ)

  sn!(θ::Vector, gg) = gg[:] = sn(θ)
  sw_closure!(θ::Vector, gg) = gg[:] = sw(θ, __p)
  sl_closure!(θ::Vector, gg) = gg[:] = sl(θ, __λ)
  swl_closure!(θ::Vector, gg) = gg[:] = swl(θ, __p, __λ)

  # gn(θ::Vector, p::Vector) = s(θ)'*p
  # g1(θ::Vector, λ::Vector) = s(θ)*λ
  # g2(θ::Vector, p::Vector, λ::Vector) = (p'*s(θ)*λ)[1]

  # gn_closure(θ::Vector) = gn(θ, __p)
  # g1_closure(θ::Vector) = g1(θ, __λ)
  # g2_closure(θ::Vector) = g2(θ, __p, __λ)

  # sn!(θ::Vector, gg) = gg[:] = sn(θ)
  # gn_closure!(θ::Vector, gg) = gg[:] = gn(θ, __p)
  # g1_closure!(θ::Vector, gg) = gg[:] = g1(θ, __λ)
  # g2_closure!(θ::Vector, gg) = gg[:] = g2(θ, __p, __λ)

  if dtype==:typed
  	∂sw   = ForwardDiff.forwarddiff_jacobian(sw_closure, Float64, fadtype=:typed)
    ∂sn   = ForwardDiff.forwarddiff_jacobian(sn, Float64, fadtype=:typed)
  	∂sl   = ForwardDiff.forwarddiff_jacobian(sl_closure, Float64, fadtype=:typed)
    ∂swl  = ForwardDiff.forwarddiff_jacobian(swl_closure, Float64, fadtype=:typed)
  	∂²swl = ForwardDiff.forwarddiff_hessian(swl_closure, Float64, fadtype=:typed)
  elseif dtype==:dual
  	∂sw  = ForwardDiff.forwarddiff_jacobian(sw_closure!, Float64, fadtype=:dual,
      n = npar, m = nmom)
    ∂sn  = ForwardDiff.forwarddiff_jacobian(sn!, Float64, fadtype=:dual,
      n = npar, m = nmom)
  	∂sl  = ForwardDiff.forwarddiff_jacobian(sl_closure!, Float64, fadtype=:dual,
      n = npar, m = nobs)
    ∂swl = ForwardDiff.forwarddiff_jacobian(swl_closure!, Float64, fadtype=:dual,
      n = npar, m = 1)
  	∂²swl = ForwardDiff.forwarddiff_hessian(swl_closure, Float64, fadtype=:typed)
  elseif dtype==:diff
  	∂sw(θ::Vector)   = Calculus.jacobian(sw_closure, θ, :central)
    ∂sn(θ::Vector)   = Calculus.jacobian(sn, θ, :central)
  	∂sl(θ::Vector)   = Calculus.jacobian(sl_closure, θ, :central)
    ∂swl(θ::Vector)  = Calculus.jacobian(swl_closure, Float64, :central)
  	∂²swl(θ::Vector) = Calculus.hessian(swl_closure, θ, :central)
  end
  return (g, s, sw, sn, ∂sw, ∂sn, ∂sl, ∂swl, ∂²swl)
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

function MomentMatrix(g_eq::AbstractMatrix,
                         ker::SmoothingKernel)
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
  MomentFunction(get_mom_deriv(g, dtype, IdentityKernel(), nobs, nmom, npar)...,
                                IdentityKernel(), nobs, nmom, npar)
end

function MomentFunction(g::Function, dtype::Symbol, k::SmoothingKernel;
                         nobs = Nothing, npar = Nothing, nmom = Nothing)
  ## Default is no smoothing
  MomentFunction(get_mom_deriv(g, dtype, k, nobs, nmom, npar)..., k, nobs, nmom, npar)
end