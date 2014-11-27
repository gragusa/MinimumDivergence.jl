smooth(g::AbstractMatrix, k::IdentityKernel) = g
smooth(g::AbstractMatrix, k::SmoothingKernel) = k.smoother(g, k.ξ)

nobs(mf::MomentFunction) = mf.nobs
nmom(mf::MomentFunction) = mf.nmom
npar(mf::MomentFunction) = mf.npar

function get_mom_deriv(g::Function, dtype::Symbol, k::SmoothingKernel, nobs, nmom, npar)
  
  s(θ::Vector)             = smooth(g(θ), k)	
  gn(θ::Vector, p::Vector) = s(θ)'*p	
  g1(θ::Vector, λ::Vector) = s(θ)*λ
  
  g2(θ::Vector, p::Vector, λ::Vector) = (p'*s(θ)*λ)[1]
  
  gn_closure(θ::Vector) = gn(θ, __p)
  g1_closure(θ::Vector) = g1(θ, __λ)
  g2_closure(θ::Vector) = g2(θ, __p, __λ)
  
  gn_closure!(θ::Vector, gg) = gg[:] = gn(θ, __p)
  g1_closure!(θ::Vector, gg) = gg[:] = g1(θ, __λ)
  
  if dtype==:typed
  	∂gn  = ForwardDiff.forwarddiff_jacobian(gn_closure, Float64, fadtype=:typed)
  	∂g1  = ForwardDiff.forwarddiff_jacobian(g1_closure, Float64, fadtype=:typed)
  	∂²g2 = ForwardDiff.forwarddiff_hessian(g2_closure, Float64, fadtype=:typed)
  elseif dtype==:dual
  	∂gn  = ForwardDiff.forwarddiff_jacobian(gn_closure!, Float64, fadtype=:dual, n = npar, m = nmom)
  	∂g1  = ForwardDiff.forwarddiff_jacobian(g1_closure!, Float64, fadtype=:dual, n = npar, m = nobs)
  	∂²g2 = ForwardDiff.forwarddiff_hessian(g2_closure, Float64, fadtype=:typed)
  elseif dtype==:diff
  	∂gn(θ::Vector)  = Calculus.jacobian(gn_closure, θ, :central)
  	∂g1(θ::Vector)  = Calculus.jacobian(g1_closure, θ, :central)
  	∂²g2(θ::Vector) = Calculus.hessian(g2_closure, θ, :central)
  end
  return (g, s, gn, ∂gn, ∂g1, ∂²g2)
end 

function MomentFunction(g::Function, dtype::Symbol; nobs = Nothing, npar = Nothing, nmom = Nothing)	  
  ## Default is no smoothing
  MomentFunction(get_mom_deriv(g, dtype, IdentityKernel(), nobs, nmom, npar)..., 
                                IdentityKernel(), nobs, nmom, npar)
end

function MomentFunction(g::Function, dtype::Symbol, k::SmoothingKernel; 
                         nobs = Nothing, npar = Nothing, nmom = Nothing)  	  
  ## Default is no smoothing
  MomentFunction(get_mom_deriv(g, dtype, k, nobs, nmom, npar)..., k, nobs, nmom, npar)
end