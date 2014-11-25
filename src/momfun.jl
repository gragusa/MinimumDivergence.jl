type MomentFunction
  gᵢ::Function      ## Moment Function
  sᵢ::Function      ## Smoothed moment function
  ∂∑pᵢsᵢ::Function  ## k x m
  ∂sᵢλ::Function    ## n x k 
  ∂²sᵢλ::Functio    ## k x k
end  


smooth(g::AbstractMatrix, k::IdentityKernel) = g


get_mom_deriv(g::Function, dtype::Symbol, k::Kernel)
  
  s(θ::Vector) = smooth(g(θ), k)	
  gn(θ::Vector, p::Vector) = s(θ)'*p	
  g1(θ::Vector, λ::Vector) = s(θ)*λ
  g2(θ::Vector, p::Vector, λ::Vector) = (p'*s(θ)*λ)[1]
  
  gn_closure(θ::Vector) = gn(θ, __p)
  g1_closure(θ::Vector) = g1(θ, __λ)
  g2_closure(θ::Vector) = g2(θ, __p, __λ)
  
  gn_closure!(θ::Vector, gg) = gg[:] = gn(θ, __p)
  g1_closuer!(θ::Vector, gg) = gg[:] = g1(θ, __p, __λ)
  
  if dtype==:typed
  	∂gn  = forwarddiff_jacobian(gn_closure, Float64, fadtype=:typed)
  	∂g1  = forwarddiff_jacobian(g1_closure, Float64, fadtype=:typed)
  	∂²g2 = forwarddiff_hessian(g2_closure, Float64, fadtype=:typed)
  elseif dtype==:dual
  	∂gn  = forwarddiff_jacobian(gn_closure!, Float64, fadtype=:dual, n = nvars, m = nmom)
  	∂g1  = forwarddiff_jacobian(g1_closure!, Float64, fadtype=:dual, n = nvars, m = nmom)
  	∂²g2 = forwarddiff_hessian(g2_closure, Float64, fadtype=:typed)
  elseif dtype==:diff
  	∂gn(θ::Vector)  = Calculus.jacobian(gn_closure, θ, :central)
  	∂g1(θ::Vector)  = Calculus.jacobian(g1_closure, θ, :central)
  	∂²g2(θ::Vector) = Calculus.hessian(g2_closure, θ, :central)
  end
  return (g, s, ∂gn, ∂g1, ∂²g2)
end 


function MomentFunction(g::Function, dtype::Symbol)	  
  ## Default is no smoothing
  MomentFunction(get_mom_deriv(g, dtype, IdentityKernel())..., IdentityKernel())
end

function MomentFunction(g::Function, dtype::Symbol, k::Kernel)	  
  ## Default is no smoothing
  MomentFunction(get_mom_deriv(g, dtype, k)..., k)
end








function pgl(p, g, l) 
	N, m = size(g)
	a = 0.0
	for j = 1:m
		for i = 1:N
			a += p[i]*g[i,j]*l[j]
		end 
	end 
	return a
end 