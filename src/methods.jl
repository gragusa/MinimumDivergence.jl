
# bandwidth(md::MinimumDivergenceProblem) = md.mf.sf.St
# κ₁(md::MinimumDivergenceProblem) = md.mf.sf.κ₁
# κ₂(md::MinimumDivergenceProblem) = md.mf.sf.κ₂

# bandwidth(mf::MomentFunction) = mf.sf.St
# κ₁(mf::MomentFunction) = mf.sf.κ₁
# κ₂(mf::MomentFunction) = mf.sf.κ₂


# nobs(md::MDP) = md.n
# ncond(md::MDP) = md.m

# npar(md::MinimumDivergenceProblem) = md.p
# npar(md::MinimumDivergenceProblemPlain) = 0.0

# lambda(md::MinimumDivergenceProblem) = md.lambda
# coef(md::MinimumDivergenceProblem) = md.fprob.x[nobs(md)+1:end]
# weights(md::MDP) = md.fprob.x[1:nobs(md)]
# obj_val(md::MDP) = md.fprob.obj_val     ## Should I multiply this by 2?


# function vcov!(md::MinimumDivergenceProblem)
#   global __p

#   if typeof(md.Ω)==Nothing
#     meat!(md)
#   end

#   if typeof(md.∇g_n)==Nothing
#     mfjacobian!(md)
#   end

#   w   = weights(md)
#   θ   = coef(md)
#   __p = ones(nobs(md))
#   uw_g_i = md.mf.g_i(θ)
#   __p = weights(md)
#   w_g_i = md.mf.g_i(θ)

#   nvar = length(md.Ω)

#   out = Array(PDMat, nvar)

#   for j=1:nvar
#     out[j] = inv(PDMat(Xt_invA_X(md.Ω[j], md.∇g_n[j].∇)))
#   end

#   md.Σ = out

# end


# vcov(md::MinimumDivergenceProblem) = vcov(md::MinimumDivergenceProblem, :weighted)  ## Default to weighted var
# vcov(md::MinimumDivergenceProblem, vtype::Symbol) = vtype==:unweighted ? md.Σ[2] : md.Σ[1]


# stderr(md::MinimumDivergenceProblem) = stderr(md::MinimumDivergenceProblem, :weighted)

# ## function stderr(md::MinimumDivergenceProblem, vtype=:Symbol)
# ##     if vtype==:unweighted
# ##         sqrt(diag(md.Σ[2]))
# ##     elseif vtype==:weighted
# ##         sqrt(diag(md.Σ[1]))
# ##     elseif vtype==:hessian
# ##         md.Σ[1]
# ## end 


# function mfjacobian!(md::MinimumDivergenceProblem)
#   global __p
#   w   = weights(md)
#   θ   = coef(md)

#   out = Array(MomentFunctionJacobian, 2)

#   __p = ones(nobs(md))
#   out[1] = MomentFunctionJacobian(md.mf.Dg_n(θ))
#   __p = weights(md)
#   out[2] = MomentFunctionJacobian(md.mf.Dg_n(θ)./κ₁(md))
#   ## This is devided by κ₁  
#   md.∇g_n = out
# end

# function meat!(md::MinimumDivergenceProblem)
#     global __p
#     if typeof(md.∇g_n)==Nothing
#       mfjacobian!(md)
#     end
#     ## Adjustment constant
#     γ = md.mf.sf.St/md.mf.sf.κ₂
            
#     θ   = coef(md)
#     __p = ones(nobs(md))
#     uw_g_i = md.mf.g_i(θ)
#     __p = weights(md)
#     w_g_i = md.mf.g_i(θ)
#     md.Ω =  [PDMat(γ.*uw_g_i'*uw_g_i), PDMat(γ.*w_g_i'*w_g_i)]
# end



# ## Jacobian mdproblem
# ## Calculate
# ##
# ## \partial obj(theta) / \partial \theta
# ##

# function gradient(md::MinimumDivergenceProblem)
#   global __p
#   __p = weights(md)
#   md.mf.Dg_n(coef(md))'*md.lambda[1:ncond(md)]
# end

# function gradient(mdobj::MinimumDivergenceProblem, theta_0::Array{Float64,1}, s::Symbol)
#   obj(theta) = obj_val(md(mdobj.mf.g_i(theta), mdobj.div))
#   Calculus.gradient(obj, theta_0)
# end

# function hessian!(mdobj::MinimumDivergenceProblem)
#   mdobj.H = hessian(mdobj, coef(mdobj))
# end

# function hessian(mdobj::MinimumDivergenceProblem)
#   if typeof(mdobj.H)==Nothing
#     hessian!(mdobj)
#   end
#   return mdobj.H
# end

# function hessian(mdobj::MinimumDivergenceProblem, theta_0::Array{Float64,1})
#   obj(theta) = obj_val(md(mdobj.mf.g_i(theta), mdobj.div))
#   PDMat(Calculus.hessian(obj, theta_0))
# end


# function show(io::IO, obj::MinimumDivergenceProblem)
#     println(io,"$(typeof(obj)):\n\nCoefficients:", "\n", coef(obj))
# end


# ## function coeftable(mm::MinimumDivergenceProblem)
# ##     cc = coef(mm)
# ## #    se = stderr(mm)
# ## #    tt = cc ./ se
# ##     ## CoefTable(hcat(cc,se,tt,ccdf(FDist(1, df_residual(mm)), abs2(tt))),
# ##     ##           ["Estimate","Std.Error","t value", "Pr(>|t|)"],
# ##     ##           ["θ\_$i" for i = 1:length(cc)], 4)
# ##     println(["θ\_$i: cc[$i]" for i = 1:length(cc)])
# ## end
