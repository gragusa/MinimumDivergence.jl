
nobs(md::MinimumDivergenceProblem) = md.n
ncond(md::MinimumDivergenceProblem) = md.m
npar(md::MinimumDivergenceProblem) = md.p

lambda(md::MinimumDivergenceProblem) = md.lambda


coef(md::MinimumDivergenceProblem) = md.x[obs(md)+1:end]



WeightVec(md::MinimumDivergenceProblem) = WeightVec(md.fprob.x[1:nobs(md)])
weights(md::MinimumDivergenceProblem) = WeightVec(md.fprob.x[1:nobs(md)])




function vcov!(md::MinimumDivergenceProblem)
  if isempty(md.Ω)
    meat!(md)
  end

  if isempty(md.∇g_n)
    jacobian!(md)
  end

  w   = WeightVec(md).value
  θ   = coef(md)
  uw_g_i = md.mf.g_i(θ)
  __p = w.values
  w_g_i = md.mf.g_i(θ)

  nvar = length(md.Ω)

  out = Array(PDMat, nvar)


  for j=1:nvar
    out[j] = Xt_invA_Xt(md.Ω[j], md.∇[j])
  end

  md.Σ = [PDMats(uw_g_i*uw_g_i'), PDMats(w_g_i*w_g_i')]

end


  function meat!(md::MinimumDivergenceProblem)
    w   = WeightVec(md).value
    θ   = coef(md)
    uw_g_i = md.mf.g_i(θ)
    uw_∇g_n = md.mf.Dg_n(θ)
    md.Ω =  [PDMat(uw_g_i*uw_g_i'), PDMat(w_g_i*w_g_i')]
    return Ω
  end





