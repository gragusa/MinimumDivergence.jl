function ar(iv::MinimumDivergence.InstrumentalVariableModel, theta0::Vector)
    n = length(iv.y)
    k = size(iv.x)[2]
    g  = iv.y - iv.x*theta0
    Mz = eye(n) - iv.Pz
    (n-k)*((g'*iv.Pz*g)/(g'Mz*g))[1]
end

function k(mdp::MDP, theta0::Vector)
    
    
    
    
    
end


type InstrumentalVariableEstimator
    coef::Array{Float64, 1}
    vhom::Array{Float64, 2}
    vhet::Array{Float64, 2}
end

function ivreg(iv::iv::MinimumDivergence.InstrumentalVariableModel)
    xPz= x'*Pz
    beta = xPz*x\xPz*y
    vhom = _
    vhet = _
end 
        




