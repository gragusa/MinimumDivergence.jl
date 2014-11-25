immutable WhiteKernel <: SmoothingKernels
    St::Real
    κ₁::Real
    κ₂::Real
end    

WhiteKernel() = WhiteKernel(1.0, 1.0, 1.0)

immutable TruncatedKernel <: SmoothingKernels
    ξ::Real
    St::Real
    SmoothingFunction::Function
    κ₁::Real
    κ₂::Real
end

immutable BartlettKernel <: SmoothingKernels
    ξ::Real
    St::Real
    SmoothingFunction::Function
    κ₁::Real
    κ₂::Real
end

function TruncatedKernel(ξ::Real)
    function SmoothingFunction(G::Array{Float64, 2})
        T, M = size(G)
        nG   = zeros(T, M)        
        for m=1:M
            for t=1:T
			             low = max((t-T), -ξ)
			             high = min(t-1, ξ)
					           for s = low:high
                    @inbounds nG[t, m] += G[t-s, m]
                end
            end
        end
        return(nG/(2.0*ξ+1.0))
    end
    TruncatedKernel(ξ, (2.0*ξ+1.0)/2.0, SmoothingFunction, 1.0, 1.0)
end 

function BartlettKernel(ξ::Real)
    function SmoothingFunction(G::Array{Float64, 2})
        T, M = size(G)
        nG   = zeros(T, M)
        St   = (2.0*ξ+1.0)/2.0
        for m=1:M
            for t=1:T
			             low = max((t-T), -ξ)
			             high = min(t-1, ξ)
					           for s = low:high
                    κ = 1.0-s/St
                    @inbounds nG[t, m] += κ*G[t-s, m]
                end
            end
        end
        return(nG/(2*ξ+1))
    end
    BartlettKernel(ξ, (2.0*ξ+1.0)/2.0, SmoothingFunction, 1.0, 2.0/3.0)
end 
