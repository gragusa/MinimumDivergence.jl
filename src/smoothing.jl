abstract SmoothingKernel

immutable IdentitySmoother <: SmoothingKernel
    scale::Real
    κ₁::Real
    κ₂::Real
end

immutable TruncatedSmoother <: SmoothingKernel
    ξ::Integer
    scale::Real
    smoother::Function
    κ₁::Real
    κ₂::Real
end

immutable BartlettSmoother <: SmoothingKernel
    ξ::Integer
    scale::Real
    smoother::Function
    κ₁::Real
    κ₂::Real
end

IdentitySmoother() = IdentitySmoother(2.0, 1.0, 1.0)

function TruncatedSmoother(ξ::Integer)
    function smoother{T}(G::Array{T, 2})
        N, M = size(G)
        nG   = zeros(T, N, M)
        for m=1:M
            for t=1:N
			     low = max((t-N), -ξ)
			     high = min(t-1, ξ)
				 for s = low:high
                    @inbounds nG[t, m] += G[t-s, m]
                end
            end
        end
        return(nG/(2.0*ξ+1.0))
    end
    TruncatedSmoother(ξ, 2.0/(2.0*ξ+1.0), smoother, 1.0, 1.0)
end

function BartlettSmoother(ξ::Integer)
    function smoother{T}(G::Array{T, 2})
        N, M = size(G)
        nG   = zeros(T, N, M)
        St   = (2.0*ξ+1.0)/2.0
        for m=1:M
            for t=1:N
			    low = max((t-N), -ξ)
			    high = min(t-1, ξ)
				for s = low:high
                    κ = 1.0-s/St
                    @inbounds nG[t, m] += κ*G[t-s, m]
                end
            end
        end
        return(nG/(2*ξ+1))
    end
    BartlettSmoother(ξ, 2.0/(2.0*ξ+1.0)*(2.0/3.0)^2, smoother, 1.0, 2.0/3.0)
end
