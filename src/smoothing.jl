abstract SmoothingKernel

immutable IdentityKernel <: SmoothingKernel
    scale::Real
    κ₁::Real
    κ₂::Real
end

immutable TruncatedKernel <: SmoothingKernel
    ξ::Integer
    scale::Real
    smoother::Function
    κ₁::Real
    κ₂::Real
end

immutable BartlettKernel <: SmoothingKernel
    ξ::Integer
    scale::Real
    smoother::Function
    κ₁::Real
    κ₂::Real
end

IdentityKernel() = IdentityKernel(2.0, 1.0, 1.0)

function TruncatedKernel(ξ::Integer)
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
    TruncatedKernel(ξ, 2.0/(2.0*ξ+1.0), smoother, 1.0, 1.0)
end

function BartlettKernel(ξ::Integer)
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
    BartlettKernel(ξ, 2.0/(2.0*ξ+1.0)*(2.0/3.0)^2, smoother, 1.0, 2.0/3.0)
end