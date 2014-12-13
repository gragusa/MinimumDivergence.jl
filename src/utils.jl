function gettril(x::Array{Float64, 2})
  n, m = size(x)
  a = zeros(convert(Int, n.*(n+1)/2))
  k::Int = 1
  for i = 1:n
    for j = 1:i
      a[k] = x[i, j]
      k += 1
    end
  end
  return a
end