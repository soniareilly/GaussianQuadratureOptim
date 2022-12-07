using LinearAlgebra
using Polynomials,SpecialPolynomials

# define phi i

# roots of the kth order Legendre polynomial
function LegendreRoots(k)
    a = zeros(k+1)
    a[end] = 1.0
    return roots(Legendre(a))
end

function recursiveInterp(f, a, b, tol, Vqr, legnodes)
    x = (b-a)*legnodes/2 + (b+a)/2
    # From Lagrange find Legendre coefficients
    alphas = Vqr\f.(x)
    # Test stopping condition
    err = sum(alphas[k:2k].^2)
    if err < tol
        return ([alphas[1:k]], [[a,b]])
    else
        midpt = 0.5*(a+b)
        left = recursiveInterp(f, a, midpt, tol, Vqr, legnodes)
        right = recursiveInterp(f, midpt, b, tol, Vqr, legnodes)
        return (cat(left[0], right[0]), cat(left[1], right[1]))
    end
end

function adaptiveInterp(f, a, b, tol)
    # Construct 2k Legendre nodes x1:x2k on [-1,1]
    legnodes = LegendreRoots(2k)
    # Construct Lagrange interpolating matrix
    V = vander(::Legendre, legnodes, 2k)
    # Pre-factor it
    Vqr = qr(V)
    # Adaptively interpolate
    coeffs,brackets = recursiveInterp(f,a,b,tol,Vqr,legnodes)
    interval_breaks = unique(brackets)
    return coeffs, brackets, interval_breaks
end

function bisect_srch(x,Y)
    n = length(Y)
    lo = 1
    hi = n
    while lo+1 < hi
      ix = (hi+lo)>>1
      yy = Y[ix]
      if yy <= x
        lo = ix
      else
        hi = ix
      end
    end
    return lo
  end

function interpoly(xx, coeffs, interval_breaks)
    nbrac = length(coeffs)
    n = length(x)
    interps = [Legendre(coeffs[i][0]) for i = 1:nbrac]
    ip = zeros(n)
    for i = 1:n
        x = xx[i]
        lo = bisect_srch(x, interval_breaks)
        a = interval_breaks[lo]
        b = interval_breaks[hi]
        xnorm = (2x-(a+b))/(b-a)
        ip[i] = interps[lo](xnorm)
    end
    return ip
end


f = log
a = 1e-3
b = 3
tol = 1e-5
coeffs, brackets, interval_breaks = adaptiveInterp(f,a,b,tol)

xx = LinRange(a,b,300)
ip = interpoly(xx, coeffs, interval_breaks)
plot(xx,ip)