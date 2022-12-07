using LinearAlgebra
using Polynomials,SpecialPolynomials
using Plots

# define phi i

# roots of the kth order Legendre polynomial
function LegendreRoots(k)
    a = zeros(k+1)
    a[end] = 1.0
    return real.(roots(Legendre(a)))
end

# Compute a set of quadrature weights for a set of quadrature nodes
function quadWeights(xx)
    k = length(xx)
    V = vander(Legendre,xx,k)
    iV = inv(V)
    poly = [integrate(Legendre(iV[:,i])) for i = 1:k]
    return [poly[i](1.0) - poly[i](-1.0) for i = 1:k]
end

# Helper fn. for adaptiveInterp
function recursiveInterp(f, a, b, k, tol, Vlu, legnodes)
    x = 0.5*(b-a)*legnodes .+ (b+a)/2
    # From Lagrange find Legendre coefficients
    alphas = Vlu\f.(x)
    # Test stopping condition
    err = sum(alphas[k:2k].^2)
    if err < tol
        return ([alphas[1:k]], [a,b])
    else
        midpt = 0.5*(a+b)
        left = recursiveInterp(f, a, midpt, k, tol, Vlu, legnodes)
        right = recursiveInterp(f, midpt, b, k, tol, Vlu, legnodes)
        return (cat(left[1], right[1],dims=1), cat(left[2], right[2],dims=1))
    end
end

# Adaptive interpolation
function adaptiveInterp(f, a, b, k, tol)
    # Construct 2k Legendre nodes x1:x2k on [-1,1]
    legnodes = LegendreRoots(2k)
    # Construct Lagrange interpolating matrix
    V = vander(Legendre, legnodes, 2k-1)
    # Pre-factor it
    Vlu = lu(V)
    # Adaptively interpolate
    coeffs,brackets = recursiveInterp(f,a,b,k,tol,Vlu,legnodes)
    interval_breaks = unique(brackets)
    return coeffs, interval_breaks
end

function totalInterp(fs, a, b, k, tol)
    # Find all subdivision points for all f_is
    interval_breaks = []
    for f in fs
        cf,ib = adaptiveInterp(f,a,b,k,tol)
        interval_breaks = cat(interval_breaks,ib,dims=1)
    end
    interval_breaks = unique(interval_breaks)
    sort!(interval_breaks)

    # Construct interpolation polynomials for all f_i on each subinterval
    nfuns = length(fs)
    nbreaks = length(interval_breaks)-1
    fi = zeros(k,nbreaks,nfuns)
    legnodes = LegendreRoots(k)
    V = vander(Legendre,legnodes,k-1)
    Vlu = lu(V)
    ff = zeros(k,nfuns)
    xx = zeros(k)
    for j = 1:nbreaks
        lo = interval_breaks[j]
        hi = interval_breaks[j+1]
        xx .= 0.5*(hi-lo)*legnodes .+ (hi+lo)/2
        for i = 1:nfuns
            ff[:,i] .= fs[i].(xx)
        end
        fi[:,j,:] .= Vlu\ff
    end
    return fi, interval_breaks
end

# Helper fn; bisection search
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

# Evaluates an adaptively interpolated function at an array of points xx
function interpoly(xx, coeffs, interval_breaks)
    nbrac = length(coeffs)
    n = length(xx)
    interps = [Legendre(coeffs[i]) for i = 1:nbrac]
    ip = zeros(n)
    for i = 1:n
        x = xx[i]
        lo = bisect_srch(x, interval_breaks)
        a = interval_breaks[lo]
        b = interval_breaks[lo+1]
        xnorm = (2x-(a+b))/(b-a)
        ip[i] = interps[lo](xnorm)
    end
    return ip
end
function interpoly2(xx, coeffs, interval_breaks, idx=1)
    nbrac = length(interval_breaks)-1
    n = length(xx)
    # Only difference is format of coeffs
    interps = [Legendre(coeffs[:,i,idx]) for i = 1:nbrac]
    ip = zeros(n)
    for i = 1:n
        x = xx[i]
        lo = bisect_srch(x, interval_breaks)
        a = interval_breaks[lo]
        b = interval_breaks[lo+1]
        xnorm = (2x-(a+b))/(b-a)
        ip[i] = interps[lo](xnorm)
    end
    return ip
end

# Stage 1, Step 2 - compress the phi_j functions
function compressPhi(phi,x,w,m)
    n = length(x)
    A = zeros(n,m)
    for j = 1:m
        A[:,j] = phi.(j,x).*sqrt.(w)
    end
    F = qr(A, pivot == ColumnNorm())
    U = F.Q 
    for i = 1:n
        U[i,:] /= sqrt(w[i])
    end
    lambda = Diagonal(F.R)
    return U,lambda
end


function F(x,y,o,l)
    sql = sqrt(max(0.0,l*l-o*o))
    return exp(-x*sql)*l/sql*cos(y*l)
end
function sinphi(j,x)
    return sin.(2*pi/j * x)
end
js = 1:6
fs = [x->sinphi(j,x) for j in js]
a = -pi
b = pi
k = 16
tol = 1e-7
#=
a = 1
b = 5
xs = LinRange(1,4,8)
ys = LinRange(0,4*sqrt(2)*b,12)
os = LinRange(a,b,10)
fs = [l->F(x,y,o,l) for x in xs, y in ys, o in os][:]
k = 16
tol = 1e-3
lmin = 0
lmax = 60
=#
coeffs, interval_breaks = totalInterp(fs,a,b,k,tol)
println(interval_breaks)

xx = LinRange(a,b,300)
test_idx = 1
ip = interpoly2(xx, coeffs, interval_breaks, test_idx)
plot(xx,ip)
plot!(xx,fs[test_idx].(xx))
