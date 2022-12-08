using LinearAlgebra
using Jacobi
using Plots

# define phi i

# Vandermonde matrix for Legendre polynomials
function legvander(xx,k)
    n = length(xx)
    if k == 0
        return ones(n,1)
    elseif k == 1
        return hcat(ones(n,1),xx)
    else
        L = ones(n,k+1)
        L[:,2] .= xx
        for i = 1:k-1
            @. L[:,i+2] = (2i+3)/(i+2)*xx*L[:,i+1] - (i+1)/(i+2)*L[:,i]
        end
        return L
    end
end

# Helper fn. for adaptiveInterp
function recursiveInterp(f, a, b, k, tol, Vlu, legnodes, depth)
    x = 0.5*(b-a)*legnodes .+ (b+a)/2
    # From Lagrange find Legendre coefficients
    alphas = Vlu\f.(x)
    # Test stopping condition
    err = sum(alphas[k:2k].^2)
    if err < tol
        return ([alphas[1:k]], [a,b])
    elseif depth >= 40
        error("Recursion has proceeded too far! Check your function and domain.")
    else
        midpt = 0.5*(a+b)
        left = recursiveInterp(f, a, midpt, k, tol, Vlu, legnodes, depth+1)
        right = recursiveInterp(f, midpt, b, k, tol, Vlu, legnodes, depth+1)
        return (cat(left[1], right[1],dims=1), cat(left[2], right[2],dims=1))
    end
end

# Adaptive interpolation
function adaptiveInterp(f, a, b, k, tol)
    # Construct 2k Legendre nodes x1:x2k on [-1,1]
    legnodes = legendre_zeros(2k)
    # Construct Lagrange interpolating matrix
    V = legvander(legnodes, 2k-1)
    # Pre-factor it
    Vlu = lu(V)
    # Adaptively interpolate
    coeffs,brackets = recursiveInterp(f,a,b,k,tol,Vlu,legnodes,1)
    interval_breaks = unique(brackets)
    return coeffs, interval_breaks
end

# debugging code
function basicInterp(f, a, b, k, tol)
    legnodes = legendre_zeros(k)
    V = legvander(legnodes, k-1)
    Vlu = lu(V)
    x = 0.5*(b-a)*legnodes .+ (b+a)/2
    return Vlu\f.(x)
end

function totalInterp(fs, a, b, k, tol)
    # Find all subdivision points for all f_is
    interval_breaks = []
    xx = LinRange(a,b,300)
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
    legnodes = legendre_zeros(k)
    V = legvander(legnodes,k-1)
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
    k = length(coeffs[1])
    n = length(xx)
    ip = zeros(n)
    for i = 1:n
        x = xx[i]
        lo = bisect_srch(x, interval_breaks)
        a = interval_breaks[lo]
        b = interval_breaks[lo+1]
        xnorm = (2x-(a+b))/(b-a)
        ip[i] = dot(legvander(xnorm,k-1)[:],coeffs[lo])
    end
    return ip
end
function interpoly2(xx, coeffs, interval_breaks, idx=1)
    nbrac = length(interval_breaks)-1
    k = size(coeffs)[1]
    n = length(xx)
    ip = zeros(n)
    for i = 1:n
        x = xx[i]
        lo = bisect_srch(x, interval_breaks)
        a = interval_breaks[lo]
        b = interval_breaks[lo+1]
        xnorm = (2x-(a+b))/(b-a)
        # Only difference is format of coeffs
        ip[i] = dot(legvander(xnorm,k-1)[:],coeffs[:,lo,idx])
    end
    return ip
end

# Stage 1, Step 2,3,4 - compress the phi_j functions
function compressPhi(phi,x,w,m,eps_quad)
    n = length(x)
    A = zeros(n,m)
    for j = 1:m
        for i = 1:n
            A[i,j] = phi(j,x[i])*sqrt(w[i])
        end
    end
    A_svd = svd(A)
    U = A_svd.U 
    for i = 1:n
        U[i,:] /= sqrt(w[i])
    end
    k = sum(A_svd.s .> eps_quad)
    return U[:,1:k], A_svd.s[1:k]
end

# Stage 2 -- algorithm 3.3, modified Gram-Schmidt to get k-point quadrature for u1...uk
function modifiedGS(U,x,w)
    n,k = size(U)
    r = zeros(k,1)
    B = zeros(k,n)
    for j = 1:n
        for i = 1:k
            r[i] += U[j,i]*w[j]
            B[i,j] = U[j,i]*sqrt(w[j])
        end
    end
    F = qr(B, Val(true))
    idxs = F.p[1:k]
    R11 = F.R[:,1:k]
    z = R11\F.Q'*r
    xnew = x[idxs]
    wnew = z.*sqrt.(w[1:k]) # seems suspish -- why 1:k instead of idxs? Might be a typo
    return xnew, wnew
end

function F(x,y,o,l)
    sql = sqrt(max(0.0,l*l-o*o))
    return exp(-x*sql)*l/sql*cos(y*l)
end
#=
function sinphi(j,x)
    return sin.(2*pi/j * x)
end
js = 1:2
fs = [x->sinphi(j,x) for j in js]
a = -pi
b = pi
k = 16
tol = 1e-7
=#

a = 1
b = 5
xs = LinRange(1,4,8)
ys = LinRange(0,4*sqrt(2)*b,12)
os = LinRange(a,b,10)
fs = [l->F(x,y,o,l) for x in xs, y in ys, o in os][:]
k = 16
tol = 1e-3
lmin = 6
lmax = 60

coeffs, interval_breaks = totalInterp(fs,lmin,lmax,k,tol)

xx = LinRange(lmin,lmax,300)
test_idx = 1
ip = interpoly2(xx, coeffs, interval_breaks, test_idx)
plot(xx,ip)
plot!(xx,fs[test_idx].(xx))
