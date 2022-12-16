using LinearAlgebra
using Jacobi
using Plots, LaTeXStrings

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
    err = sum(alphas[k:2k].^2)#/sum(alphas[1:k].^2)
    if err < tol
        return ([alphas[1:k]], [a,b])
    elseif depth >= 10
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
    ff = zeros(k*nbreaks,nfuns)
    xx = zeros(k)
    for j = 1:nbreaks
        lo = interval_breaks[j]
        hi = interval_breaks[j+1]
        xx .= 0.5*(hi-lo)*legnodes .+ (hi+lo)/2
        for i = 1:nfuns
            ff[(j*k+1-k):(j*k),i] .= fs[i].(xx)
        end
        fi[:,j,:] .= Vlu\ff[(j*k+1-k):(j*k),:]
    end
    return fi, interval_breaks, ff
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
function compressPhi(phi,x,w,eps_quad)
    m = size(phi,2)        # Number of functions
    n = length(x)
    A = zeros(n,m)
    for j = 1:m
        for i = 1:n
            A[i,j] = phi[i,j]*sqrt(w[i])
        end
    end
    A_svd = svd(A)
    U = A_svd.U 
    for i = 1:n
        U[i,:] /= sqrt(w[i])
    end
    k = sum(A_svd.S .> eps_quad)
    return U[:,1:k], A_svd.S[1:k]
end

# Stage 2 -- algorithm 3.3, modified Gram-Schmidt to get k-point quadrature for u1...uk
function modifiedGS(U,x,w)
    n,k = size(U)
    r = zeros(k)
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
    wnew = z.*sqrt.(w[idxs])
    return xnew, wnew
end

# helper function for GaussNewton! -- evaluates f and gradf
function Newtonf(x_n,w_n,U,r,legnodes,interval_breaks,iV)
    k = size(U,2)
    n = length(x_n)
    f = 0
    r_n = zeros(k,1)
    grad1 = zeros(n,1)
    grad2 = zeros(n,1)
    U_n = funcInterp(x_n,U,legnodes,interval_breaks)
    dU_n = derivInterp(x_n,U,legnodes,interval_breaks,iV)
    for jj = 1:n
        for ii = 1:k
            r_n[ii] += U_n[jj,ii]*w_n[jj]
        end
    end
    for ii = 1:k
        f += (r_n[ii] - r[ii])^2
        for jj = 1:n
            grad1[jj] += w_n[jj]*dU_n[jj,ii]*(r_n[ii] - r[ii])
            grad2[jj] += U_n[jj,ii]*(r_n[ii] - r[ii])
        end
    end
    gradf = vcat(grad1,grad2)
    return f,gradf
end

function GaussNewton!(x_n,w_n,U,r,iters,legnodes,interval_breaks,iV)
    k = size(U,2)
    n = length(x_n)
    testx = zeros(n)
    testw = zeros(n)
    for idx = 1:iters
        U_n = funcInterp(x_n,U,legnodes,interval_breaks)
        dU_n = derivInterp(x_n,U,legnodes,interval_breaks,iV)
        #println(size(dU_n'),size(w_n),size(U_n'))
        J = hcat(dU_n'.*w_n',U_n')
        # Compute Gauss-Newton direction
        Dx = J\r
        # line search
        alpha = 1
        lambda = 0.1
        beta = 0.2
        while true
            testx .= x_n .+ alpha*Dx[1:n]
            testw .= w_n .+ alpha*Dx[n+1:2n]
            f_old, g_old = Newtonf(x_n,w_n,U,r,legnodes,interval_breaks,iV)
            f_new, g_new = Newtonf(testx,testw,U,r,legnodes,interval_breaks,iV)
            if (f_new - lambda*alpha*dot(g_old,Dx) <= f_old) && (dot(g_new,Dx) >= beta*dot(g_old,Dx))
                break 
            end
            alpha *= 0.5
        end

        # update x and w
        x_n .+= alpha*Dx[1:n]
        w_n .+= alpha*Dx[n+1:2n]
    end
    return x_n, w_n
end

# Stage 3 - reduce k-node quadrature rule to one that is as small as possible
function quadReduce(U,x_step1,w_step1,x_tilde,w_tilde,a,b, interval_breaks,eps_quad)
    @show size(U)
    k = size(U,2)
    # current size of quadrature rule
    n = k
    # initialize x, w
    x_n = copy(x_tilde)
    w_n = copy(w_tilde)
    # reduction not complete
    complete = false

    # Find correct number of x points
    n_int = length(interval_breaks)-1
    nx = div(size(U,1),n_int)
    legnodes = legendre_zeros(nx)
    V = legvander(legnodes,nx-1)
    iV = inv(V)
    
    # compute RHS
    r = zeros(k,1)
    for j = 1:n
        for i = 1:k
            r[i] += U[j,i]*w_step1[j]
        end
    end

    # loop until can't reduce quadrature rule any further
    while !complete
        println(complete)
        # Step 1
        U_n = funcInterp(x_n,U,legnodes,interval_breaks)
        dU_n = derivInterp(x_n,U,legnodes,interval_breaks,iV)
        J = hcat(dU_n'.*w_n',U_n')
        #A = inv(J'*J)
        # gradient magnitude
        eta = zeros(n)
        for idx = 1:n
            # Sherman Morrison Woodbury (twice)
            #=
            e_idx = zeros(2n,1); e_idx[idx] = 1
            e_idxn = zeros(2n,1); e_idxn[idx+n] = 1
            upd1 = J[:,idx]*e_idx'
            upd2 = J[:,idx+n]*e_idxn'
            J_idx = J-upd1-upd2
            A_idx = A - A*upd1*inv(I + upd1'*A*upd1)*upd1'*A
            A_idx = A_idx - A_idx*upd2*inv(I + upd2'*A_idx*upd2)*upd2'*A_idx
            # Compute Gauss-Newton direction
            println(size(A_idx),size(J_idx),size(r))
            =#
            J_idx = copy(J)
            J_idx[:,idx] = zeros(size(J,1))
            J_idx[:,idx+n] = zeros(size(J,1))
            Dx_idx = J_idx\r
            eta[idx] = norm(Dx_idx)
        end
        ord = sortperm(eta)
        eta = eta[ord]
        x_n = x_n[ord]
        w_n = w_n[ord]

        # Step 2
        accepted = false
        j = 1
        eps = zeros(n)
        println("Looking for columns to remove")
        while !accepted && j <= n
            println("Index removed: ",j)
            x_j = copy(x_n)
            w_j = copy(w_n)
            deleteat!(x_j,j)
            deleteat!(w_j,j)
            x_jnew,w_jnew = GaussNewton!(x_j,w_j,U,r,4,legnodes,interval_breaks,iV)
            # compute error estimate
            r_jnew = zeros(k,1)
            U_jnew = funcInterp(x_jnew,U,legnodes,interval_breaks)
            for jj = 1:n-1
                for ii = 1:k
                    r_jnew[ii] += U_jnew[jj,ii]*w_jnew[jj]
                end
            end
            for ii = 1:k
                eps[j] += (r_jnew[ii] - r[ii])^2
            end
            if eps[j] < eps_quad
                accepted = true
            end
            j += 1
        end
        # Step 3
        if !accepted
            println("No columns removable, looking again in more detail")
            ord = sortperm(eps)
            x_n = x_n[ord]
            w_n = w_n[ord]
            accepted2 = false
            j = 1
            while !accepted2 && j <= n
                println("Column removed: ",j)
                x_j = copy(x_n)
                w_j = copy(w_n)
                deleteat!(x_j,j)
                deleteat!(w_j,j)
                x_jnew,w_jnew = GaussNewton!(x_j,w_j,U,r,30,legnodes,interval_breaks,iV)
                # compute error estimate
                eps = 0
                r_jnew = zeros(k,1)
                U_jnew = funcInterp(x_jnew,U,legnodes,interval_breaks)
                for jj = 1:n-1
                    for ii = 1:k
                        r_jnew[ii] += U_jnew[jj,ii]*w_jnew[jj]
                    end
                end
                for ii = 1:k
                    eps += (r_jnew[ii] - r[ii])^2
                end
                if eps < eps_quad
                    accepted2 = true
                end
                j += 1
            end
            if !accepted2
                # completed, cannot reduce to n-1, do not update x_n,w_n
                complete = true
            else
                # set new n-1-point quadrature
                x_n = x_jnew
                w_n = w_jnew
                n -= 1
            end
        # Step 4
        else
            x_n = x_jnew
            w_n = w_jnew
            n -= 1
        end
    end
    return x_n, w_n
end

#= 
Interpolate Uij = u_i(x_j) at arbitrary points xx
interval_breaks is the set of divisions in the piecewise polynomial ui
legnodes is the Legendre nodes xi which form the interpolation pts
    in each subinterval
=#
function funcInterp(xx, U, legnodes, interval_breaks)
    k = length(legnodes)
    nx = length(xx)
    nu = size(U,2)
    ux = zeros(nx,nu)
    for j = 1:nx
        x = xx[j]
        lo = bisect_srch(x,interval_breaks)
        a = interval_breaks[lo]
        b = interval_breaks[lo+1]
        xnorm = (2x-(a+b))/(b-a)
        ivec = interp_mat([xnorm],legnodes)[:]
        # We know there are k nodes per subinterval in x_is
        ux[j,:] .= U[(k*lo+1-k):(k*lo),:]'*ivec
    end
    return ux
end

#= 
Calculate derivatives of u_i at xx given Uij = u_i(x_j)
interval_breaks is the set of divisions in the piecewise polynomial ui
legnodes is the Legendre nodes xi which form the interpolation pts
    in each subinterval
iV is the inverse of the Vandermonde matrix Vij = Pj(xi)
=#
function derivInterp(xx, U, legnodes, interval_breaks, iV)
    k = length(legnodes)
    nx = length(xx)
    nu = size(U,2)
    ux = zeros(nx,nu)
    for j = 1:nx
        x = xx[j]
        lo = bisect_srch(x,interval_breaks)
        a = interval_breaks[lo]
        b = interval_breaks[lo+1]
        xnorm = (2x-(a+b))/(b-a)
        # We have Pj'(x) but Uij is in nodal form rather than modal form
        # iV inverts from nodal to modal form
        lvec = 2/(b-a)*iV'*[dlegendre(xnorm,j) for j = 0:k-1]
        # We know there are k nodes per subinterval in x_is
        ux[j,:] .= U[(k*lo+1-k):(k*lo),:]'*lvec
    end
    return ux
end

function F(x,y,o,l)
    sql = sqrt(max(0.0,l*l-o*o))
    return exp(-x*sql)*l/sql*cos(y*l)
end
#=
function sinphi(j,x)
    return sin.(2*pi*j * x)
end
js = 1:6
fs = [x->sinphi(j,x) for j in js]
a = -1
b = 1
k = 16
tol = 1e-7
=#
#=
a = 1
b = 5
xs = LinRange(1,4,4)
ys = LinRange(0,4*sqrt(2)*b,4)
os = LinRange(a,b,4)
fs = [l->F(x,y,o,l) for x in xs, y in ys, o in os][:]
k = 16
tol = 1e-3
lmin = 6
lmax = 16

coeffs, interval_breaks, ff = totalInterp(fs,lmin,lmax,k,tol)
legnodes = legendre_zeros(k)
V = legvander(legnodes,k-1)
iV = inv(V)
xx = LinRange(lmin,lmax,300)
fI = funcInterp(xx,ff,legnodes,interval_breaks)
#=
test_idx = 1
ip = interpoly2(xx, coeffs, interval_breaks, test_idx)
=#
#plot(xx,fI)
plot(xx,[fs[i](xx[j])-fI[j,i] for j=1:length(xx), i=1:20])
=#

#=
f = x -> abs(x)^(1/3)
a = -1
b = 1
k = 3
tols = [5e-2,1e-2,1e-5]
xx = LinRange(a,b,300)
nt = length(tols)
iS = zeros(length(xx),nt)
=#

#=
a = 1
b = 5
x = 1
y = 5
o = 4
f = l -> F(x,y,o,l)
lmin = 6
lmax = 16
xx = LinRange(lmin,lmax,300)
k = 5
tols = [5e-1,1e-1,1e-4]
for i = 1:nt
    coeffs,interval_breaks = adaptiveInterp(f, lmin, lmax, k, tols[i])
    iS[:,i] .= interpoly(xx,coeffs,interval_breaks)
    println(interval_breaks)
end

plot(xx,f.(xx), label=L"$\phi(\xi;z=%$x,x=%$y,\omega=%$o$)", xlabel="\$x\$", ylabel="\$f(x)\$", title="Adaptive Interpolation, \$N=5\$")
plot!(xx,iS, label = ["tol = $(tols[1])" "tol = $(tols[2])" "tol = $(tols[3])"])
=#