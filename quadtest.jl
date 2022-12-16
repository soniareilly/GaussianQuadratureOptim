include("./quadoptim.jl")

function procedure(fs, a, b, k, tol, eps_quad)
    legnodes = legendre_zeros(k)
    gauss_wts = wgj(legnodes)
    # Step 1
    fi, interval_breaks, ff = totalInterp(fs, a, b, k, tol)
    n_int = length(interval_breaks)-1   # # intervals
    x_step1 = zeros(n_int*k)
    w_step1 = zeros(n_int*k)
    for i = 1:n_int
        a = interval_breaks[i]
        b = interval_breaks[i+1]
        x_step1[(i-1)*k+1:i*k] .= legnodes*(b-a)/2 .+ (b+a)/2
        w_step1[(i-1)*k+1:i*k] .= gauss_wts*(b-a)/2
    end
    # Step 1, stages 2,3,4
    U,A_svd = compressPhi(ff,x_step1,w_step1,eps_quad)
    gui(plot(x_step1,U))
    # Step 2
    x_tilde,w_tilde = modifiedGS(U,x_step1,w_step1)
    # Step 3
    x_n,w_n = quadReduce(U,x_step1,w_step1,x_tilde,w_tilde,a,b,interval_breaks,eps_quad)
    return x_n,w_n,x_step1,w_step1,x_tilde,w_tilde
end

m = 10
fs = [x -> sin(i*x) for i = 1:m]
a = 0.0
b = pi
k = 10
tol = 1e-8
eps_quad = 1e-6
x_n,w_n,x_step1,w_step1,x_tilde,w_tilde = procedure(fs, a, b, k, tol, eps_quad)

# Verification -- does it integrate fs[m]?
fi = sum(fs[m].(x_n).*w_n);
@show fi
@show x_n
@show x_step1
@show x_tilde