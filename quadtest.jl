include("./quadoptim.jl")

function F(x,y,o,l)
    sql = sqrt(max(0.0,l*l-o*o))
    return exp(-x*sql)*l/sql*cos(y*l)
end
a = 1
b = 5
xs = LinRange(1,4,4)
ys = LinRange(0,4*sqrt(2)*b,4)
os = LinRange(a,b,4)
fs = [l->F(x,y,o,l) for x in xs, y in ys, o in os][:]
k = 16
tol = 1e-14
lmin = 6
lmax = 16
#x_n,w_n,x_step1,w_step1,U,interval_breaks = procedure(fs, a, b, k, tol, eps_quad)

# Plotting error vs eps_quad
#=
epses = 10.0.^(-12:-3)
n_eps = length(epses)
n_fun = length(fs)
errs = zeros(n_eps,n_fun)
for i = 1:n_eps
    eps_quad = epses[i]
    x_n,w_n,x_1,w_1 = procedure(fs, a, b, k, tol, eps_quad)
    for j = 1:n_fun
        itrue = sum(fs[j].(x_1).*w_1)
        errs[i,j] = abs((sum(fs[j].(x_n).*w_n) - itrue)/(itrue+1e-15))
    end
end
gui(plot(epses,errs.+1e-15,xscale=:log10,yscale=:log10,label=nothing,xlabel="eps_quad", ylabel="relative error in integral of \$f_i(x)\$", linewidth=2, tickfontsize=10, guidefontsize=12))
=#

# Plotting n_quadrature vs. eps_quad

epses = 10.0.^(-12:-3)
n_eps = length(epses)
n_fun = length(fs)
nquad = zeros(n_eps)
for i = 1:n_eps
    eps_quad = epses[i]
    x_n,w_n = procedure(fs, lmin, lmax, k, tol, eps_quad)
    nquad[i] = length(x_n)
end
gui(plot(epses,nquad,label=nothing,xscale=:log10,xlabel="eps_quad", ylabel="# quadrature pts", linewidth=2, tickfontsize=10, guidefontsize=12))


# Plotting
#=
xx = LinRange(a, b, 1000)
legnodes = legendre_zeros(k)
uu = funcInterp(xx,U,legnodes,interval_breaks)
gui(plot(xx,[fs[j](xx[i]) for i=1:length(xx),j=1:length(fs)],label=nothing,xlabel="\$x\$", ylabel="\$f_i(x)\$", linewidth=2, tickfontsize=10, guidefontsize=12))
=#

# Verification -- does it integrate fs[m]?
fi = sum(fs[end].(x_n).*w_n);
@show fi
@show length(x_n)
@show length(x_step1)

#=
function F(x,y,o,l)
    sql = sqrt(max(0.0,l*l-o*o))
    return exp(-x*sql)*l/sql*cos(y*l)
end
a = 1
b = 5
xs = LinRange(1,4,4)
ys = LinRange(0,4*sqrt(2)*b,4)
os = LinRange(a,b,4)
fs = [l->F(x,y,o,l) for x in xs, y in ys, o in os][:]
k = 16
tol = 1e-14
lmin = 6
lmax = 16
eps_quad = 1e-1
=#

#=
js = 0:6
cs = LinRange(-0.6,1.0,4)
fs = [x->sin(j*x)*x^c for j in js, c in cs][:]
a = 0.01
b = 1
k = 16
tol = 1e-14
eps_quad = 1e-5
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