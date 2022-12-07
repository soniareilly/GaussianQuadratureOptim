using LinearAlgebra
using Polynomials,SpecialPolynomials

# define phi i

# roots of the kth order Legendre polynomial
function LegendreRoots(k)
    a = zeros(k+1)
    a[end] = 1.0
    return roots(Legendre(a))
end

# construct Lagrange polynomial


legnodes = LegendreRoots(2k)
# Stage 1
    # Step 1
        # Construct 2k Legendre nodes x1:x2k on [a,b]
        x = (b-a)*legnodes/2 + (b+a)/2
        # Construct Lagrange interpolating polynomial
        # From Lagrange find Legendre coefficients
        # Test stopping condition
        # Discretize adaptively and recurse