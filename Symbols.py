# بسم الله الرحمن الرحيم

import sympy as sym
from sympy import solve, lambdify, limit, sin, pprint

'''
* We could've also imported "sympy" as "sym", so that we don't have to import many methods from the "sympy" class, we can simply: "sym.method"
* To print "pretty" (readable) expressions, use "pprint"
* To print expressions that are compatible with LaTeX, import "print_latex" from "sympy", then do "print_latex(fun)"
'''

x, y = sym.symbols('x y')


f1 = (x ** 2 - 5 * x + 6)
answer = solve(f1)  # Always solves the argument given (f1) as equalling 0 --> f1 = 0
print("f1 =", f1, "\n" "x =", answer, "\n")

f1_prime = f1.diff(x)  # If it's a single variable function (like "f1"), you don't have to specify the variable ==> "f1.diff()"
print("f1_prime =", f1_prime)
print("f1_prime(10) =", f1_prime.subs(x, 10), "\n")

# Note: functions with symbols in them are not callable (yet), they aren't actual functions. To be able to use them as actual functions invoke: "lambdify"
# Example:
f1 = lambdify(x, f1)
f1_prime = lambdify(x, f1_prime)

# Now they are "real" functions that can take arguments and give back results:
print("f1(2) =", f1(2))
print("f1_prime(2) =", f1_prime(2), "\n")

f2 = (2 * x + 5 * y)
print("f2 =", f2)
print("f2(x) =", f2.subs(y, 2))
print("f2(y) =", f2.subs(x, 2))
f2 = lambdify((x, y), f2)  # Making "f2" a real function with 2 input arguments (x,y)
print("f2(3,2) =", f2(3, 2), "\n")


# Substituting value for more than one variable in a function:
g = x**2 + 3*y
print(f"g = {g}")
print(f"g(1, 2) = {g.subs({x: 1, y: 2})} \n")


f3 = (x*y + 3*x + 2*y)**2
print("f3 =", f3.expand(), "\n")  # Foils out the expression and multiplies things out

f4 = ((x + 2*y)**2 - 4*y**2 - x**2)  # Combines like terms and reduces complexity of expression (if possible)
print(f"f4 (unsimplified) = {f4} \nf4 = {f4.simplify()}\n")

# Solving "n" number of equations with "n" unknowns (0 DOF):
eq1 = (x + y - 5)
eq2 = (x - y + 3)
print("eq1 =", eq1, "\n" "eq2 =", eq2)
sol = solve((eq1, eq2), (x, y))
print(f"sol = {sol}")

# Note: the result "sol" is a DICTIONARY, with the input variables (x,y) as "keys", and (2,3) as values for the corresponding keys:
print("x =", sol[x])
print("y =", sol[y])

# Or:
print(f'The solution is x = {sol[x]}, y = {sol[y]}')
print('The solution is x = {}, y = {} \n'.format(sol[x], sol[y]))

# Taking the derivative of composite functions:
g = (x ** 2 + 2 * y - x)
f = (5 * x) * g
print(f"g = {g} \nf = {f} \n")
f_prime_x = f.diff(x)
print(f"f_prime_x(unexpanded) = {f_prime_x}")  # Deriving it this way without expanding (foiling the 5*x term inside the "g" function) gives you unsimplified derivative

# Need to expand first (foil out):
f_prime_x = f_prime_x.expand()
f_prime_y = f.diff(y)
f_prime_x_y = f.diff(x, y)
print(f"f_prime_x = {f_prime_x} \nf_prime_y = {f_prime_y} \nf_prime_x_y = {f_prime_x_y} \n\n")


# To integrate functions (indefinite):
h = y + 3*x**2 + 2*x + 10
h_int = h.integrate(x, y)  # Recall: Order of integration does NOT matter!
print(f"h = {h} \nh_int = {h_int} \n")

# To find definite integrals:
h_int = h.integrate((x, 0, 1), (y, 0, 1))
print(f"0∫1 0∫1 h ∂x∂y = {h_int} \n\n")

# To find the Limits of a function: "limit(f, x, target)"
lim = limit(sin(x)/x, x, 0)  # lim[x->0]: 1 <= sin(x)/x <= cos(x)
print(f"lim = {lim}")

# Can also introduce infinity (sym.inf):
fx = 1/x
lim1 = limit(fx, x, 0)
print(f"lim1 = {lim1}")

lim2 = limit(fx, x, sym.oo)
print(f"lim2 = {lim2} \n\n")


# To differentiate existing mathematical functions, need to import them from "sym":
print(f"∂sin(2x)/∂x = {sym.sin(2*x).diff(x)}")
print(f"∂tan(x)/∂x = {sym.tan(x).diff(x)} \n\n")


# To find higher order derivatives:
h = x**4 + x**3/6 + 10 * x
h3 = h.diff(x, 3)
print(f"h = {h} \nh''' = {h3} \n\n")


# To apply Taylor series expansion: "f.series(x)"
print(f"cos(x-0) = {sym.cos(x).series(x, x0=0, n=6)} \n")  # "n" = number if terms = 6 by default, "x0" = point of interest (a) = 0 by default
print(f"e^x = {sym.exp(x).series(x)} \n\n")

# To factor polynomials:
print(sym.factor(x ** 4 - 3 * x ** 2 + 1))
