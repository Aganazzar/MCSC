import sympy as sp

x = sp.symbols('x')
#f = sp.sin(x)*sp.exp(x) + x**2 + 10*x + sp.cos(x)
f= sp.exp(x) + x**2 + 10*x

# Try to find the roots
roots = sp.solve(sp.Eq(f, 0), x)
print("Roots:", roots)
