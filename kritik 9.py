import sympy as sp

# Define symbols
x, y = sp.symbols('x y')

# (a) Define the function and compute first partial derivatives
f = sp.exp(x) * sp.sin(y) + y**3
partial_f_x = sp.diff(f, x)
partial_f_y = sp.diff(f, y)

print("(a) First partial derivatives:")
print("∂f/∂x =", partial_f_x)
print("∂f/∂y =", partial_f_y, "\n")

# (b) Define g(x,y), find the gradient vector and magnitude at (1,-1)
g = x**2 * y + x * y**2
grad_g = [sp.diff(g, x), sp.diff(g, y)]
magnitude_grad_g = sp.sqrt(grad_g[0]**2 + grad_g[1]**2).subs({x: 1, y: -1})

print("(b) Gradient vector:")
print("∇g =", grad_g)
print("Magnitude at (1, -1) =", magnitude_grad_g, "\n")

# (c) Define h(x,y) and compute second partial derivatives
h = sp.ln(x**2 + y**2)
second_partial_xx = sp.diff(h, x, x)
second_partial_yy = sp.diff(h, y, y)
second_partial_xy = sp.diff(h, x, y)

print("(c) Second partial derivatives:")
print("∂²h/∂x² =", second_partial_xx)
print("∂²h/∂y² =", second_partial_yy)
print("∂²h/∂x∂y =", second_partial_xy)
print("Mixed partial derivatives are symmetric if ∂²h/∂x∂y = ∂²h/∂y∂x, which is", second_partial_xy == sp.diff(h, y, x), "\n")

# (d) Contour plot of j(x,y) = x^3 - 3xy + y^3
import numpy as np
import matplotlib.pyplot as plt

# Define function
j = x**3 - 3*x*y + y**3
j_func = sp.lambdify((x, y), j, 'numpy')

# Create grid
x_vals = np.linspace(-3, 3, 400)
y_vals = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = j_func(X, Y)

# Plot contour
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar()
plt.title(r'Contour plot of $j(x, y) = x^3 - 3xy + y^3$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

# (e) Contour plot of sin(x) * cos(y)
f_new = sp.sin(x) * sp.cos(y)
f_new_func = sp.lambdify((x, y), f_new, 'numpy')

# Compute values
Z_new = f_new_func(X, Y)

# Plot contour
plt.contourf(X, Y, Z_new, levels=50, cmap='coolwarm')
plt.colorbar()
plt.title(r'Contour plot of $f(x, y) = \sin(x) \cos(y)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
