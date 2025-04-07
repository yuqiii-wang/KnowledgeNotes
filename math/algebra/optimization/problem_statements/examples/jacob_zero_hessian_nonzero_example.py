import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Symbolic analysis using SymPy
x, y = sp.symbols('x y')
f = x**2 - y**2  # Saddle point function

# Calculate gradient (Jacobian for scalar function)
gradient = sp.derive_by_array(f, (x, y))
print("Gradient:", gradient)
print("At (0,0):", gradient.subs({x: 0, y: 0}))

# Calculate Hessian
hessian = sp.hessian(f, (x, y))
print("\nHessian:", hessian)
print("At (0,0):", hessian.subs({x: 0, y: 0}))

# Numerical visualization using Matplotlib
x_vals = np.linspace(-1, 1, 30)
y_vals = np.linspace(-1, 1, 30)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X**2 - Y**2

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.scatter(0, 0, 0, color='red', s=100, label='Critical Point (0,0)')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Saddle Point: $f(x, y) = x^2 - y^2$\n' +
             'Zero Gradient but Non-Zero Hessian at (0,0)')
plt.legend()
plt.show()