import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

# Define control points (knots)
control_points = np.array([
    [0, 0],
    [1, 1],
    [2, 3],
    [4, 1]
])
x_control, y_control = control_points[:, 0], control_points[:, 1]
x_control = x_control[:, np.newaxis]  # RBFInterpolator requires 2D input for points

# Create a Thin Plate Spline interpolator
# Rbf with function='thin_plate' computes the TPS interpolation
lambda_reg = 0
rbf = RBFInterpolator(x_control, y_control, kernel='thin_plate_spline', smoothing=lambda_reg)

# Generate a smooth range of x values to interpolate
grid_x = np.linspace(min(x_control), max(x_control), 500)

# Interpolate y values using the Rbf interpolator
grid_y = rbf(grid_x)

# Plot the control points and the resulting spline curve
plt.figure(figsize=(8, 6))
plt.plot(grid_x, grid_y, label="Spline Curve", color="blue")
plt.scatter(x_control, y_control, color="red", label="Control Points")
plt.title(f"Thin Plate Spline Curve (Smoothness: {lambda_reg})")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
