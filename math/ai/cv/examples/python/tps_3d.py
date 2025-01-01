import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator

# Define 2D scattered data points
np.random.seed(42)  # For reproducibility
num_points = 20
x = np.random.rand(num_points, 2) * 2 - 1  # Random points in [-1, 1]^2
y = np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])  # Values for the 2D points

# Create an RBF interpolator with a Gaussian kernel
interpolator = RBFInterpolator(x, y, kernel='gaussian', epsilon=0.5)

# Generate a regular grid in 2D space for evaluation
grid_x, grid_y = np.meshgrid(
    np.linspace(-1, 1, 100), 
    np.linspace(-1, 1, 100)
)
grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

# Interpolate values on the grid
grid_values = interpolator(grid_points)

# Reshape interpolated values to the grid shape for visualization
grid_values = grid_values.reshape(grid_x.shape)

# Visualization: 3D Surface Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(
    grid_x, grid_y, grid_values, 
    cmap='viridis', alpha=0.8
)

# Scatter the original points
ax.scatter(x[:, 0], x[:, 1], y, color='red', label='Data Points', s=50)

# Add labels, legend, and color bar
ax.set_title('2D RBF Interpolation with 3D Visualization')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Interpolated Value')
fig.colorbar(surf, ax=ax, label='Interpolated Value')
ax.legend()

# Enable interactive rotation
plt.show()
