import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the transformation matrix A
A = np.array([[2, 1], [1, 2]])

# Define the original points
points = np.array([
    [1, 3],
    [1, 2],
    [1, 1],
    [1, 0],
    [1, -1],
    [1, -2],
    [1, -3]
])

# Compute transformed points
transformed_points = np.dot(points, A.T)

# Eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A)

# Animation setup
fig, ax = plt.subplots()
ax.set_xlim(-2, 12)
ax.set_ylim(-6, 6)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid()

# Plot x and y axes
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)

# Plot eigenvectors
for i in range(2):
    vec = eigenvectors[:, i] * eigenvalues[i]  # Scale for better visualization
    ax.plot([-vec[0], vec[0]], [-vec[1], vec[1]], color='plum', linestyle='dashed',
            label=f"Eigenvector [%.3f %.3f], Eigenvalue %.3f" %
                    (eigenvectors[0, i], eigenvectors[1, i], eigenvalues[i]))
# Plot original points
scatter_original, = ax.plot(points[:, 0], points[:, 1], 'o', color='lightblue', label="Original Points")
scatter_transformed, = ax.plot([], [], 'o', color='lightgreen', label="Transforming Points")
ax.legend(loc='lower right')

# Animate transformation
def update(frame):
    t = frame / 30  # interpolation factor (0 to 1)
    intermediate_points = (1 - t) * points + t * transformed_points
    scatter_transformed.set_data(intermediate_points[:, 0], intermediate_points[:, 1])
    return scatter_transformed,

ani = animation.FuncAnimation(fig, update, frames=31, interval=50, blit=True)

# Save the animation as a GIF
ani.save("linear_transform_example.gif", writer="pillow", fps=15)

plt.show()
