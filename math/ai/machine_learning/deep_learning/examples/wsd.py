import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# Define the enhanced non-convex loss landscape function with local valleys, peaks, and ravines.
def f(x, y):
    """
    Complex non-convex function with local valleys, peaks, and ravines.
    """
    return (x**2 + y**2 +
         0.7 * np.sin(5*x) * np.cos(5*y) +
         0.4 * np.sin(7*x + 2*y) +
         0.4 * np.sin(3*x) * np.cos(3*y) +
         0.15 * np.cos(10*x) * np.sin(10*y) +
         0.25 * np.exp(-0.5*(x**2 + y**2)) +
         0.7 * np.exp(-0.5*(x-2)**2 - (y-2)**2) +
         0.5 * np.cos(5*(x-2)) * np.sin(5*(y-2)) +
         0.4 * np.cos(4*(x+1)) * np.sin(4*(y+1)) +
         0.45 * np.exp(-0.05*(x+1)**2 - (y+1)**2) + 0.6)

# Compute the gradient of f with respect to x and y.
def grad_f(x, y):
    df_dx = (2*x +
            0.7*5*np.cos(5*x)*np.cos(5*y) +
            0.4*7*np.cos(7*x + 2*y) +
            0.4*3*np.cos(3*x)*np.cos(3*y) +
            0.15*(-10)*np.sin(10*x)*np.sin(10*y) +
            0.25*(-x)*np.exp(-0.5*(x**2 + y**2)) +
            0.7*(-(x-2))*np.exp(-0.5*(x-2)**2 - (y-2)**2) +
            0.5*(-5)*np.sin(5*(x-2))*np.sin(5*(y-2)) +
            0.4*(-4)*np.sin(4*(x+1))*np.sin(4*(y+1)) +
            0.45*(-0.1*(x+1))*np.exp(-0.05*(x+1)**2 - (y+1)**2))
    
    df_dy = (2*y +
            0.7*(-5)*np.sin(5*x)*np.sin(5*y) +
            0.4*2*np.cos(7*x + 2*y) +
            0.4*(-3)*np.sin(3*x)*np.sin(3*y) +
            0.15*10*np.cos(10*x)*np.cos(10*y) +
            0.25*(-y)*np.exp(-0.5*(x**2 + y**2)) +
            0.7*(-(y-2))*np.exp(-0.5*(x-2)**2 - (y-2)**2) +
            0.5*5*np.cos(5*(x-2))*np.cos(5*(y-2)) +
            0.4*4*np.cos(4*(x+1))*np.cos(4*(y+1)) +
            0.45*(-2*(y+1))*np.exp(-0.05*(x+1)**2 - (y+1)**2))
    
    return np.array([df_dx, df_dy])

# -------------------------------
# Adam optimizer with momentum and learning rate decay
# -------------------------------
def adam_optimizer(f, grad_f, initial_params, max_lr=0.1, 
                   beta1=0.9, beta2=0.95, epsilon=1e-8, max_iter=1000, 
                   warmup_steps=50, stable_steps=300):
    params = np.array(initial_params)
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    t = 0
    trajectory = []
    loss_history = []
    
    for iteration in range(max_iter):
        t += 1
        # Compute gradients
        grads = grad_f(params[0], params[1])
        
        # Update biased first and second moment estimates
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads ** 2)
        
        # Correct bias in first and second moment estimates
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update parameters with learning rate

        if t < warmup_steps:  # Warmup
            lr = max_lr * (t/warmup_steps)
        elif t < warmup_steps + stable_steps:  # Stable
            lr = max_lr
        else:  # Decay
            decay_steps = t - warmup_steps - stable_steps
            lr = max_lr * np.exp(-0.1 * decay_steps / 1000)

        params -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

        # Save the trajectory and loss history
        trajectory.append((params[0], params[1], f(params[0], params[1])))
        loss_history.append(f(params[0], params[1]))
        
        # Print the loss at each iteration
        if iteration % 50 == 0:  # print every 50 steps
            print(f"Iteration {iteration}, Loss: {loss_history[-1]}, Learning rate: {lr}")
        
    return np.array(trajectory), loss_history, params

# -------------------------------
# Visualization
# -------------------------------

# Generate 3D loss landscape
X = np.linspace(-2, 2, 80)
Y = np.linspace(-2, 2, 80)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)

# Initial guess for the optimizer
initial_params = [-1.5, -1.5]

# Optimize using Adam with dynamic learning rate
trajectory, loss_history, final_params = adam_optimizer(f, grad_f, initial_params)

# Plot the 3D surface and trajectory
fig = plt.figure(figsize=(14, 6))

# 3D Surface Plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='r', marker='o', markersize=3, linewidth=1.5, label='Trajectory')
ax1.set_title('3D Loss Landscape with Adam+WSD Optimization')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.legend()

# Contour Plot with Trajectory
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
ax2.plot(trajectory[:, 0], trajectory[:, 1], color='r', marker='o', markersize=3, linewidth=1.5, label='Trajectory')
ax2.set_title('Contour Map with Optimization Trajectory')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
plt.colorbar(contour, ax=ax2)

plt.tight_layout()
plt.show()

# Print final optimized parameters and loss
print(f"Final optimized parameters: x = {final_params[0]}, y = {final_params[1]}")
print(f"Final loss value: {f(final_params[0], final_params[1])}")
