import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
n_paths = 10
n_steps = 500
T = 2.0  # in years
dt = T / n_steps
time_grid = np.linspace(0, T, n_steps + 1)

# Vasicek parameters
a_v, b_v, sigma_v, r0_v = 0.15, 0.05, 0.01, 0.03

# Hull-White parameters
a_hw, sigma_hw, r0_hw = 0.15, 0.01, 0.03
theta_hw = lambda t: 0.05 + 0.01 * np.sin(2 * np.pi * t)

# Ho-Lee parameters
theta_hl = 0.02  # constant drift
sigma_hl, r0_hl = 0.01, 0.03

def simulate_vasicek():
    rates = np.zeros((n_paths, n_steps + 1))
    rates[:, 0] = r0_v
    for i in range(n_steps):
        dr = a_v * (b_v - rates[:, i]) * dt + sigma_v * np.sqrt(dt) * np.random.randn(n_paths)
        rates[:, i + 1] = rates[:, i] + dr
    return rates

def simulate_hull_white():
    rates = np.zeros((n_paths, n_steps + 1))
    rates[:, 0] = r0_hw
    for i in range(n_steps):
        t = time_grid[i]
        dr = (theta_hw(t) - a_hw * rates[:, i]) * dt + sigma_hw * np.sqrt(dt) * np.random.randn(n_paths)
        rates[:, i + 1] = rates[:, i] + dr
    return rates

def simulate_ho_lee():
    rates = np.zeros((n_paths, n_steps + 1))
    rates[:, 0] = r0_hl
    for i in range(n_steps):
        dr = theta_hl * dt + sigma_hl * np.sqrt(dt) * np.random.randn(n_paths)
        rates[:, i + 1] = rates[:, i] + dr
    return rates

# Run simulations
vasicek_paths = simulate_vasicek()
hw_paths = simulate_hull_white()
hl_paths = simulate_ho_lee()

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models = {
    'Vasicek': vasicek_paths,
    'Ho-Lee': hl_paths,
    'Hull-White': hw_paths,
}

for ax, (name, paths) in zip(axes, models.items()):
    mean = paths.mean(axis=0)
    std = paths.std(axis=0)
    # Shading for mean ± std
    ax.fill_between(time_grid, mean - std, mean + std, alpha=0.3)
    # Individual paths
    for i in range(n_paths):
        ax.plot(time_grid, paths[i], linewidth=0.8)
    # Mean path
    ax.plot(time_grid, mean, linewidth=2)
    ax.set_title(f"{name} Model")
    ax.set_xlabel("Time (years)")
    if name == 'Vasicek':
        ax.set_ylabel("Interest Rate")

plt.tight_layout()
plt.show()
