import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters
n = 50
num_experiments = 1000
mu = 0.5
sigma_squared = 1/12
std_error = np.sqrt(sigma_squared / n)

# Generate samples
samples = np.random.uniform(0, 1, size=(num_experiments, n))
sample_means = np.mean(samples, axis=1)
Z_n = (sample_means - mu) / std_error

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(8, 5))

# Histogram (count) on the right y-axis
ax2 = ax1.twinx()
counts, bins, patches = ax2.hist(Z_n, bins=100, alpha=0.6, color='skyblue',
                                 label='Count of $Z_n$')

# Standard normal density on the left y-axis
x = np.linspace(-4, 4, 200)
ax1.plot(x, stats.norm.pdf(x), 'r-', lw=2, label='Standard Normal $\mathcal{{N}}(0,1)$ PDF')

# Labeling
ax1.set_xlabel("$Z_n$")
ax1.set_ylabel("Density (Standard Normal)", color='lightcoral')
ax2.set_ylabel("Count of $Z_n$", color='deepskyblue')
ax1.tick_params(axis='y',)
ax2.tick_params(axis='y',)
ax1.set_title("CLT Simulation: Standardized Sample Means from Uniform(0,1)")

# Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
legend = ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
    loc='upper left', frameon=True, fancybox=True)
legend.set_zorder(10)  # Bring legend above the grid


# Show plot
plt.grid(True, zorder=0)
plt.tight_layout()
plt.show()
