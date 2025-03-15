import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configuration: 256-dimensional vector gives 128 pairs.
d = 256
pairs = d // 2  # 128 pairs
base = 10000.0

# Compute rotary frequencies for each pair:
# θ_i = 1 / (base^(2*i/d))
freqs = 1.0 / (base ** (2 * np.arange(pairs) / d))

# For the bottom plot, select three frequency indices.
selected_indices = [63, 95, 127]
# x-axis range from 0 to max_wavelength (for the selected lowest frequency)
max_wavelength = (2 * np.pi) / (1/10000**(127/128))
max_wavelength_half = max_wavelength / 2
max_wavelength_tenth = max_wavelength / 10
ps_bottom = np.linspace(0, max_wavelength, num=1000)

# Define relative positions for the upper plots:
# Upper Left: 0 to max_wavelength_tenth
ps1 = np.linspace(0, max_wavelength_tenth, num=2000)
# Upper Right: 0 to 64k
ps2 = np.linspace(0, 64e3, num=2000)

# Compute total dot product for each relative position:
# Each pair contributes 2*cos(θ*(n-m))
dot_total1 = np.array([2 * np.sum(np.cos(p * freqs)) for p in ps1])
dot_total2 = np.array([2 * np.sum(np.cos(p * freqs)) for p in ps2])

# Prepare green colors using the recommended colormap access:
cmap = mpl.colormaps["Greens"]
# We'll choose values between 0.2 (lighter) and 0.6 (darker) for our curves.
norm_values = np.linspace(0.2, 0.6, len(selected_indices))
colors_bottom = [cmap(norm) for norm in norm_values]

# Create the figure with three subplots:
fig = plt.figure(figsize=(12, 8))
# GridSpec: top row has two columns; bottom row spans both.
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

# Upper Left: Dot Product from 0 to 4k (light blue)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(ps1, dot_total1, color="lightblue")
ax1.set_title(r"RoPE $\mathbf{q}^{\top}_m\mathbf{k}_n$ ($|n - m|$ from 0 to %.2f k)" % (max_wavelength_tenth / 1000))
ax1.set_xlabel(r"Relative Position $|n - m|$")
ax1.set_ylabel(r"$\mathbf{q}^{\top}_m\mathbf{k}_n$")
ax1.grid(True)

# Upper Right: Dot Product from 0 to 128k (light coral)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(ps2, dot_total2, color="lightcoral")
ax2.set_title(r"RoPE $\mathbf{q}^{\top}_m\mathbf{k}_n$ ($|n - m|$ from 0 to 64k)")
ax2.set_xlabel(r"Relative Position $|n - m|$")
ax2.set_ylabel(r"$\mathbf{q}^{\top}_m\mathbf{k}_n$")
ax2.grid(True)

# --- Add horizontal bracket in the upper right plot ---
# We'll draw the bracket in data coordinates so it sits within the first max_wavelength_tenth region.
# First, get the current y-limits to position the bracket near the top.
y_min, y_max = ax2.get_ylim()
# Place the bracket at 35% of the way from y_min to y_max.
y_bracket = y_min + 0.35*(y_max - y_min)

x_start = 0
x_end = max_wavelength_tenth  # The max_wavelength_tenth region

# Draw a horizontal line for the bracket.
ax2.plot([x_start, x_end], [y_bracket, y_bracket], color='black', lw=2)
# Draw vertical ticks at the ends.
ax2.plot([x_start, x_start], [y_bracket, y_bracket - (y_max-y_min)*0.02], color='black', lw=2)
ax2.plot([x_end, x_end], [y_bracket, y_bracket - (y_max-y_min)*0.02], color='black', lw=2)
# Place the label "max_wavelength_tenth" centered above the bracket.
ax2.text((x_start + x_end) / 2, y_bracket + (y_max-y_min)*0.01, r"%.2f k" % (max_wavelength_tenth / 1000),
         ha="center", va="bottom", fontsize=10)
# --- Add description text to the right of the upper right plot ---
ax2.text(0.95, 0.95, r"Despite $2\pi/\theta_{\text{max}}\approx %.2f k$, effective range stops early at %.2f k" %
                    (max_wavelength / 1000, max_wavelength_tenth / 1000), 
         transform=ax2.transAxes, ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

# Bottom: Individual cosine contributions over 0 to max_wavelength.
ax3 = fig.add_subplot(gs[1, :])
for i, idx in enumerate(selected_indices):
    theta = freqs[idx]
    period = 2 * np.pi / theta  # period for the current frequency
    curve = 2 * np.cos(ps_bottom * theta)
    label = (r"$\theta_{\mathrm{freq}\, %d}=10000^{-\frac{2\, \times %d}{%d}},\quad "
             r"\lambda_{\mathrm{freq}\, %d}=\frac{2\pi}{\theta}\approx %5d$"
             % (idx, idx, d, idx, period))
    ax3.plot(ps_bottom, curve, label=label, color=colors_bottom[i])

# Draw a vertical dashed line at x = max_wavelength.
dim_max_freq = d/2-1
ax3.axvline(max_wavelength, color='gray', linestyle=':', 
    label=r"$\lambda_{\mathrm{freq}\, %d}=\frac{2\pi}{\theta_{\text{max}}}\approx %.2f k$" % (dim_max_freq, max_wavelength / 1000))
ax3.set_title("Individual Cosine Contributions (Selected Frequencies)")
ax3.set_xlabel(r"Relative Position $|n - m|$")
ax3.set_ylabel("2*cos(θ*(n - m))")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()
