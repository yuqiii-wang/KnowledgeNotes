import numpy as np
import matplotlib.pyplot as plt

# Define periods for the two pulse trains
period1 = 50    # Top pulse train: pulse occurs at sample 49 (0-indexed)
period2 = 300   # Bottom pulse train: pulse occurs at sample 299 (0-indexed)

# Number of periods to display
num_periods1 = 5   # For top pulse train (0 to 249)
num_periods2 = 3   # For bottom pulse train (0 to 899)

# Create sample index arrays
n1 = np.arange(num_periods1 * period1)
n2 = np.arange(num_periods2 * period2)

# Create pulse trains: pulse (value 1) at the last sample of each period, 0 elsewhere.
pulse1 = ((n1 % period1) == (period1 - 1)).astype(float)
pulse2 = ((n2 % period2) == (period2 - 1)).astype(float)

# Dampening (scaling): top signal by 1/50; bottom signal by 1/300 then further reduced by factor 0.5.
pulse1_damped = pulse1 * (1/50)
pulse2_damped = pulse2 * (1/300) * 0.5  # further lowering the amplitude

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

# Plot the top pulse train with a darker blue color
markerline1, stemlines1, _ = ax1.stem(n1, pulse1_damped, linefmt='dodgerblue', markerfmt='o', basefmt=' ')
plt.setp(stemlines1, 'linewidth', 1.5)
plt.setp(markerline1, 'markersize', 6)

# Plot the bottom pulse train with a darker green color
markerline2, stemlines2, _ = ax2.stem(n2, pulse2_damped, linefmt='mediumseagreen', markerfmt='o', basefmt=' ')
plt.setp(stemlines2, 'linewidth', 1.5)
plt.setp(markerline2, 'markersize', 6)

# Set x-axis ticks:
# For top plot: ticks at 0, 50, 100, ... up to the total length.
xticks_top = np.arange(0, num_periods1 * period1 + 1, period1)
ax1.set_xticks(xticks_top)
ax1.set_xticklabels(xticks_top)

# For bottom plot: ticks at 0, 300, 600, ... up to the total length.
xticks_bottom = np.arange(0, num_periods2 * period2 + 1, period2)
ax2.set_xticks(xticks_bottom)
ax2.set_xticklabels(xticks_bottom)

# Add text annotations for each interval:
# For the top plot, place "...49..." in the center between 0 & 50, 50 & 100, etc.
for i in range(len(xticks_top) - 1):
    mid = (xticks_top[i] + xticks_top[i+1]) / 2
    ax1.text(mid, 0.001, '...49...', ha='center', va='bottom', color='gray', fontsize=9)

# For the bottom plot, place "...299..." in the center between 0 & 300, 300 & 600, etc.
for i in range(len(xticks_bottom) - 1):
    mid = (xticks_bottom[i] + xticks_bottom[i+1]) / 2
    ax2.text(mid, 0.0001, '...299...', ha='center', va='bottom', color='gray', fontsize=9)

# Set titles, labels, and adjust y-axis limits
ax1.set_title('Dampened Pulse Train (Period = 50, Scaled by 1/50)')
ax1.set_xlabel('Sample Number')
ax1.set_ylabel('Amplitude (1/50 scale)')
ax1.set_ylim(-0.002, 0.025)
ax1.grid(True)

ax2.set_title('Dampened Pulse Train (Period = 300, Scaled by 1/300)')
ax2.set_xlabel('Sample Number')
ax2.set_ylabel('Amplitude (Scaled 1/300)')
ax2.set_ylim(-0.0005, 0.003)
ax2.grid(True)

plt.tight_layout()
plt.show()
