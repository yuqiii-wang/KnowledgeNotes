import numpy as np
import matplotlib.pyplot as plt

# Define the tiers for each institution
tiers = [
    {'institution': 'A', 'min': 1, 'max': 10, 'spread': 10},
    {'institution': 'A', 'min': 10, 'max': 100, 'spread': 8},
    {'institution': 'A', 'min': 100, 'max': float('inf'), 'spread': 6},
    {'institution': 'B', 'min': 0.1, 'max': 0.5, 'spread': 20},
    {'institution': 'B', 'min': 0.5, 'max': 1, 'spread': 12},
    {'institution': 'B', 'min': 1, 'max': 7.5, 'spread': 10},
    {'institution': 'B', 'min': 7.5, 'max': 20, 'spread': 9},
    {'institution': 'B', 'min': 20, 'max': 110, 'spread': 8.5},
    {'institution': 'B', 'min': 110, 'max': float('inf'), 'spread': 6},
]

# Generate quantities from 0.1 to 200 million with a step of 0.1
quantities = np.arange(0.1, 200.1, 0.1)
min_spreads = []
institutions_list = []

for q in quantities:
    applicable_tiers = []
    for tier in tiers:
        if tier['min'] <= q:
            if tier['max'] == float('inf') or q < tier['max']:
                applicable_tiers.append(tier)
    if applicable_tiers:
        min_spread = min(t['spread'] for t in applicable_tiers)
        min_tiers = [t for t in applicable_tiers if t['spread'] == min_spread]
        institutions = list(set([t['institution'] for t in min_tiers]))
    else:
        min_spread = None
        institutions = []
    min_spreads.append(min_spread)
    institutions_list.append(institutions)

# Detect turning points where the spread changes
turning_points = []
prev_spread = None
for i in range(1, len(quantities)):
    current_spread = min_spreads[i]
    if current_spread != min_spreads[i-1]:
        x_turn = quantities[i]
        y_turn = current_spread
        institutions = institutions_list[i]
        turning_points.append((x_turn, y_turn, institutions))

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(quantities, min_spreads, color='blue', label='Lowest Spread (bp)')
ax = plt.gca()

# Draw horizontal light lines and annotate institutions at turning points
seen_y = set()
for x_turn, y_turn, institutions in turning_points:
    if y_turn not in seen_y:
        ax.axhline(y=y_turn, color='gray', linestyle='--', alpha=0.3)
        seen_y.add(y_turn)
    # Annotate institution, rate, and quantity on the right
    ax.annotate(
        f"{', '.join(institutions)}: {y_turn} bp\n@ {x_turn:.1f}M",
        xy=(200, y_turn),
        xycoords='data',
        xytext=(5, 0),
        textcoords='offset points',
        va='center',
        ha='left',
        fontsize=8,
        color='darkgreen',
        arrowprops=dict(arrowstyle="-", color='gray', alpha=0.5)
    )

plt.xlabel('Quantity (million USD)')
plt.ylabel('Spread over SOFR (basis points)')
plt.title('Lowest Available Spread vs Quantity with Institution Annotations')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xlim(0, 200)
plt.ylim(0, 22)
plt.show()