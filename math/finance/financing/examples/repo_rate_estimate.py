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
base_quantities = np.arange(0.1, 200.1, 0.1)
min_spreads = []
institutions_list = []

# Right-shift by 35 + 5 + 18 = 58 million
total_shift = 35 + 5 + 18
quantities = base_quantities + total_shift

for q in base_quantities:
    applicable = [t for t in tiers if t['min'] <= q < (t['max'] if t['max'] != float('inf') else q+1)]
    if applicable:
        min_sp = min(t['spread'] for t in applicable)
        min_spreads.append(min_sp)
        institutions_list.append(list({t['institution'] for t in applicable if t['spread'] == min_sp}))
    else:
        min_spreads.append(np.nan)
        institutions_list.append([])
min_spreads = np.array(min_spreads)

# Detect turning points where the spread changes
turning_points = []
prev_spread = None
for i in range(0, len(quantities)):
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
        arrowprops=dict(arrowstyle="-", color='gray', alpha=0.)
    )

ax.text(0.1, 1, '    35mil at t+0', color='orange', va='bottom', ha='left')
x_t2 = 35
y_t2 = np.interp(x_t2, quantities, min_spreads)
ax.axvline(x=x_t2, color='gray', linestyle='--', alpha=0.2)
x_t4 = 35+5
y_t4 = np.interp(x_t4, quantities, min_spreads)
ax.axvline(x=x_t4, color='gray', linestyle='--', alpha=0.2,)
x_borrow = 35+5+18
y_borrow = np.interp(x_borrow, quantities, min_spreads)
ax.axvline(x=x_borrow, color='gray', linestyle='--', alpha=0.2)

ticks = list(ax.get_xticks()) + [x_t2, x_t4, x_borrow]
ticks = sorted(set(ticks))
labels = []
for t in ticks:
    if np.isclose(t, x_t2):
        labels.append('t+2 available new 5mil')
    elif np.isclose(t, x_t4):
        labels.append('t+4 available new 18mil')
    elif np.isclose(t, x_borrow):
        labels.append('need to borrow from external')
    else:
        labels.append(f'{int(t)}')
ax.set_xticks(ticks)
ax.set_xticklabels(labels, rotation=-45)
for lbl in ax.get_xticklabels():
    if 'available' in lbl.get_text() or 'need' in lbl.get_text():
        lbl.set_rotation(90)
        lbl.set_color('orange')
        lbl.set_verticalalignment('bottom')
        lbl.set_y(0.1)
    else:
        lbl.set_rotation(45)

plt.xlabel('Quantity (million USD)')
plt.ylabel('Spread over SOFR (basis points)')
plt.title('Lowest Available Spread vs Quantity with Institution Borrow')
plt.grid(False, linestyle='--', alpha=0.1)
plt.legend()
plt.xlim(0, 200)
plt.ylim(0, 22)
plt.show()