import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt

# Input bond data: maturities in years, annual coupon rates, and dirty prices
bonds = [
    {'maturity': 1,  'coupon_rate': 0.02,  'dirty_price': 101.5},
    {'maturity': 2,  'coupon_rate': 0.025, 'dirty_price': 102.2},
    {'maturity': 5,  'coupon_rate': 0.03,  'dirty_price': 104.0},
    {'maturity': 10, 'coupon_rate': 0.035,'dirty_price': 108.7},
    {'maturity': 30, 'coupon_rate': 0.04,  'dirty_price': 120.0},
]

face_value = 100
spot_rates = []

# Bootstrapping zero-coupon (spot) rates
for bond in sorted(bonds, key=lambda x: x['maturity']):
    T = bond['maturity']
    c = bond['coupon_rate'] * face_value  # annual coupon payment
    P = bond['dirty_price']
    
    def price_diff(r):
        # Present value of coupon payments before final maturity
        pv_coupons = sum(
            c / (1 + spot_rates[k])**(k + 1)
            for k in range(len(spot_rates))
        )
        # Present value of final coupon + principal
        pv_final = (c + face_value) / (1 + r)**T
        return pv_coupons + pv_final - P
    
    # Initial guess = coupon rate
    r_guess = bond['coupon_rate']
    r_solved = newton(price_diff, r_guess)
    spot_rates.append(r_solved)

# Prepare data for plotting
maturities = [b['maturity'] for b in sorted(bonds, key=lambda x: x['maturity'])]
spot_percent = [r * 100 for r in spot_rates]

# Plot the bootstrapped zero-coupon yield curve
plt.figure()
plt.plot(maturities, spot_percent, marker='o')
plt.xlabel('Maturity (years)')
plt.ylabel('Spot Rate (%)')
plt.title('Bootstrapped Zero-Coupon Yield Curve')
plt.grid(True)
plt.show()
