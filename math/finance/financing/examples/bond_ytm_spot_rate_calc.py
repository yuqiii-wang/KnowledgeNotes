import numpy as np
from scipy.optimize import brentq

# Given bond details
dirty_price = 978.12
coupon_amt = 50.0         # 5% of 1000
face_value = 1000.0

# 1) Solve for YTM: 1020 = 50/(1+r) + 50/(1+r)^2 + 1050/(1+r)^3
def f_ytm(r):
    return (coupon_amt/(1+r) +
            coupon_amt/(1+r)**2 +
            (coupon_amt + face_value)/(1+r)**3
           ) - dirty_price

ytm = brentq(f_ytm, 0.0, 0.20)
print(f"YTM: {ytm*100:.4f}%")

# 2) Bootstrapping spot rates
#   Year 1 spot
r1 = 0.05
print(f"Year 1 spot rate r1: {r1*100:.4f}%")

#   Year 2 spot: solve
#   LHS from 2‑yr par bond at 5.1%: 5.1/1.051 + 105.1/1.051^2
coupon2 = 5.1
final2 = 105.1
lhs2 = coupon2/(1+0.051) + final2/(1+0.051)**2

def f_r2(r2):
    return (coupon2/(1+r1) + final2/(1+r2)**2) - lhs2

r2 = brentq(f_r2, 0.0, 0.20)
print(f"Year 2 spot rate r2: {r2*100:.4f}%")

#   Year 3 spot: solve
#   LHS from 3‑yr par bond at 5.5%: 5.5/1.055 + 5.5/1.055^2 + 105.5/1.055^3
coupon3 = 5.5
final3 = 105.5
lhs3 = (coupon3/(1+0.055) +
        coupon3/(1+0.055)**2 +
        final3/(1+0.055)**3)

def f_r3(r3):
    return (coupon3/(1+r1) +
            coupon3/(1+r2)**2 +
            final3/(1+r3)**3) - lhs3

r3 = brentq(f_r3, 0.0, 0.20)
print(f"Year 3 spot rate r3: {r3*100:.4f}%")

# Present Value calculation
cashflows = [50, 50, 1050]
discount_factors = [(1+r1)**1, (1+r2)**2, (1+r3)**3]

pv = sum(cf / df for cf, df in zip(cashflows, discount_factors))

print(f"Present value: {pv:.2f}")
