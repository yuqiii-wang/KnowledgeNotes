import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline, make_interp_spline, UnivariateSpline
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# Simulate yield data for various maturities with multiple observations per tenor
maturities = np.array([7, 14, 30, 60, 90, 180, 365, 730, 1095, 1825])  # days
years = maturities / 365
true_yields = np.array([2.5, 2.6, 2.7, 2.9, 3.0, 3.3, 3.6, 4.0, 4.2, 4.5])  # in %

# Generate multiple sample points per tenor
np.random.seed(42)
num_samples = 5  # Number of historical observations per tenor
total_num_samples = num_samples*2
noise_std = 0.00015  # Yield noise standard deviation
trend_std = 0.0003  # Daily trend variation

# Create extended dataset with historical observations
all_years = np.repeat(years, total_num_samples)
all_yields = np.zeros_like(all_years)

for i in range(len(years)):
    
    # Create trend and noise components
    bid_trend = np.random.normal(trend_std, trend_std, num_samples) * 365  # Annualized trend
    ask_trend = np.random.normal(-trend_std, trend_std, num_samples) * 365  # Annualized trend
    base_yield = true_yields[i]
    noise = np.random.normal(0, noise_std, num_samples)
    
    # Simulate historical yields
    all_yields[i*total_num_samples:(i+1)*total_num_samples-num_samples] = (
        base_yield + ask_trend + noise
    )
    all_yields[(i+1)*total_num_samples-num_samples:(i+1)*total_num_samples] = (
        base_yield + bid_trend + noise
    )

# Fit linear regressions for each tenor
linear_regression_curves = np.zeros_like(true_yields)

for i, tenor in enumerate(years):
    mask = all_years == tenor
    y = all_yields[mask]
    X = np.array([0] * total_num_samples).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    linear_regression_curves[i] = model.predict([[0]])[0]  # Current time (t=0)

# Original interpolation and models (using true_yields) -------------------------

# Hermite interpolation
dy = np.gradient(true_yields, years)
hermite = CubicHermiteSpline(years, true_yields, dy)

# Cubic B-spline
b_spline = make_interp_spline(years, true_yields, k=3)

# Smooth spline
smooth_spline = UnivariateSpline(years, true_yields, s=1.0)

# Nelson-Siegel model
def nelson_siegel(tau, beta0, beta1, beta2, lambd):
    term1 = (1 - np.exp(-lambd * tau)) / (lambd * tau)
    term2 = term1 - np.exp(-lambd * tau)
    return beta0 + beta1 * term1 + beta2 * term2

popt_ns, _ = curve_fit(nelson_siegel, years, true_yields, 
                       bounds=([0, -10, -10, 0.01], [10, 10, 10, 10]))

# Nelson-Siegel-Svensson model
def nelson_siegel_svensson(tau, beta0, beta1, beta2, beta3, lambda1, lambda2):
    term1 = (1 - np.exp(-lambda1 * tau)) / (lambda1 * tau)
    term2 = term1 - np.exp(-lambda1 * tau)
    term3 = (1 - np.exp(-lambda2 * tau)) / (lambda2 * tau) - np.exp(-lambda2 * tau)
    return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3

popt_nss, _ = curve_fit(nelson_siegel_svensson, years, true_yields,
                        bounds=([0, -10, -10, -10, 0.01, 0.01], [10, 10, 10, 10, 10, 10]))

# New method: Linear regression per tenor with piecewise linear interpolation
lr_spline = make_interp_spline(years, linear_regression_curves, k=1)

# Generate smooth time grid
t_grid = np.linspace(years.min(), years.max(), 500)

# Evaluate all models
hermite_y = hermite(t_grid)
b_spline_y = b_spline(t_grid)
smooth_y = smooth_spline(t_grid)
ns_y = nelson_siegel(t_grid, *popt_ns)
nss_y = nelson_siegel_svensson(t_grid, *popt_nss)
lr_y = lr_spline(t_grid)

# Plot results
plt.figure(figsize=(12, 8))

# ask_all_yields = all_yields[np.arange(0, 100, 10).repeat(5) + np.tile(np.arange(5), 10)]
# ask_all_years = all_years[np.arange(0, 100, 10).repeat(5) + np.tile(np.arange(5), 10)]
# bid_all_yields = all_yields[np.arange(5, 100, 10).repeat(5) + np.tile(np.arange(5), 10)]
# bid_all_years = all_years[np.arange(5, 100, 10).repeat(5) + np.tile(np.arange(5), 10)]
# plt.plot(bid_all_years, bid_all_yields, '^', color='gray', alpha=0.6, 
#          label='Best Buy Quote')
# plt.plot(ask_all_years, ask_all_yields, 'v', color='gray', alpha=0.6, 
#          label='Best Sell Quote')

plt.plot(years, true_yields, 'ko', label='Yield Rate')

plt.plot(t_grid, hermite_y, label='Hermite Interpolation', alpha=0.6)
plt.plot(t_grid, b_spline_y, label='Cubic B-Spline', alpha=0.6)
plt.plot(t_grid, smooth_y, label='Smooth Spline', alpha=0.6)
plt.plot(t_grid, ns_y, label='Nelson-Siegel', alpha=0.6)
plt.plot(t_grid, nss_y, label='Nelson-Siegel-Svensson', alpha=0.6)
plt.plot(t_grid, lr_y, '--', linewidth=2, label='Linear Regression per Tenor', alpha=0.6)

plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (%)')
plt.title('Yield Curve Construction Methods Comparison')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()