import numpy as np
import matplotlib.pyplot as plt

# Define the range for t-values
t_values = np.linspace(-5, 5, 1000)

# Compute Gaussian (Normal) PDF using NumPy
def gaussian_pdf(t):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-t**2 / 2)

# Compute Student's t-distribution (1 d.o.f.) PDF using NumPy
def student_t_pdf(t, nu=1):
    gamma_numerator = np.sqrt(np.pi)  # Γ(1/2) = √π
    gamma_denominator = 1.0           # Γ(1) = 1
    coefficient = gamma_numerator / (np.sqrt(nu * np.pi) * gamma_denominator)
    return coefficient * (1 + t**2 / nu) ** (-(nu + 1) / 2)

# Compute PDFs
gaussian = gaussian_pdf(t_values)
student_t = student_t_pdf(t_values, nu=1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t_values, gaussian, label="Gaussian (Normal)", linestyle="--")
plt.plot(t_values, student_t, label="Student's t (1 d.o.f.)", linestyle="-")
plt.title("Gaussian vs. Student's t-Distribution (1 d.o.f.)")
plt.xlabel("t")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()