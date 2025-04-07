import numpy as np
import matplotlib.pyplot as plt

# Define float32 and int4 ranges
float_min, float_max = -2.0, 5.0
int4_min, int4_max = -8, 7  # Signed int4 range

# Calculate scale and zero point
scale = (float_max - float_min) / (int4_max - int4_min)  # 7 / 15
zero_point_float = float_min
zero_point_int = round(-float_min / scale) + int4_min  # Asymmetric zero point

print(f"Scale: {scale:.4f}")
print(f"Zero Point (int): {zero_point_int}")

# Quantization function
def quantize(val):
    q = round((val - zero_point_float) / scale) + int4_min
    return np.clip(q, int4_min, int4_max)

# Dequantization function
def dequantize(q):
    return (q - int4_min) * scale + zero_point_float

# Generate float32 values
float_vals = np.linspace(float_min, float_max, 100)
int4_vals = np.array([quantize(f) for f in float_vals])

# Plot
plt.figure(figsize=(8, 5))
plt.plot(float_vals, int4_vals, label='Quantized int4', color='blue')
plt.scatter(float_vals, int4_vals, s=10, color='blue')
plt.title("Asymmetric Quantization: float32 [-2.0, 5.0] → int4 [-8, 7]")
plt.xlabel("float32 value")
plt.ylabel("int4 quantized value")
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.legend()
plt.tight_layout()
plt.show()
