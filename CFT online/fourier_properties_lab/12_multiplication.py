"""
===============================================================================
PROPERTY 12: TIME MULTIPLICATION (FREQUENCY CONVOLUTION)
Formula: CFT{ x1(t) * x2(t) } = X1(f) conv X2(f)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_cft(signal, t, f):
    cft = np.zeros(len(f), dtype=complex)
    for i, fi in enumerate(f):
        kernel = np.exp(-1j * 2 * np.pi * fi * t)
        cft[i] = np.trapezoid(signal * kernel, t) if hasattr(np, 'trapezoid') else np.trapz(signal * kernel, t)
    return cft

t = np.linspace(-5, 5, 2000)
f = np.linspace(-8, 8, 1000)
df = f[1] - f[0]

x1 = np.exp(-1.0 * t**2)
x2 = np.cos(2 * np.pi * 2.0 * t)
y = x1 * x2

X1_f = compute_cft(x1, t, f)
X2_f = compute_cft(x2, t, f)
y_f = compute_cft(y, t, f)
y_theory = np.convolve(X1_f, X2_f, mode='same') * df

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)

print(f"[12. Multiplication] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, y, 'b'); axs[0, 0].set_title(r'Product $y(t) = x_1(t)x_2(t)$')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Frequency Convolution')
axs[1, 0].plot(f, np.angle(y_f), 'b'); axs[1, 0].set_title('Phase Spectrum')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
