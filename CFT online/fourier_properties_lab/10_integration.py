"""
===============================================================================
PROPERTY 10: TIME INTEGRATION
Formula: CFT{ int x(tau) dtau } = X(f) / (j * 2 * pi * f)  (for X(0) = 0)
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

t = np.linspace(-5, 5, 3000)
f = np.linspace(-4, 4, 1000)
dt = t[1] - t[0]

x = -2.0 * t * np.exp(-t**2)
y = np.cumsum(x) * dt

x_f = compute_cft(x, t, f)
y_f = compute_cft(y, t, f)

with np.errstate(divide='ignore', invalid='ignore'):
    y_theory = x_f / (1j * 2 * np.pi * f)
    y_theory[f == 0] = 0.0

mask = np.abs(f) > 0.1
mse_mag = np.mean((np.abs(y_f)[mask] - np.abs(y_theory)[mask]) ** 2)

print(f"[10. Integration] Magnitude MSE (f != 0) : {mse_mag:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, x, 'k--', label='x(t)'); axs[0, 0].plot(t, y, 'b', label=r'$\int x(\tau)d\tau$'); axs[0, 0].legend(); axs[0, 0].set_title('Integrated Signal')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f[mask], np.angle(y_f)[mask], 'b', label='Numerical'); axs[1, 0].plot(f[mask], np.angle(y_theory)[mask], 'g--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase')
axs[1, 1].plot(f[mask], np.abs(np.abs(y_f)[mask] - np.abs(y_theory)[mask]), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
