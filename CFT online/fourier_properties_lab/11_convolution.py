"""
===============================================================================
PROPERTY 11: TIME CONVOLUTION
Formula: CFT{ x1(t) * x2(t) } = X1(f) * X2(f)
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
f = np.linspace(-5, 5, 1000)
dt = t[1] - t[0]

x1 = np.where(np.abs(t) <= 0.5, 1.0, 0.0)
x2 = np.where(np.abs(t) <= 0.5, 1.0, 0.0)
y = np.convolve(x1, x2, mode='same') * dt

X1_f = compute_cft(x1, t, f)
X2_f = compute_cft(x2, t, f)
y_f = compute_cft(y, t, f)
y_theory = X1_f * X2_f

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
mse_phase = np.mean((np.angle(y_f) - np.angle(y_theory)) ** 2)

print(f"[11. Convolution] Magnitude MSE : {mse_mag:.6e}")
print(f"[11. Convolution] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, x1, 'k--', label='x1=x2 (Rect)'); axs[0, 0].plot(t, y, 'b', label='y = x1*x2 (Triangle)'); axs[0, 0].legend(); axs[0, 0].set_title('Time Domain Convolution')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f, np.angle(y_f), 'b'); axs[1, 0].set_title('Phase Spectrum')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
