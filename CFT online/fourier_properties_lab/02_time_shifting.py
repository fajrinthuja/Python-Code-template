"""
===============================================================================
PROPERTY 02: TIME SHIFTING
Formula: CFT{ x(t - t0) } = X(f) * exp(-j * 2 * pi * f * t0)
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

t = np.linspace(-6, 6, 2500)
f = np.linspace(-5, 5, 1000)
t0 = 1.5

x = np.exp(-2.0 * t**2)
y = np.exp(-2.0 * (t - t0)**2)

x_f = compute_cft(x, t, f)
y_f = compute_cft(y, t, f)
y_theory = x_f * np.exp(-1j * 2 * np.pi * f * t0)

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
sig_mask = np.abs(x_f) > 1e-3
mse_phase = np.mean((np.unwrap(np.angle(y_f))[sig_mask] - np.unwrap(np.angle(y_theory))[sig_mask]) ** 2)

print(f"[02. Time Shift] Magnitude MSE : {mse_mag:.6e}")
print(f"[02. Time Shift] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, x, 'k--', label='x(t)'); axs[0, 0].plot(t, y, 'r', label='x(t - t0)'); axs[0, 0].legend(); axs[0, 0].set_title('Time Domain')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='|Y(f)|'); axs[0, 1].plot(f, np.abs(x_f), 'r--', label='|X(f)|'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude (Invariant)')
axs[1, 0].plot(f, np.unwrap(np.angle(y_f)), 'b', label='Numerical'); axs[1, 0].plot(f, np.unwrap(np.angle(y_theory)), 'g--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase Spectrum')
axs[1, 1].plot(f, np.abs(np.unwrap(np.angle(y_f)) - np.unwrap(np.angle(y_theory))), 'purple'); axs[1, 1].set_title('Phase Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
