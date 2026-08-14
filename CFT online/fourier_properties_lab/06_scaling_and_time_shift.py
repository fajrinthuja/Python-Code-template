"""
===============================================================================
PROPERTY 06: COMBINED TIME SCALING AND TIME SHIFT
Formula: CFT{ x(a * (t - t0)) } = (1 / |a|) * X(f / a) * exp(-j * 2 * pi * f * t0)
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

def interp_complex(x_new, xp, yp):
    return np.interp(x_new, xp, yp.real) + 1j * np.interp(x_new, xp, yp.imag)

t = np.linspace(-6, 6, 3000)
f = np.linspace(-10, 10, 1000)
a, t0 = 2.0, 1.0

x = np.exp(-1.5 * t**2)
y = np.exp(-1.5 * (a * (t - t0))**2)

x_f = compute_cft(x, t, f)
y_f = compute_cft(y, t, f)
y_theory = (1.0 / np.abs(a)) * interp_complex(f / a, f, x_f) * np.exp(-1j * 2 * np.pi * f * t0)

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
sig_mask = np.abs(y_f) > 1e-3
mse_phase = np.mean((np.unwrap(np.angle(y_f))[sig_mask] - np.unwrap(np.angle(y_theory))[sig_mask]) ** 2)

print(f"[06. Scaling + Time Shift] Magnitude MSE : {mse_mag:.6e}")
print(f"[06. Scaling + Time Shift] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, x, 'k--', label='x(t)'); axs[0, 0].plot(t, y, 'b', label='x(a(t - t0))'); axs[0, 0].legend(); axs[0, 0].set_title('Time Domain')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f, np.unwrap(np.angle(y_f)), 'b', label='Numerical'); axs[1, 0].plot(f, np.unwrap(np.angle(y_theory)), 'g--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
