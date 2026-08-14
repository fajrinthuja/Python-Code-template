"""
===============================================================================
PROPERTY 04: TIME SCALING
Formula: CFT{ x(a * t) } = (1 / |a|) * X(f / a)
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

t = np.linspace(-5, 5, 2500)
f = np.linspace(-10, 10, 1000)
a = 2.0

x = np.exp(-1.0 * t**2)
y = np.exp(-1.0 * (a * t)**2)

x_f = compute_cft(x, t, f)
y_f = compute_cft(y, t, f)
y_theory = (1.0 / np.abs(a)) * interp_complex(f / a, f, x_f)

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
mse_phase = np.mean((np.unwrap(np.angle(y_f)) - np.unwrap(np.angle(y_theory))) ** 2)

print(f"[04. Time Scaling] Magnitude MSE : {mse_mag:.6e}")
print(f"[04. Time Scaling] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, x, 'k--', label='x(t)'); axs[0, 0].plot(t, y, 'b', label=f'x({a}t)'); axs[0, 0].legend(); axs[0, 0].set_title('Time Domain')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f, np.angle(y_f), 'b', label='Numerical'); axs[1, 0].plot(f, np.angle(y_theory), 'r--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
