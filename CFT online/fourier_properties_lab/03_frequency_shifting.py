"""
===============================================================================
PROPERTY 03: FREQUENCY SHIFTING (MODULATION)
Formula: CFT{ x(t) * exp(j * 2 * pi * f0 * t) } = X(f - f0)
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

t = np.linspace(-5, 5, 2000)
f = np.linspace(-15, 15, 1500)
f0 = 4.0

x = np.exp(-1.5 * t**2)
y = x * np.exp(1j * 2 * np.pi * f0 * t)

x_f = compute_cft(x, t, f)
y_f = compute_cft(y, t, f)
y_theory = interp_complex(f - f0, f, x_f)

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
sig_mask = np.abs(y_theory) > 1e-3
mse_phase = np.mean((np.unwrap(np.angle(y_f))[sig_mask] - np.unwrap(np.angle(y_theory))[sig_mask]) ** 2)

print(f"[03. Frequency Shift] Magnitude MSE : {mse_mag:.6e}")
print(f"[03. Frequency Shift] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, y.real, 'b'); axs[0, 0].set_title(r'Re{$y(t) = x(t)e^{j2\pi f_0 t}$}')
axs[0, 1].plot(f, np.abs(x_f), 'k--', label='Base |X(f)|'); axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical |Y(f)|'); axs[0, 1].plot(f, np.abs(y_theory), 'r:', label='Theory |X(f-f0)|'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f, np.angle(y_f), 'b', label='Numerical'); axs[1, 0].plot(f, np.angle(y_theory), 'r--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
