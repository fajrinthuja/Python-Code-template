"""
===============================================================================
PROPERTY 05: COMBINED TIME SCALING AND FREQUENCY SHIFT
Formula: CFT{ x(a * t) * exp(j * 2 * pi * f0 * t) } = (1 / |a|) * X((f - f0) / a)
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

def base_signal(t_in):
    sq = np.where(np.abs(t_in) <= 1.0, 1.0, 0.0)
    tr = np.where(np.abs(t_in) <= 1.0, 1.0 - np.abs(t_in), 0.0)
    return sq + tr

t = np.linspace(-5, 5, 3000)
f = np.linspace(-15, 15, 1200)
a, f0 = 2.0, 3.0

x = base_signal(t)
y = base_signal(a * t) * np.exp(1j * 2 * np.pi * f0 * t)

x_f = compute_cft(x, t, f)
y_f = compute_cft(y, t, f)
y_theory = (1.0 / np.abs(a)) * interp_complex((f - f0) / a, f, x_f)

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
sig_mask = np.abs(y_theory) > 1e-2
mse_phase = np.mean((np.unwrap(np.angle(y_f))[sig_mask] - np.unwrap(np.angle(y_theory))[sig_mask]) ** 2)

print(f"[05. Scaling + Freq Shift] Magnitude MSE : {mse_mag:.6e}")
print(f"[05. Scaling + Freq Shift] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, y.real, 'b'); axs[0, 0].set_title(r'Re{$y(t) = x(at)e^{j2\pi f_0 t}$}')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f, np.unwrap(np.angle(y_f)), 'b', label='Numerical'); axs[1, 0].plot(f, np.unwrap(np.angle(y_theory)), 'g--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
