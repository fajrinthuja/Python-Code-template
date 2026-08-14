"""
===============================================================================
PROPERTY 01: LINEARITY / SUPERPOSITION
Formula: CFT{ a*x1(t) + b*x2(t) } = a*X1(f) + b*X2(f)
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
f = np.linspace(-10, 10, 1000)

x1 = np.exp(-1.0 * t**2)
x2 = np.where(np.abs(t) <= 1.0, 1.0, 0.0)
a_const, b_const = 2.0, -1.5
y = a_const * x1 + b_const * x2

X1_f = compute_cft(x1, t, f)
X2_f = compute_cft(x2, t, f)
y_f = compute_cft(y, t, f)
y_theory = a_const * X1_f + b_const * X2_f

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
mse_phase = np.mean((np.unwrap(np.angle(y_f)) - np.unwrap(np.angle(y_theory))) ** 2)

print(f"[01. Linearity] Magnitude MSE : {mse_mag:.6e}")
print(f"[01. Linearity] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, y, 'b'); axs[0, 0].set_title(r'$y(t) = a x_1(t) + b x_2(t)$')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f, np.unwrap(np.angle(y_f)), 'b', label='Numerical'); axs[1, 0].plot(f, np.unwrap(np.angle(y_theory)), 'r--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
