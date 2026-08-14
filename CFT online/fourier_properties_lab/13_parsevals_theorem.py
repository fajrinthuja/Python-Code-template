"""
===============================================================================
PROPERTY 13: PARSEVAL'S THEOREM (ENERGY CONSERVATION)
Formula: int |x(t)|^2 dt == int |X(f)|^2 df
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

t = np.linspace(-5, 5, 2500)
f = np.linspace(-15, 15, 1500)

x = np.exp(-2.0 * t**2)
x_f = compute_cft(x, t, f)

energy_time = np.trapezoid(np.abs(x)**2, t) if hasattr(np, 'trapezoid') else np.trapz(np.abs(x)**2, t)
energy_freq = np.trapezoid(np.abs(x_f)**2, f) if hasattr(np, 'trapezoid') else np.trapz(np.abs(x_f)**2, f)
diff = np.abs(energy_time - energy_freq)

print(f"[13. Parseval] Energy Time Domain : {energy_time:.6f} J")
print(f"[13. Parseval] Energy Freq Domain : {energy_freq:.6f} J")
print(f"[13. Parseval] Difference         : {diff:.6e} J")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(t, np.abs(x)**2, 'b'); axs[0].set_title(r'Instantaneous Energy $|x(t)|^2$')
axs[1].plot(f, np.abs(x_f)**2, 'r'); axs[1].set_title(r'Energy Spectral Density $|X(f)|^2$')
for ax in axs: ax.grid(True)
plt.tight_layout(); plt.show()
