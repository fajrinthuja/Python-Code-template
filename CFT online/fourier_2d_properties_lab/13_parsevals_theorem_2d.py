"""
===============================================================================
2D PROPERTY 13: 2D PARSEVAL'S / PLANCHEREL'S THEOREM
===============================================================================
Mathematical Formula:
    \iint |x(t1, t2)|^2 dt1 dt2 = \iint |X(u, v)|^2 du dv
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_2d_cft(signal_2d, t1, t2, u, v):
    T1, T2 = np.meshgrid(t1, t2)
    X_uv = np.zeros((len(v), len(u)), dtype=complex)
    for i, v_val in enumerate(v):
        for j, u_val in enumerate(u):
            kernel = np.exp(-1j * 2 * np.pi * (u_val * T1 + v_val * T2))
            integrand = signal_2d * kernel
            if hasattr(np, 'trapezoid'):
                int_t1 = np.trapezoid(integrand, t1, axis=1)
                X_uv[i, j] = np.trapezoid(int_t1, t2, axis=0)
            else:
                int_t1 = np.trapz(integrand, t1, axis=1)
                X_uv[i, j] = np.trapz(int_t1, t2, axis=0)
    return X_uv

t1 = np.linspace(-3, 3, 90)
t2 = np.linspace(-3, 3, 90)
T1, T2 = np.meshgrid(t1, t2)

u = np.linspace(-3, 3, 60)
v = np.linspace(-3, 3, 60)

x = np.exp(-2.0 * (T1**2 + T2**2))
X_uv = compute_2d_cft(x, t1, t2, u, v)

# Energy in Spatial Domain
if hasattr(np, 'trapezoid'):
    energy_space = np.trapezoid(np.trapezoid(np.abs(x)**2, t1, axis=1), t2, axis=0)
    energy_freq = np.trapezoid(np.trapezoid(np.abs(X_uv)**2, u, axis=1), v, axis=0)
else:
    energy_space = np.trapz(np.trapz(np.abs(x)**2, t1, axis=1), t2, axis=0)
    energy_freq = np.trapz(np.trapz(np.abs(X_uv)**2, u, axis=1), v, axis=0)

diff = np.abs(energy_space - energy_freq)

print(f"[13. 2D Parseval] Energy Space Domain : {energy_space:.6f} J")
print(f"[13. 2D Parseval] Energy Freq Domain  : {energy_freq:.6f} J")
print(f"[13. 2D Parseval] Absolute Difference : {diff:.6e} J")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(np.abs(x)**2, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Spatial Energy Density $|x(t_1, t_2)|^2$')
axs[1].imshow(np.abs(X_uv)**2, extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[1].set_title(r'Spectral Energy Density $|X(u, v)|^2$')
plt.tight_layout(); plt.show()
