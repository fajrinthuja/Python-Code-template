"""
===============================================================================
2D PROPERTY 10: 2D SPATIAL INTEGRATION
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(t1, t2) = \int_{-\infty}^{t1} \int_{-\infty}^{t2} x(	au1, 	au2) d	au1 d	au2
    Frequency Domain: Y(u, v)   = X(u, v) / [(j * 2 * pi * u) * (j * 2 * pi * v)]  (for zero DC)
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
dt1, dt2 = t1[1] - t1[0], t2[1] - t2[0]

u = np.linspace(-2, 2, 45)
v = np.linspace(-2, 2, 45)
U, V = np.meshgrid(u, v)

# Zero-mean 2D derivative signal
x = (-2.0 * T1 * np.exp(-T1**2)) * (-2.0 * T2 * np.exp(-T2**2))
# 2D Cumulative Integration
y = np.cumsum(np.cumsum(x, axis=0) * dt2, axis=1) * dt1

X_uv = compute_2d_cft(x, t1, t2, u, v)
Y_num = compute_2d_cft(y, t1, t2, u, v)

# Theoretical formula
with np.errstate(divide='ignore', invalid='ignore'):
    Y_theo = X_uv / ((1j * 2 * np.pi * U) * (1j * 2 * np.pi * V))
    Y_theo[(U == 0) | (V == 0)] = 0.0

mask = (np.abs(U) > 0.15) & (np.abs(V) > 0.15)
mse_mag = np.mean((np.abs(Y_num)[mask] - np.abs(Y_theo)[mask]) ** 2)

print(f"[10. 2D Integration] Magnitude MSE (u,v != 0) : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(y, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'2D Integral $\iint x dt_1 dt_2$')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
