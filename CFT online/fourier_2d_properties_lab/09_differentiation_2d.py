"""
===============================================================================
2D PROPERTY 09: 2D SPATIAL DIFFERENTIATION
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(t1, t2) = \partial x(t1, t2) / \partial t1
    Frequency Domain: Y(u, v)   = (j * 2 * pi * u) * X(u, v)
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

u = np.linspace(-2, 2, 45)
v = np.linspace(-2, 2, 45)
U, V = np.meshgrid(u, v)

x = np.exp(-(T1**2 + T2**2))
# Derivative along t1 (axis 1)
y = np.gradient(x, t1, axis=1)

X_uv = compute_2d_cft(x, t1, t2, u, v)
Y_num = compute_2d_cft(y, t1, t2, u, v)

# Theoretical: (j * 2 * pi * u) * X(u, v)
Y_theo = (1j * 2 * np.pi * U) * X_uv

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[09. 2D Differentiation] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(y, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Derivative $\partial x / \partial t_1$')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
