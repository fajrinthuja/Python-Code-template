"""
===============================================================================
2D PROPERTY 02: SPATIAL / TIME SHIFTING
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(t1, t2) = x(t1 - t1_0, t2 - t2_0)
    Frequency Domain: Y(u, v)   = X(u, v) * exp(-j * 2 * pi * (u * t1_0 + v * t2_0))
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

t1 = np.linspace(-4, 4, 80)
t2 = np.linspace(-4, 4, 80)
T1, T2 = np.meshgrid(t1, t2)

u = np.linspace(-2, 2, 40)
v = np.linspace(-2, 2, 40)
U, V = np.meshgrid(u, v)

t1_0, t2_0 = 1.0, -1.0
x = np.exp(-(T1**2 + T2**2))
y = np.exp(-((T1 - t1_0)**2 + (T2 - t2_0)**2))

X_uv = compute_2d_cft(x, t1, t2, u, v)
Y_num = compute_2d_cft(y, t1, t2, u, v)

# Theoretical Shift Formula
Y_theo = X_uv * np.exp(-1j * 2 * np.pi * (U * t1_0 + V * t2_0))

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
sig_mask = np.abs(X_uv) > 1e-2
mse_phase = np.mean((np.angle(Y_num)[sig_mask] - np.angle(Y_theo)[sig_mask]) ** 2)

print(f"[02. 2D Time Shift] Magnitude MSE : {mse_mag:.6e}")
print(f"[02. 2D Time Shift] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(y, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Shifted $y(t_1, t_2)$')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Magnitude $|Y(u, v)|$ (Invariant)')
axs[2].imshow(np.abs(np.angle(Y_num) - np.angle(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Phase Error')
plt.tight_layout(); plt.show()
