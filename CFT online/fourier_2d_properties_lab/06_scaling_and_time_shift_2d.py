"""
===============================================================================
2D PROPERTY 06: COMBINED 2D SCALING AND TIME SHIFT
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(t1, t2) = x(a1 * (t1 - t1_0), a2 * (t2 - t2_0))
    Frequency Domain: Y(u, v)   = (1 / |a1 * a2|) * X(u / a1, v / a2) * exp(-j * 2 * pi * (u * t1_0 + v * t2_0))
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

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

def interp2d_complex(X_uv, u_grid, v_grid, u_target, v_target):
    interp_r = RegularGridInterpolator((v_grid, u_grid), X_uv.real, bounds_error=False, fill_value=0.0)
    interp_i = RegularGridInterpolator((v_grid, u_grid), X_uv.imag, bounds_error=False, fill_value=0.0)
    U_tar, V_tar = np.meshgrid(u_target, v_target)
    pts = np.stack([V_tar.ravel(), U_tar.ravel()], axis=-1)
    return (interp_r(pts) + 1j * interp_i(pts)).reshape(len(v_target), len(u_target))

t1 = np.linspace(-4, 4, 80)
t2 = np.linspace(-4, 4, 80)
T1, T2 = np.meshgrid(t1, t2)

u = np.linspace(-3, 3, 50)
v = np.linspace(-3, 3, 50)
U, V = np.meshgrid(u, v)

a1, a2 = 2.0, 2.0
t1_0, t2_0 = 1.0, -0.5

x = np.exp(-(T1**2 + T2**2))
y = np.exp(-((a1 * (T1 - t1_0))**2 + (a2 * (T2 - t2_0))**2))

X_uv = compute_2d_cft(x, t1, t2, u, v)
Y_num = compute_2d_cft(y, t1, t2, u, v)

# Theoretical formula
X_scaled = (1.0 / np.abs(a1 * a2)) * interp2d_complex(X_uv, u, v, u / a1, v / a2)
Y_theo = X_scaled * np.exp(-1j * 2 * np.pi * (U * t1_0 + V * t2_0))

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[06. 2D Scaling + Time Shift] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(y, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'$y(t_1, t_2) = x(a_1(t_1-t_{10}), a_2(t_2-t_{20}))$')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
