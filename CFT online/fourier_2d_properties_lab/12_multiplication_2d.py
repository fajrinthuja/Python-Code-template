"""
===============================================================================
2D PROPERTY 12: 2D MULTIPLICATION IN SPACE (FREQUENCY CONVOLUTION)
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(t1, t2) = x1(t1, t2) * x2(t1, t2)
    Frequency Domain: Y(u, v)   = X1(u, v) ** X2(u, v)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

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

t1 = np.linspace(-3, 3, 70)
t2 = np.linspace(-3, 3, 70)
T1, T2 = np.meshgrid(t1, t2)

u = np.linspace(-3, 3, 40)
v = np.linspace(-3, 3, 40)
du, dv = u[1] - u[0], v[1] - v[0]

x1 = np.exp(-(T1**2 + T2**2))
x2 = np.cos(2 * np.pi * (1.0 * T1 + 1.0 * T2))
y = x1 * x2

X1_uv = compute_2d_cft(x1, t1, t2, u, v)
X2_uv = compute_2d_cft(x2, t1, t2, u, v)
Y_num = compute_2d_cft(y, t1, t2, u, v)

# Theoretical: 2D Frequency Convolution
Y_theo = fftconvolve(X1_uv, X2_uv, mode='same') * du * dv

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[12. 2D Multiplication] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(y, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Product $y(t_1, t_2) = x_1 x_2$')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
