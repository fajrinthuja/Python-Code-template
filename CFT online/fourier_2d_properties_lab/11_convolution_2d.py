"""
===============================================================================
2D PROPERTY 11: 2D SPATIAL CONVOLUTION
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(x, y) = I1(x, y) ** I2(x, y)
    Frequency Domain: Y(u, v) = F1(u, v) * F2(u, v)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

class CFT2D:
    def __init__(self, I, x, y, u, v):
        self.I = I; self.x = x; self.y = y; self.u = u; self.v = v

    def compute_cft(self):
        cos_ux = np.cos(2 * np.pi * self.u[None, :] * self.x[:, None])
        sin_ux = np.sin(2 * np.pi * self.u[None, :] * self.x[:, None])
        trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        A = trap(self.I[:, :, None] * cos_ux[None, :, :], x=self.x, axis=1)
        B = trap(self.I[:, :, None] * sin_ux[None, :, :], x=self.x, axis=1)
        cos_vy = np.cos(2 * np.pi * self.v[:, None] * self.y[None, :])
        sin_vy = np.sin(2 * np.pi * self.v[:, None] * self.y[None, :])
        term1 = trap(A[:, None, :] * cos_vy.T[:, :, None], x=self.y, axis=0)
        term2 = trap(B[:, None, :] * sin_vy.T[:, :, None], x=self.y, axis=0)
        term3 = trap(A[:, None, :] * sin_vy.T[:, :, None], x=self.y, axis=0)
        term4 = trap(B[:, None, :] * cos_vy.T[:, :, None], x=self.y, axis=0)
        return (term1 - term2) + 1j * (-(term3 + term4))

x = np.linspace(-3, 3, 90)
y = np.linspace(-3, 3, 90)
X, Y = np.meshgrid(x, y)
dx, dy = x[1] - x[0], y[1] - y[0]

u = np.linspace(-2, 2, 60)
v = np.linspace(-2, 2, 60)

I1 = np.where((np.abs(X) <= 0.5) & (np.abs(Y) <= 0.5), 1.0, 0.0)
I2 = np.where((np.abs(X) <= 0.5) & (np.abs(Y) <= 0.5), 1.0, 0.0)

# 2D Convolution in spatial domain
I_conv = fftconvolve(I1, I2, mode='same') * dx * dy

F1_uv = CFT2D(I1, x, y, u, v).compute_cft()
F2_uv = CFT2D(I2, x, y, u, v).compute_cft()
Y_num = CFT2D(I_conv, x, y, u, v).compute_cft()

# Theoretical: F1(u, v) * F2(u, v)
Y_theo = F1_uv * F2_uv

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[11. 2D Convolution] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(I_conv, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Spatial Convolution $I_1 ** I_2$')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
