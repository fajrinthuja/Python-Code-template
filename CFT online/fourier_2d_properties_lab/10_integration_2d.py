"""
===============================================================================
2D PROPERTY 10: 2D SPATIAL INTEGRATION
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(x, y) = \iint I(	au1, 	au2) d	au1 d	au2
    Frequency Domain: Y(u, v) = F(u, v) / [(j * 2 * pi * u) * (j * 2 * pi * v)] (u,v != 0)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

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

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
dx, dy = x[1] - x[0], y[1] - y[0]

u = np.linspace(-2, 2, 70)
v = np.linspace(-2, 2, 70)
U, V = np.meshgrid(u, v)

# Zero-mean 2D derivative signal
I = (-2.0 * X * np.exp(-X**2)) * (-2.0 * Y * np.exp(-Y**2))
I_int = np.cumsum(np.cumsum(I, axis=0) * dy, axis=1) * dx

F_uv = CFT2D(I, x, y, u, v).compute_cft()
Y_num = CFT2D(I_int, x, y, u, v).compute_cft()

# Theoretical formula
with np.errstate(divide='ignore', invalid='ignore'):
    Y_theo = F_uv / ((1j * 2 * np.pi * U) * (1j * 2 * np.pi * V))
    Y_theo[(U == 0) | (V == 0)] = 0.0

mask = (np.abs(U) > 0.15) & (np.abs(V) > 0.15)
mse_mag = np.mean((np.abs(Y_num)[mask] - np.abs(Y_theo)[mask]) ** 2)

print(f"[10. 2D Integration] Magnitude MSE (u,v != 0) : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(I_int, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'2D Integral $\iint I dx dy$')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
