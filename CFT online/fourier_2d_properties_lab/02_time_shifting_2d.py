"""
===============================================================================
2D PROPERTY 02: SPATIAL / TIME SHIFTING
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(x, y) = I(x - x0, y - y0)
    Frequency Domain: Y(u, v) = F(u, v) * exp(-j * 2 * pi * (u * x0 + v * y0))
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

x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
u = np.linspace(-2, 2, 80)
v = np.linspace(-2, 2, 80)
U, V = np.meshgrid(u, v)

x0, y0 = 1.0, -0.8
I_base = np.exp(-(X**2 + Y**2))
I_shift = np.exp(-((X - x0)**2 + (Y - y0)**2))

F_base = CFT2D(I_base, x, y, u, v).compute_cft()
Y_num = CFT2D(I_shift, x, y, u, v).compute_cft()

# Theoretical Shift
Y_theo = F_base * np.exp(-1j * 2 * np.pi * (U * x0 + V * y0))

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
sig_mask = np.abs(F_base) > 1e-2
mse_phase = np.mean((np.angle(Y_num)[sig_mask] - np.angle(Y_theo)[sig_mask]) ** 2)

print(f"[02. 2D Time Shift] Magnitude MSE : {mse_mag:.6e}")
print(f"[02. 2D Time Shift] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(I_shift, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Shifted $I(x - x_0, y - y_0)$')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Magnitude $|Y(u, v)|$ (Invariant)')
axs[2].imshow(np.abs(np.angle(Y_num) - np.angle(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Phase Error')
plt.tight_layout(); plt.show()
