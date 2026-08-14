"""
===============================================================================
2D PROPERTY 01: LINEARITY / SUPERPOSITION
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(x, y) = a * I1(x, y) + b * I2(x, y)
    Frequency Domain: Y(u, v) = a * F1(u, v) + b * F2(u, v)
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

# 1. Grids
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
u = np.linspace(-2, 2, 80)
v = np.linspace(-2, 2, 80)

# 2. Signals
I1 = np.exp(-(X**2 + Y**2))
I2 = np.where((np.abs(X) <= 1.0) & (np.abs(Y) <= 1.0), 1.0, 0.0)
a_const, b_const = 2.0, -1.5
I_combo = a_const * I1 + b_const * I2

# 3. Fast Separable 2D CFT
F1 = CFT2D(I1, x, y, u, v).compute_cft()
F2 = CFT2D(I2, x, y, u, v).compute_cft()
Y_num = CFT2D(I_combo, x, y, u, v).compute_cft()
Y_theo = a_const * F1 + b_const * F2

# 4. Error Metrics
mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
mse_phase = np.mean((np.angle(Y_num) - np.angle(Y_theo)) ** 2)

print(f"[01. 2D Linearity] Magnitude MSE : {mse_mag:.6e}")
print(f"[01. 2D Linearity] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(I_combo, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'$y(x, y) = a I_1 + b I_2$')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
