"""
===============================================================================
2D PROPERTY 06: COMBINED 2D SCALING AND TIME SHIFT
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(x, y) = I(a1 * (x - x0), a2 * (y - y0))
    Frequency Domain: Y(u, v) = (1 / |a1 * a2|) * F(u / a1, v / a2) * exp(-j * 2 * pi * (u * x0 + v * y0))
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

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

def interp2d_complex(F_uv, u_grid, v_grid, u_tar, v_tar):
    interp_r = RegularGridInterpolator((v_grid, u_grid), F_uv.real, bounds_error=False, fill_value=0.0)
    interp_i = RegularGridInterpolator((v_grid, u_grid), F_uv.imag, bounds_error=False, fill_value=0.0)
    U_tar, V_tar = np.meshgrid(u_tar, v_tar)
    pts = np.stack([V_tar.ravel(), U_tar.ravel()], axis=-1)
    return (interp_r(pts) + 1j * interp_i(pts)).reshape(len(v_tar), len(u_tar))

x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
u = np.linspace(-3, 3, 70)
v = np.linspace(-3, 3, 70)
U, V = np.meshgrid(u, v)

a1, a2 = 2.0, 2.0
x0, y0 = 1.0, -0.5

I = np.exp(-(X**2 + Y**2))
I_trans = np.exp(-((a1 * (X - x0))**2 + (a2 * (Y - y0))**2))

F_uv = CFT2D(I, x, y, u, v).compute_cft()
Y_num = CFT2D(I_trans, x, y, u, v).compute_cft()

# Theoretical formula
F_scaled = (1.0 / np.abs(a1 * a2)) * interp2d_complex(F_uv, u, v, u / a1, v / a2)
Y_theo = F_scaled * np.exp(-1j * 2 * np.pi * (U * x0 + V * y0))

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[06. 2D Scaling + Time Shift] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(I_trans, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'$I(a_1(x - x_0), a_2(y - y_0))$')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
