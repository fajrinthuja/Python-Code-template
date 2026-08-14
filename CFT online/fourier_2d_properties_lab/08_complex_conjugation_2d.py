"""
===============================================================================
2D PROPERTY 08: 2D COMPLEX CONJUGATION
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(x, y) = I*(x, y)
    Frequency Domain: Y(u, v) = F*(-u, -v)
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

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
u = np.linspace(-3, 3, 70)
v = np.linspace(-3, 3, 70)

I = np.exp(-(X**2 + Y**2)) * np.exp(1j * 2 * np.pi * (1.0 * X + 0.5 * Y))
I_conj = np.conj(I)

F_uv = CFT2D(I, x, y, u, v).compute_cft()
Y_num = CFT2D(I_conj, x, y, u, v).compute_cft()

# Theoretical: conj(F(-u, -v))
Y_theo = np.conj(interp2d_complex(F_uv, u, v, -u, -v))

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[08. 2D Conjugation] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(I_conj.imag, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Im{$I^*(x, y)$}')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
