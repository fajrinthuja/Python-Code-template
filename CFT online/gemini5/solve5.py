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


x = np.linspace(-3,3,80)
y = np.linspace(-3,3,80)

u = np.linspace(-3,3,80)
v = np.linspace(-3,3,80)

X, Y = np.meshgrid(x, y)
I = np.exp(-(2 * X**2 + 0.5 * Y**2))

a = 0.5
b = 2
# I_trans = np.exp(-(2 * (a * x) ** 2 + 0.5 * (b * y) ** 2))
I_trans = np.exp(-(2 * (a * X)**2 + 0.5 * (b * Y)**2))

ycal = CFT2D(I_trans, x,y,u,v)
xcal = CFT2D(I, x, y, u, v)
y_f = ycal.compute_cft()
x_f = xcal.compute_cft()

y_theo = interp2d_complex(x_f, u, v, u / a, v / b)
y_theo /= a * b

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theo)) ** 2)
print(f"[04. 2D Time Scaling] Magnitude MSE : {mse_mag:.6e}")

fig, axes = plt.subplots(2,1, figsize = (8,4))
axes[0].plot(y_f)
axes[1].plot(y_theo)
fig.tight_layout()
plt.show()