import os
import zipfile

TARGET_DIR = r"D:\2-2\Python Code Template\Python-Code-template\CFT online\fourier_2d_properties_lab"
os.makedirs(TARGET_DIR, exist_ok=True)

files_2d = {
    # -------------------------------------------------------------------------
    # 01. 2D LINEARITY
    # -------------------------------------------------------------------------
    "01_linearity_2d.py": '''"""
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
''',

    # -------------------------------------------------------------------------
    # 02. 2D TIME (SPATIAL) SHIFTING
    # -------------------------------------------------------------------------
    "02_time_shifting_2d.py": '''"""
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
''',

    # -------------------------------------------------------------------------
    # 03. 2D FREQUENCY SHIFTING (MODULATION)
    # -------------------------------------------------------------------------
    "03_frequency_shifting_2d.py": '''"""
===============================================================================
2D PROPERTY 03: FREQUENCY SHIFTING (MODULATION)
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(x, y) = I(x, y) * exp(j * 2 * pi * (u0 * x + v0 * y))
    Frequency Domain: Y(u, v) = F(u - u0, v - v0)
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
u = np.linspace(-4, 4, 80)
v = np.linspace(-4, 4, 80)

u0, v0 = 1.0, 1.0
I = np.exp(-(X**2 + Y**2))
I_mod = I * np.exp(1j * 2 * np.pi * (u0 * X + v0 * Y))

F_uv = CFT2D(I, x, y, u, v).compute_cft()
Y_num = CFT2D(I_mod, x, y, u, v).compute_cft()

# Theoretical: evaluate F at (u - u0, v - v0)
Y_theo = interp2d_complex(F_uv, u, v, u - u0, v - v0)

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[03. 2D Frequency Shift] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(I_mod.real, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Re{$I(x, y)e^{j2\pi(u_0 x + v_0 y)}$}')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 04. 2D TIME (SPATIAL) SCALING
    # -------------------------------------------------------------------------
    "04_time_scaling_2d.py": '''"""
===============================================================================
2D PROPERTY 04: SPATIAL SCALING
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(x, y) = I(a1 * x, a2 * y)
    Frequency Domain: Y(u, v) = (1 / |a1 * a2|) * F(u / a1, v / a2)
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

a1, a2 = 2.0, 1.5
I = np.exp(-(X**2 + Y**2))
I_scaled = np.exp(-((a1 * X)**2 + (a2 * Y)**2))

F_uv = CFT2D(I, x, y, u, v).compute_cft()
Y_num = CFT2D(I_scaled, x, y, u, v).compute_cft()

# Theoretical: (1/|a1*a2|) * F(u/a1, v/a2)
Y_theo = (1.0 / np.abs(a1 * a2)) * interp2d_complex(F_uv, u, v, u / a1, v / a2)

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[04. 2D Time Scaling] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(I_scaled, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Compressed $I(a_1 x, a_2 y)$')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 05. 2D SCALING AND FREQUENCY SHIFT
    # -------------------------------------------------------------------------
    "05_scaling_and_freq_shift_2d.py": '''"""
===============================================================================
2D PROPERTY 05: COMBINED 2D SCALING AND FREQUENCY SHIFT
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(x, y) = I(a1*x, a2*y) * exp(j * 2 * pi * (u0*x + v0*y))
    Frequency Domain: Y(u, v) = (1 / |a1 * a2|) * F((u - u0)/a1, (v - v0)/a2)
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
u = np.linspace(-4, 4, 70)
v = np.linspace(-4, 4, 70)

a1, a2 = 2.0, 2.0
u0, v0 = 1.0, 1.0

I = np.exp(-(X**2 + Y**2))
I_trans = np.exp(-((a1 * X)**2 + (a2 * Y)**2)) * np.exp(1j * 2 * np.pi * (u0 * X + v0 * Y))

F_uv = CFT2D(I, x, y, u, v).compute_cft()
Y_num = CFT2D(I_trans, x, y, u, v).compute_cft()

# Theoretical formula
Y_theo = (1.0 / np.abs(a1 * a2)) * interp2d_complex(F_uv, u, v, (u - u0)/a1, (v - v0)/a2)

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[05. 2D Scaling + Freq Shift] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(I_trans.real, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Re{$I(a_1 x, a_2 y)e^{j2\pi(u_0 x + v_0 y)}$}')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 06. 2D SCALING AND TIME SHIFT
    # -------------------------------------------------------------------------
    "06_scaling_and_time_shift_2d.py": '''"""
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
''',

    # -------------------------------------------------------------------------
    # 07. 2D TIME REVERSAL
    # -------------------------------------------------------------------------
    "07_time_reversal_2d.py": '''"""
===============================================================================
2D PROPERTY 07: 2D SPATIAL REVERSAL
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(x, y) = I(-x, -y)
    Frequency Domain: Y(u, v) = F(-u, -v)
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

# Asymmetric 2D signal
I = np.where((X >= 0) & (Y >= 0), np.exp(-2.0 * X - 1.5 * Y), 0.0)
I_rev = np.where((-X >= 0) & (-Y >= 0), np.exp(-2.0 * (-X) - 1.5 * (-Y)), 0.0)

F_uv = CFT2D(I, x, y, u, v).compute_cft()
Y_num = CFT2D(I_rev, x, y, u, v).compute_cft()

# Theoretical: F(-u, -v)
Y_theo = interp2d_complex(F_uv, u, v, -u, -v)

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[07. 2D Time Reversal] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(I_rev, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Reversed $I(-x, -y)$')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 08. 2D COMPLEX CONJUGATION
    # -------------------------------------------------------------------------
    "08_complex_conjugation_2d.py": '''"""
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
''',

    # -------------------------------------------------------------------------
    # 09. 2D DIFFERENTIATION
    # -------------------------------------------------------------------------
    "09_differentiation_2d.py": '''"""
===============================================================================
2D PROPERTY 09: 2D SPATIAL DIFFERENTIATION
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(x, y) = \partial I(x, y) / \partial x
    Frequency Domain: Y(u, v) = (j * 2 * pi * u) * F(u, v)
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
u = np.linspace(-2, 2, 70)
v = np.linspace(-2, 2, 70)
U, V = np.meshgrid(u, v)

I = np.exp(-(X**2 + Y**2))
# Gradient along x (axis 1)
dI_dx = np.gradient(I, x, axis=1)

F_uv = CFT2D(I, x, y, u, v).compute_cft()
Y_num = CFT2D(dI_dx, x, y, u, v).compute_cft()

# Theoretical: (j * 2 * pi * u) * F(u, v)
Y_theo = (1j * 2 * np.pi * U) * F_uv

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[09. 2D Differentiation] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(dI_dx, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Derivative $\partial I / \partial x$')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 10. 2D INTEGRATION
    # -------------------------------------------------------------------------
    "10_integration_2d.py": '''"""
===============================================================================
2D PROPERTY 10: 2D SPATIAL INTEGRATION
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(x, y) = \iint I(\tau1, \tau2) d\tau1 d\tau2
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
''',

    # -------------------------------------------------------------------------
    # 11. 2D CONVOLUTION IN SPACE
    # -------------------------------------------------------------------------
    "11_convolution_2d.py": '''"""
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
''',

    # -------------------------------------------------------------------------
    # 12. 2D MULTIPLICATION IN SPACE (2D FREQUENCY CONVOLUTION)
    # -------------------------------------------------------------------------
    "12_multiplication_2d.py": '''"""
===============================================================================
2D PROPERTY 12: 2D MULTIPLICATION IN SPACE (FREQUENCY CONVOLUTION)
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(x, y) = I1(x, y) * I2(x, y)
    Frequency Domain: Y(u, v) = F1(u, v) ** F2(u, v)
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

u = np.linspace(-3, 3, 60)
v = np.linspace(-3, 3, 60)
du, dv = u[1] - u[0], v[1] - v[0]

I1 = np.exp(-(X**2 + Y**2))
I2 = np.cos(2 * np.pi * (1.0 * X + 1.0 * Y))
I_prod = I1 * I2

F1_uv = CFT2D(I1, x, y, u, v).compute_cft()
F2_uv = CFT2D(I2, x, y, u, v).compute_cft()
Y_num = CFT2D(I_prod, x, y, u, v).compute_cft()

# Theoretical: 2D Frequency Convolution
Y_theo = fftconvolve(F1_uv, F2_uv, mode='same') * du * dv

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[12. 2D Multiplication] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(I_prod, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Product $I_1(x, y) \cdot I_2(x, y)$')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 13. 2D PARSEVAL'S THEOREM (ENERGY CONSERVATION)
    # -------------------------------------------------------------------------
    "13_parsevals_theorem_2d.py": '''"""
===============================================================================
2D PROPERTY 13: 2D PARSEVAL'S / PLANCHEREL'S THEOREM
===============================================================================
Mathematical Formula:
    \iint |I(x, y)|^2 dx dy = \iint |F(u, v)|^2 du dv
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

u = np.linspace(-3, 3, 80)
v = np.linspace(-3, 3, 80)

I = np.exp(-2.0 * (X**2 + Y**2))
F_uv = CFT2D(I, x, y, u, v).compute_cft()

trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# Energy in Spatial Domain
energy_space = trap(trap(np.abs(I)**2, x, axis=1), y, axis=0)
# Energy in Frequency Domain
energy_freq = trap(trap(np.abs(F_uv)**2, u, axis=1), v, axis=0)

diff = np.abs(energy_space - energy_freq)

print(f"[13. 2D Parseval] Energy Space Domain : {energy_space:.6f} J")
print(f"[13. 2D Parseval] Energy Freq Domain  : {energy_freq:.6f} J")
print(f"[13. 2D Parseval] Absolute Difference : {diff:.6e} J")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(np.abs(I)**2, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Spatial Energy Density $|I(x, y)|^2$')
axs[1].imshow(np.abs(F_uv)**2, extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[1].set_title(r'Spectral Energy Density $|F(u, v)|^2$')
plt.tight_layout(); plt.show()
'''
}

# Write files to target folder
for fname, content in files_2d.items():
    fpath = os.path.join(TARGET_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

# Create zip archive
zip_path = os.path.join(TARGET_DIR, "fourier_2d_properties_lab.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fname in files_2d:
        z.write(os.path.join(TARGET_DIR, fname), arcname=fname)

print("=" * 75)
print("SUCCESS: All 13 Fast Separable 2D Fourier files written to:")
print(f"--> {TARGET_DIR}")
print(f"--> Zip file: {zip_path}")
print("=" * 75)

# Open File Explorer to folder
try:
    os.startfile(TARGET_DIR)
except Exception:
    pass