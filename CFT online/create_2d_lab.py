import os
import zipfile

# Absolute target directory
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
    Spatial Domain:   y(t1, t2) = a * x1(t1, t2) + b * x2(t1, t2)
    Frequency Domain: Y(u, v)   = a * X1(u, v) + b * X2(u, v)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_2d_cft(signal_2d, t1, t2, u, v):
    """Computes 2D continuous Fourier transform using separable trapezoidal integration."""
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

# Spatial & Frequency Grids
t1 = np.linspace(-3, 3, 80)
t2 = np.linspace(-3, 3, 80)
T1, T2 = np.meshgrid(t1, t2)

u = np.linspace(-2, 2, 40)
v = np.linspace(-2, 2, 40)

# Two signals: 2D Gaussian and 2D Rect
x1 = np.exp(-(T1**2 + T2**2))
x2 = np.where((np.abs(T1) <= 1.0) & (np.abs(T2) <= 1.0), 1.0, 0.0)

a_const, b_const = 2.0, -1.5
y = a_const * x1 + b_const * x2

X1_uv = compute_2d_cft(x1, t1, t2, u, v)
X2_uv = compute_2d_cft(x2, t1, t2, u, v)
Y_num = compute_2d_cft(y, t1, t2, u, v)

# Theoretical spectrum
Y_theo = a_const * X1_uv + b_const * X2_uv

# Error Verification
mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
mse_phase = np.mean((np.angle(Y_num) - np.angle(Y_theo)) ** 2)

print(f"[01. 2D Linearity] Magnitude MSE : {mse_mag:.6e}")
print(f"[01. 2D Linearity] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(y, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'$y(t_1, t_2) = a x_1 + b x_2$')
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
    Spatial Domain:   y(t1, t2) = x(t1 - t1_0, t2 - t2_0)
    Frequency Domain: Y(u, v)   = X(u, v) * exp(-j * 2 * pi * (u * t1_0 + v * t2_0))
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

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

t1 = np.linspace(-4, 4, 80)
t2 = np.linspace(-4, 4, 80)
T1, T2 = np.meshgrid(t1, t2)

u = np.linspace(-2, 2, 40)
v = np.linspace(-2, 2, 40)
U, V = np.meshgrid(u, v)

t1_0, t2_0 = 1.0, -1.0
x = np.exp(-(T1**2 + T2**2))
y = np.exp(-((T1 - t1_0)**2 + (T2 - t2_0)**2))

X_uv = compute_2d_cft(x, t1, t2, u, v)
Y_num = compute_2d_cft(y, t1, t2, u, v)

# Theoretical Shift Formula
Y_theo = X_uv * np.exp(-1j * 2 * np.pi * (U * t1_0 + V * t2_0))

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
sig_mask = np.abs(X_uv) > 1e-2
mse_phase = np.mean((np.angle(Y_num)[sig_mask] - np.angle(Y_theo)[sig_mask]) ** 2)

print(f"[02. 2D Time Shift] Magnitude MSE : {mse_mag:.6e}")
print(f"[02. 2D Time Shift] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(y, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Shifted $y(t_1, t_2)$')
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
    Spatial Domain:   y(t1, t2) = x(t1, t2) * exp(j * 2 * pi * (u0 * t1 + v0 * t2))
    Frequency Domain: Y(u, v)   = X(u - u0, v - v0)
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

t1 = np.linspace(-3, 3, 80)
t2 = np.linspace(-3, 3, 80)
T1, T2 = np.meshgrid(t1, t2)

u = np.linspace(-4, 4, 60)
v = np.linspace(-4, 4, 60)

u0, v0 = 1.0, 1.0
x = np.exp(-(T1**2 + T2**2))
y = x * np.exp(1j * 2 * np.pi * (u0 * T1 + v0 * T2))

X_uv = compute_2d_cft(x, t1, t2, u, v)
Y_num = compute_2d_cft(y, t1, t2, u, v)

# Theoretical 2D shift: evaluate X at (u - u0, v - v0)
Y_theo = interp2d_complex(X_uv, u, v, u - u0, v - v0)

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[03. 2D Frequency Shift] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(y.real, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Re{$y(t_1, t_2)$}')
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
    Spatial Domain:   y(t1, t2) = x(a1 * t1, a2 * t2)
    Frequency Domain: Y(u, v)   = (1 / |a1 * a2|) * X(u / a1, v / a2)
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

t1 = np.linspace(-3, 3, 80)
t2 = np.linspace(-3, 3, 80)
T1, T2 = np.meshgrid(t1, t2)

u = np.linspace(-3, 3, 50)
v = np.linspace(-3, 3, 50)

a1, a2 = 2.0, 1.5
x = np.exp(-(T1**2 + T2**2))
y = np.exp(-((a1 * T1)**2 + (a2 * T2)**2))

X_uv = compute_2d_cft(x, t1, t2, u, v)
Y_num = compute_2d_cft(y, t1, t2, u, v)

# Theoretical: (1 / |a1*a2|) * X(u/a1, v/a2)
Y_theo = (1.0 / np.abs(a1 * a2)) * interp2d_complex(X_uv, u, v, u / a1, v / a2)

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[04. 2D Time Scaling] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(y, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Compressed $y(t_1, t_2) = x(a_1 t_1, a_2 t_2)$')
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
    Spatial Domain:   y(t1, t2) = x(a1*t1, a2*t2) * exp(j * 2 * pi * (u0*t1 + v0*t2))
    Frequency Domain: Y(u, v)   = (1 / |a1 * a2|) * X((u - u0)/a1, (v - v0)/a2)
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

t1 = np.linspace(-3, 3, 80)
t2 = np.linspace(-3, 3, 80)
T1, T2 = np.meshgrid(t1, t2)

u = np.linspace(-4, 4, 50)
v = np.linspace(-4, 4, 50)

a1, a2 = 2.0, 2.0
u0, v0 = 1.0, 1.0

x = np.exp(-(T1**2 + T2**2))
y = np.exp(-((a1 * T1)**2 + (a2 * T2)**2)) * np.exp(1j * 2 * np.pi * (u0 * T1 + v0 * T2))

X_uv = compute_2d_cft(x, t1, t2, u, v)
Y_num = compute_2d_cft(y, t1, t2, u, v)

# Theoretical formula
Y_theo = (1.0 / np.abs(a1 * a2)) * interp2d_complex(X_uv, u, v, (u - u0)/a1, (v - v0)/a2)

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[05. 2D Scaling + Freq Shift] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(y.real, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Re{$y(t_1, t_2)$}')
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
''',

    # -------------------------------------------------------------------------
    # 07. 2D TIME REVERSAL
    # -------------------------------------------------------------------------
    "07_time_reversal_2d.py": '''"""
===============================================================================
2D PROPERTY 07: 2D SPATIAL REVERSAL
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(t1, t2) = x(-t1, -t2)
    Frequency Domain: Y(u, v)   = X(-u, -v)
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

t1 = np.linspace(-3, 3, 80)
t2 = np.linspace(-3, 3, 80)
T1, T2 = np.meshgrid(t1, t2)

u = np.linspace(-3, 3, 50)
v = np.linspace(-3, 3, 50)

# Asymmetric 2D signal
x = np.where((T1 >= 0) & (T2 >= 0), np.exp(-2.0 * T1 - 1.5 * T2), 0.0)
y = np.where((-T1 >= 0) & (-T2 >= 0), np.exp(-2.0 * (-T1) - 1.5 * (-T2)), 0.0)

X_uv = compute_2d_cft(x, t1, t2, u, v)
Y_num = compute_2d_cft(y, t1, t2, u, v)

# Theoretical: X(-u, -v)
Y_theo = interp2d_complex(X_uv, u, v, -u, -v)

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[07. 2D Time Reversal] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(y, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Reversed $y(t_1, t_2) = x(-t_1, -t_2)$')
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
    Spatial Domain:   y(t1, t2) = x*(t1, t2)
    Frequency Domain: Y(u, v)   = X*(-u, -v)
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

t1 = np.linspace(-3, 3, 80)
t2 = np.linspace(-3, 3, 80)
T1, T2 = np.meshgrid(t1, t2)

u = np.linspace(-3, 3, 50)
v = np.linspace(-3, 3, 50)

# Complex 2D signal
x = np.exp(-(T1**2 + T2**2)) * np.exp(1j * 2 * np.pi * (1.0 * T1 + 0.5 * T2))
y = np.conj(x)

X_uv = compute_2d_cft(x, t1, t2, u, v)
Y_num = compute_2d_cft(y, t1, t2, u, v)

# Theoretical: conj(X(-u, -v))
Y_theo = np.conj(interp2d_complex(X_uv, u, v, -u, -v))

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[08. 2D Conjugation] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(y.imag, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Im{$x^*(t_1, t_2)$}')
axs[1].imshow(np.abs(Y_num), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='plasma')
axs[1].set_title(r'Numerical $|Y(u, v)|$')
axs[2].imshow(np.abs(np.abs(Y_num) - np.abs(Y_theo)), extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[2].set_title('Absolute Magnitude Error')
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 09. 2D DIFFERENTIATION (GRADIENT / LAPLACIAN)
    # -------------------------------------------------------------------------
    "09_differentiation_2d.py": '''"""
===============================================================================
2D PROPERTY 09: 2D SPATIAL DIFFERENTIATION
===============================================================================
Mathematical Formula:
    Spatial Domain:   y(t1, t2) = \partial x(t1, t2) / \partial t1
    Frequency Domain: Y(u, v)   = (j * 2 * pi * u) * X(u, v)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

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

t1 = np.linspace(-3, 3, 90)
t2 = np.linspace(-3, 3, 90)
T1, T2 = np.meshgrid(t1, t2)

u = np.linspace(-2, 2, 45)
v = np.linspace(-2, 2, 45)
U, V = np.meshgrid(u, v)

x = np.exp(-(T1**2 + T2**2))
# Derivative along t1 (axis 1)
y = np.gradient(x, t1, axis=1)

X_uv = compute_2d_cft(x, t1, t2, u, v)
Y_num = compute_2d_cft(y, t1, t2, u, v)

# Theoretical: (j * 2 * pi * u) * X(u, v)
Y_theo = (1j * 2 * np.pi * U) * X_uv

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[09. 2D Differentiation] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(y, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Derivative $\partial x / \partial t_1$')
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
    Spatial Domain:   y(t1, t2) = \int_{-\infty}^{t1} \int_{-\infty}^{t2} x(\tau1, \tau2) d\tau1 d\tau2
    Frequency Domain: Y(u, v)   = X(u, v) / [(j * 2 * pi * u) * (j * 2 * pi * v)]  (for zero DC)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

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

t1 = np.linspace(-3, 3, 90)
t2 = np.linspace(-3, 3, 90)
T1, T2 = np.meshgrid(t1, t2)
dt1, dt2 = t1[1] - t1[0], t2[1] - t2[0]

u = np.linspace(-2, 2, 45)
v = np.linspace(-2, 2, 45)
U, V = np.meshgrid(u, v)

# Zero-mean 2D derivative signal
x = (-2.0 * T1 * np.exp(-T1**2)) * (-2.0 * T2 * np.exp(-T2**2))
# 2D Cumulative Integration
y = np.cumsum(np.cumsum(x, axis=0) * dt2, axis=1) * dt1

X_uv = compute_2d_cft(x, t1, t2, u, v)
Y_num = compute_2d_cft(y, t1, t2, u, v)

# Theoretical formula
with np.errstate(divide='ignore', invalid='ignore'):
    Y_theo = X_uv / ((1j * 2 * np.pi * U) * (1j * 2 * np.pi * V))
    Y_theo[(U == 0) | (V == 0)] = 0.0

mask = (np.abs(U) > 0.15) & (np.abs(V) > 0.15)
mse_mag = np.mean((np.abs(Y_num)[mask] - np.abs(Y_theo)[mask]) ** 2)

print(f"[10. 2D Integration] Magnitude MSE (u,v != 0) : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(y, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'2D Integral $\iint x dt_1 dt_2$')
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
    Spatial Domain:   y(t1, t2) = x1(t1, t2) ** x2(t1, t2)
    Frequency Domain: Y(u, v)   = X1(u, v) * X2(u, v)
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
dt1, dt2 = t1[1] - t1[0], t2[1] - t2[0]

u = np.linspace(-2, 2, 35)
v = np.linspace(-2, 2, 35)

x1 = np.where((np.abs(T1) <= 0.5) & (np.abs(T2) <= 0.5), 1.0, 0.0)
x2 = np.where((np.abs(T1) <= 0.5) & (np.abs(T2) <= 0.5), 1.0, 0.0)

# 2D Convolution in space
y = fftconvolve(x1, x2, mode='same') * dt1 * dt2

X1_uv = compute_2d_cft(x1, t1, t2, u, v)
X2_uv = compute_2d_cft(x2, t1, t2, u, v)
Y_num = compute_2d_cft(y, t1, t2, u, v)

# Theoretical: X1(u, v) * X2(u, v)
Y_theo = X1_uv * X2_uv

mse_mag = np.mean((np.abs(Y_num) - np.abs(Y_theo)) ** 2)
print(f"[11. 2D Convolution] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(y, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'2D Spatial Convolution $x_1 ** x_2$')
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
''',

    # -------------------------------------------------------------------------
    # 13. 2D PARSEVAL'S THEOREM (ENERGY CONSERVATION)
    # -------------------------------------------------------------------------
    "13_parsevals_theorem_2d.py": '''"""
===============================================================================
2D PROPERTY 13: 2D PARSEVAL'S / PLANCHEREL'S THEOREM
===============================================================================
Mathematical Formula:
    \iint |x(t1, t2)|^2 dt1 dt2 = \iint |X(u, v)|^2 du dv
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

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

t1 = np.linspace(-3, 3, 90)
t2 = np.linspace(-3, 3, 90)
T1, T2 = np.meshgrid(t1, t2)

u = np.linspace(-3, 3, 60)
v = np.linspace(-3, 3, 60)

x = np.exp(-2.0 * (T1**2 + T2**2))
X_uv = compute_2d_cft(x, t1, t2, u, v)

# Energy in Spatial Domain
if hasattr(np, 'trapezoid'):
    energy_space = np.trapezoid(np.trapezoid(np.abs(x)**2, t1, axis=1), t2, axis=0)
    energy_freq = np.trapezoid(np.trapezoid(np.abs(X_uv)**2, u, axis=1), v, axis=0)
else:
    energy_space = np.trapz(np.trapz(np.abs(x)**2, t1, axis=1), t2, axis=0)
    energy_freq = np.trapz(np.trapz(np.abs(X_uv)**2, u, axis=1), v, axis=0)

diff = np.abs(energy_space - energy_freq)

print(f"[13. 2D Parseval] Energy Space Domain : {energy_space:.6f} J")
print(f"[13. 2D Parseval] Energy Freq Domain  : {energy_freq:.6f} J")
print(f"[13. 2D Parseval] Absolute Difference : {diff:.6e} J")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(np.abs(x)**2, extent=[t1[0], t1[-1], t2[0], t2[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Spatial Energy Density $|x(t_1, t_2)|^2$')
axs[1].imshow(np.abs(X_uv)**2, extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[1].set_title(r'Spectral Energy Density $|X(u, v)|^2$')
plt.tight_layout(); plt.show()
'''
}

# Write all files to the target directory
for fname, content in files_2d.items():
    fpath = os.path.join(TARGET_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

# Create a zip archive containing all 13 files
zip_path = os.path.join(TARGET_DIR, "fourier_2d_properties_lab.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fname in files_2d:
        z.write(os.path.join(TARGET_DIR, fname), arcname=fname)

print("=" * 75)
print("SUCCESS: All 13 2D Fourier property files written to:")
print(f"--> {TARGET_DIR}")
print(f"--> Zip file: {zip_path}")
print("=" * 75)

# Open File Explorer to the created folder
try:
    os.startfile(TARGET_DIR)
except Exception:
    pass