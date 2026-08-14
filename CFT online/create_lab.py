import os
import subprocess
import zipfile

# Hardcoded absolute destination directory
TARGET_DIR = r"D:\2-2\Python Code Template\Python-Code-template\CFT online\fourier_properties_lab"

os.makedirs(TARGET_DIR, exist_ok=True)

files = {
    # -------------------------------------------------------------------------
    # 01. LINEARITY
    # -------------------------------------------------------------------------
    "01_linearity.py": '''"""
===============================================================================
PROPERTY 01: LINEARITY / SUPERPOSITION
Formula: CFT{ a*x1(t) + b*x2(t) } = a*X1(f) + b*X2(f)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_cft(signal, t, f):
    cft = np.zeros(len(f), dtype=complex)
    for i, fi in enumerate(f):
        kernel = np.exp(-1j * 2 * np.pi * fi * t)
        cft[i] = np.trapezoid(signal * kernel, t) if hasattr(np, 'trapezoid') else np.trapz(signal * kernel, t)
    return cft

t = np.linspace(-5, 5, 2000)
f = np.linspace(-10, 10, 1000)

x1 = np.exp(-1.0 * t**2)
x2 = np.where(np.abs(t) <= 1.0, 1.0, 0.0)
a_const, b_const = 2.0, -1.5
y = a_const * x1 + b_const * x2

X1_f = compute_cft(x1, t, f)
X2_f = compute_cft(x2, t, f)
y_f = compute_cft(y, t, f)
y_theory = a_const * X1_f + b_const * X2_f

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
mse_phase = np.mean((np.unwrap(np.angle(y_f)) - np.unwrap(np.angle(y_theory))) ** 2)

print(f"[01. Linearity] Magnitude MSE : {mse_mag:.6e}")
print(f"[01. Linearity] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, y, 'b'); axs[0, 0].set_title(r'$y(t) = a x_1(t) + b x_2(t)$')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f, np.unwrap(np.angle(y_f)), 'b', label='Numerical'); axs[1, 0].plot(f, np.unwrap(np.angle(y_theory)), 'r--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 02. TIME SHIFTING
    # -------------------------------------------------------------------------
    "02_time_shifting.py": '''"""
===============================================================================
PROPERTY 02: TIME SHIFTING
Formula: CFT{ x(t - t0) } = X(f) * exp(-j * 2 * pi * f * t0)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_cft(signal, t, f):
    cft = np.zeros(len(f), dtype=complex)
    for i, fi in enumerate(f):
        kernel = np.exp(-1j * 2 * np.pi * fi * t)
        cft[i] = np.trapezoid(signal * kernel, t) if hasattr(np, 'trapezoid') else np.trapz(signal * kernel, t)
    return cft

t = np.linspace(-6, 6, 2500)
f = np.linspace(-5, 5, 1000)
t0 = 1.5

x = np.exp(-2.0 * t**2)
y = np.exp(-2.0 * (t - t0)**2)

x_f = compute_cft(x, t, f)
y_f = compute_cft(y, t, f)
y_theory = x_f * np.exp(-1j * 2 * np.pi * f * t0)

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
sig_mask = np.abs(x_f) > 1e-3
mse_phase = np.mean((np.unwrap(np.angle(y_f))[sig_mask] - np.unwrap(np.angle(y_theory))[sig_mask]) ** 2)

print(f"[02. Time Shift] Magnitude MSE : {mse_mag:.6e}")
print(f"[02. Time Shift] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, x, 'k--', label='x(t)'); axs[0, 0].plot(t, y, 'r', label='x(t - t0)'); axs[0, 0].legend(); axs[0, 0].set_title('Time Domain')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='|Y(f)|'); axs[0, 1].plot(f, np.abs(x_f), 'r--', label='|X(f)|'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude (Invariant)')
axs[1, 0].plot(f, np.unwrap(np.angle(y_f)), 'b', label='Numerical'); axs[1, 0].plot(f, np.unwrap(np.angle(y_theory)), 'g--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase Spectrum')
axs[1, 1].plot(f, np.abs(np.unwrap(np.angle(y_f)) - np.unwrap(np.angle(y_theory))), 'purple'); axs[1, 1].set_title('Phase Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 03. FREQUENCY SHIFTING
    # -------------------------------------------------------------------------
    "03_frequency_shifting.py": '''"""
===============================================================================
PROPERTY 03: FREQUENCY SHIFTING (MODULATION)
Formula: CFT{ x(t) * exp(j * 2 * pi * f0 * t) } = X(f - f0)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_cft(signal, t, f):
    cft = np.zeros(len(f), dtype=complex)
    for i, fi in enumerate(f):
        kernel = np.exp(-1j * 2 * np.pi * fi * t)
        cft[i] = np.trapezoid(signal * kernel, t) if hasattr(np, 'trapezoid') else np.trapz(signal * kernel, t)
    return cft

def interp_complex(x_new, xp, yp):
    return np.interp(x_new, xp, yp.real) + 1j * np.interp(x_new, xp, yp.imag)

t = np.linspace(-5, 5, 2000)
f = np.linspace(-15, 15, 1500)
f0 = 4.0

x = np.exp(-1.5 * t**2)
y = x * np.exp(1j * 2 * np.pi * f0 * t)

x_f = compute_cft(x, t, f)
y_f = compute_cft(y, t, f)
y_theory = interp_complex(f - f0, f, x_f)

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
sig_mask = np.abs(y_theory) > 1e-3
mse_phase = np.mean((np.unwrap(np.angle(y_f))[sig_mask] - np.unwrap(np.angle(y_theory))[sig_mask]) ** 2)

print(f"[03. Frequency Shift] Magnitude MSE : {mse_mag:.6e}")
print(f"[03. Frequency Shift] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, y.real, 'b'); axs[0, 0].set_title(r'Re{$y(t) = x(t)e^{j2\pi f_0 t}$}')
axs[0, 1].plot(f, np.abs(x_f), 'k--', label='Base |X(f)|'); axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical |Y(f)|'); axs[0, 1].plot(f, np.abs(y_theory), 'r:', label='Theory |X(f-f0)|'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f, np.angle(y_f), 'b', label='Numerical'); axs[1, 0].plot(f, np.angle(y_theory), 'r--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 04. TIME SCALING
    # -------------------------------------------------------------------------
    "04_time_scaling.py": '''"""
===============================================================================
PROPERTY 04: TIME SCALING
Formula: CFT{ x(a * t) } = (1 / |a|) * X(f / a)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_cft(signal, t, f):
    cft = np.zeros(len(f), dtype=complex)
    for i, fi in enumerate(f):
        kernel = np.exp(-1j * 2 * np.pi * fi * t)
        cft[i] = np.trapezoid(signal * kernel, t) if hasattr(np, 'trapezoid') else np.trapz(signal * kernel, t)
    return cft

def interp_complex(x_new, xp, yp):
    return np.interp(x_new, xp, yp.real) + 1j * np.interp(x_new, xp, yp.imag)

t = np.linspace(-5, 5, 2500)
f = np.linspace(-10, 10, 1000)
a = 2.0

x = np.exp(-1.0 * t**2)
y = np.exp(-1.0 * (a * t)**2)

x_f = compute_cft(x, t, f)
y_f = compute_cft(y, t, f)
y_theory = (1.0 / np.abs(a)) * interp_complex(f / a, f, x_f)

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
mse_phase = np.mean((np.unwrap(np.angle(y_f)) - np.unwrap(np.angle(y_theory))) ** 2)

print(f"[04. Time Scaling] Magnitude MSE : {mse_mag:.6e}")
print(f"[04. Time Scaling] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, x, 'k--', label='x(t)'); axs[0, 0].plot(t, y, 'b', label=f'x({a}t)'); axs[0, 0].legend(); axs[0, 0].set_title('Time Domain')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f, np.angle(y_f), 'b', label='Numerical'); axs[1, 0].plot(f, np.angle(y_theory), 'r--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 05. SCALING AND FREQUENCY SHIFT
    # -------------------------------------------------------------------------
    "05_scaling_and_freq_shift.py": '''"""
===============================================================================
PROPERTY 05: COMBINED TIME SCALING AND FREQUENCY SHIFT
Formula: CFT{ x(a * t) * exp(j * 2 * pi * f0 * t) } = (1 / |a|) * X((f - f0) / a)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_cft(signal, t, f):
    cft = np.zeros(len(f), dtype=complex)
    for i, fi in enumerate(f):
        kernel = np.exp(-1j * 2 * np.pi * fi * t)
        cft[i] = np.trapezoid(signal * kernel, t) if hasattr(np, 'trapezoid') else np.trapz(signal * kernel, t)
    return cft

def interp_complex(x_new, xp, yp):
    return np.interp(x_new, xp, yp.real) + 1j * np.interp(x_new, xp, yp.imag)

def base_signal(t_in):
    sq = np.where(np.abs(t_in) <= 1.0, 1.0, 0.0)
    tr = np.where(np.abs(t_in) <= 1.0, 1.0 - np.abs(t_in), 0.0)
    return sq + tr

t = np.linspace(-5, 5, 3000)
f = np.linspace(-15, 15, 1200)
a, f0 = 2.0, 3.0

x = base_signal(t)
y = base_signal(a * t) * np.exp(1j * 2 * np.pi * f0 * t)

x_f = compute_cft(x, t, f)
y_f = compute_cft(y, t, f)
y_theory = (1.0 / np.abs(a)) * interp_complex((f - f0) / a, f, x_f)

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
sig_mask = np.abs(y_theory) > 1e-2
mse_phase = np.mean((np.unwrap(np.angle(y_f))[sig_mask] - np.unwrap(np.angle(y_theory))[sig_mask]) ** 2)

print(f"[05. Scaling + Freq Shift] Magnitude MSE : {mse_mag:.6e}")
print(f"[05. Scaling + Freq Shift] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, y.real, 'b'); axs[0, 0].set_title(r'Re{$y(t) = x(at)e^{j2\pi f_0 t}$}')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f, np.unwrap(np.angle(y_f)), 'b', label='Numerical'); axs[1, 0].plot(f, np.unwrap(np.angle(y_theory)), 'g--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 06. SCALING AND TIME SHIFT
    # -------------------------------------------------------------------------
    "06_scaling_and_time_shift.py": '''"""
===============================================================================
PROPERTY 06: COMBINED TIME SCALING AND TIME SHIFT
Formula: CFT{ x(a * (t - t0)) } = (1 / |a|) * X(f / a) * exp(-j * 2 * pi * f * t0)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_cft(signal, t, f):
    cft = np.zeros(len(f), dtype=complex)
    for i, fi in enumerate(f):
        kernel = np.exp(-1j * 2 * np.pi * fi * t)
        cft[i] = np.trapezoid(signal * kernel, t) if hasattr(np, 'trapezoid') else np.trapz(signal * kernel, t)
    return cft

def interp_complex(x_new, xp, yp):
    return np.interp(x_new, xp, yp.real) + 1j * np.interp(x_new, xp, yp.imag)

t = np.linspace(-6, 6, 3000)
f = np.linspace(-10, 10, 1000)
a, t0 = 2.0, 1.0

x = np.exp(-1.5 * t**2)
y = np.exp(-1.5 * (a * (t - t0))**2)

x_f = compute_cft(x, t, f)
y_f = compute_cft(y, t, f)
y_theory = (1.0 / np.abs(a)) * interp_complex(f / a, f, x_f) * np.exp(-1j * 2 * np.pi * f * t0)

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
sig_mask = np.abs(y_f) > 1e-3
mse_phase = np.mean((np.unwrap(np.angle(y_f))[sig_mask] - np.unwrap(np.angle(y_theory))[sig_mask]) ** 2)

print(f"[06. Scaling + Time Shift] Magnitude MSE : {mse_mag:.6e}")
print(f"[06. Scaling + Time Shift] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, x, 'k--', label='x(t)'); axs[0, 0].plot(t, y, 'b', label='x(a(t - t0))'); axs[0, 0].legend(); axs[0, 0].set_title('Time Domain')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f, np.unwrap(np.angle(y_f)), 'b', label='Numerical'); axs[1, 0].plot(f, np.unwrap(np.angle(y_theory)), 'g--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 07. TIME REVERSAL
    # -------------------------------------------------------------------------
    "07_time_reversal.py": '''"""
===============================================================================
PROPERTY 07: TIME REVERSAL
Formula: CFT{ x(-t) } = X(-f)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_cft(signal, t, f):
    cft = np.zeros(len(f), dtype=complex)
    for i, fi in enumerate(f):
        kernel = np.exp(-1j * 2 * np.pi * fi * t)
        cft[i] = np.trapezoid(signal * kernel, t) if hasattr(np, 'trapezoid') else np.trapz(signal * kernel, t)
    return cft

def interp_complex(x_new, xp, yp):
    return np.interp(x_new, xp, yp.real) + 1j * np.interp(x_new, xp, yp.imag)

t = np.linspace(-5, 5, 2000)
f = np.linspace(-10, 10, 1000)

x = np.where(t >= 0, np.exp(-2.0 * t), 0.0)
y = np.where(-t >= 0, np.exp(-2.0 * (-t)), 0.0)

x_f = compute_cft(x, t, f)
y_f = compute_cft(y, t, f)
y_theory = interp_complex(-f, f, x_f)

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
mse_phase = np.mean((np.unwrap(np.angle(y_f)) - np.unwrap(np.angle(y_theory))) ** 2)

print(f"[07. Time Reversal] Magnitude MSE : {mse_mag:.6e}")
print(f"[07. Time Reversal] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, x, 'k--', label='x(t)'); axs[0, 0].plot(t, y, 'b', label='x(-t)'); axs[0, 0].legend(); axs[0, 0].set_title('Time Domain')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f, np.angle(y_f), 'b', label='Numerical'); axs[1, 0].plot(f, np.angle(y_theory), 'r--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 08. COMPLEX CONJUGATION
    # -------------------------------------------------------------------------
    "08_complex_conjugation.py": '''"""
===============================================================================
PROPERTY 08: COMPLEX CONJUGATION
Formula: CFT{ x*(t) } = X*(-f)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_cft(signal, t, f):
    cft = np.zeros(len(f), dtype=complex)
    for i, fi in enumerate(f):
        kernel = np.exp(-1j * 2 * np.pi * fi * t)
        cft[i] = np.trapezoid(signal * kernel, t) if hasattr(np, 'trapezoid') else np.trapz(signal * kernel, t)
    return cft

def interp_complex(x_new, xp, yp):
    return np.interp(x_new, xp, yp.real) + 1j * np.interp(x_new, xp, yp.imag)

t = np.linspace(-5, 5, 2000)
f = np.linspace(-10, 10, 1000)

x = np.exp(-1.0 * t**2) * np.exp(1j * 2 * np.pi * 2.0 * t)
y = np.conj(x)

x_f = compute_cft(x, t, f)
y_f = compute_cft(y, t, f)
y_theory = np.conj(interp_complex(-f, f, x_f))

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
mse_phase = np.mean((np.unwrap(np.angle(y_f)) - np.unwrap(np.angle(y_theory))) ** 2)

print(f"[08. Conjugation] Magnitude MSE : {mse_mag:.6e}")
print(f"[08. Conjugation] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, x.imag, 'k--', label='Im{x(t)}'); axs[0, 0].plot(t, y.imag, 'r', label='Im{x*(t)}'); axs[0, 0].legend(); axs[0, 0].set_title('Imaginary Parts')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f, np.angle(y_f), 'b', label='Numerical'); axs[1, 0].plot(f, np.angle(y_theory), 'r--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 09. DIFFERENTIATION
    # -------------------------------------------------------------------------
    "09_differentiation.py": '''"""
===============================================================================
PROPERTY 09: TIME DIFFERENTIATION
Formula: CFT{ d^k x(t) / dt^k } = (j * 2 * pi * f)^k * X(f)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_cft(signal, t, f):
    cft = np.zeros(len(f), dtype=complex)
    for i, fi in enumerate(f):
        kernel = np.exp(-1j * 2 * np.pi * fi * t)
        cft[i] = np.trapezoid(signal * kernel, t) if hasattr(np, 'trapezoid') else np.trapz(signal * kernel, t)
    return cft

t = np.linspace(-5, 5, 2500)
f = np.linspace(-4, 4, 1000)

x = np.exp(-1.5 * t**2)
y = np.gradient(x, t)

x_f = compute_cft(x, t, f)
y_f = compute_cft(y, t, f)
y_theory = (1j * 2 * np.pi * f) * x_f

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
sig_mask = np.abs(y_theory) > 1e-2
mse_phase = np.mean((np.unwrap(np.angle(y_f))[sig_mask] - np.unwrap(np.angle(y_theory))[sig_mask]) ** 2)

print(f"[09. Differentiation] Magnitude MSE : {mse_mag:.6e}")
print(f"[09. Differentiation] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, x, 'k--', label='x(t)'); axs[0, 0].plot(t, y, 'r', label='dx/dt'); axs[0, 0].legend(); axs[0, 0].set_title('Time Domain Derivative')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f, np.unwrap(np.angle(y_f)), 'b', label='Numerical'); axs[1, 0].plot(f, np.unwrap(np.angle(y_theory)), 'g--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 10. TIME INTEGRATION
    # -------------------------------------------------------------------------
    "10_integration.py": '''"""
===============================================================================
PROPERTY 10: TIME INTEGRATION
Formula: CFT{ int x(tau) dtau } = X(f) / (j * 2 * pi * f)  (for X(0) = 0)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_cft(signal, t, f):
    cft = np.zeros(len(f), dtype=complex)
    for i, fi in enumerate(f):
        kernel = np.exp(-1j * 2 * np.pi * fi * t)
        cft[i] = np.trapezoid(signal * kernel, t) if hasattr(np, 'trapezoid') else np.trapz(signal * kernel, t)
    return cft

t = np.linspace(-5, 5, 3000)
f = np.linspace(-4, 4, 1000)
dt = t[1] - t[0]

x = -2.0 * t * np.exp(-t**2)
y = np.cumsum(x) * dt

x_f = compute_cft(x, t, f)
y_f = compute_cft(y, t, f)

with np.errstate(divide='ignore', invalid='ignore'):
    y_theory = x_f / (1j * 2 * np.pi * f)
    y_theory[f == 0] = 0.0

mask = np.abs(f) > 0.1
mse_mag = np.mean((np.abs(y_f)[mask] - np.abs(y_theory)[mask]) ** 2)

print(f"[10. Integration] Magnitude MSE (f != 0) : {mse_mag:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, x, 'k--', label='x(t)'); axs[0, 0].plot(t, y, 'b', label=r'$\\int x(\\tau)d\\tau$'); axs[0, 0].legend(); axs[0, 0].set_title('Integrated Signal')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f[mask], np.angle(y_f)[mask], 'b', label='Numerical'); axs[1, 0].plot(f[mask], np.angle(y_theory)[mask], 'g--', label='Theoretical'); axs[1, 0].legend(); axs[1, 0].set_title('Phase')
axs[1, 1].plot(f[mask], np.abs(np.abs(y_f)[mask] - np.abs(y_theory)[mask]), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 11. CONVOLUTION IN TIME
    # -------------------------------------------------------------------------
    "11_convolution.py": '''"""
===============================================================================
PROPERTY 11: TIME CONVOLUTION
Formula: CFT{ x1(t) * x2(t) } = X1(f) * X2(f)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_cft(signal, t, f):
    cft = np.zeros(len(f), dtype=complex)
    for i, fi in enumerate(f):
        kernel = np.exp(-1j * 2 * np.pi * fi * t)
        cft[i] = np.trapezoid(signal * kernel, t) if hasattr(np, 'trapezoid') else np.trapz(signal * kernel, t)
    return cft

t = np.linspace(-5, 5, 2000)
f = np.linspace(-5, 5, 1000)
dt = t[1] - t[0]

x1 = np.where(np.abs(t) <= 0.5, 1.0, 0.0)
x2 = np.where(np.abs(t) <= 0.5, 1.0, 0.0)
y = np.convolve(x1, x2, mode='same') * dt

X1_f = compute_cft(x1, t, f)
X2_f = compute_cft(x2, t, f)
y_f = compute_cft(y, t, f)
y_theory = X1_f * X2_f

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)
mse_phase = np.mean((np.angle(y_f) - np.angle(y_theory)) ** 2)

print(f"[11. Convolution] Magnitude MSE : {mse_mag:.6e}")
print(f"[11. Convolution] Phase MSE     : {mse_phase:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, x1, 'k--', label='x1=x2 (Rect)'); axs[0, 0].plot(t, y, 'b', label='y = x1*x2 (Triangle)'); axs[0, 0].legend(); axs[0, 0].set_title('Time Domain Convolution')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Magnitude')
axs[1, 0].plot(f, np.angle(y_f), 'b'); axs[1, 0].set_title('Phase Spectrum')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 12. MULTIPLICATION IN TIME
    # -------------------------------------------------------------------------
    "12_multiplication.py": '''"""
===============================================================================
PROPERTY 12: TIME MULTIPLICATION (FREQUENCY CONVOLUTION)
Formula: CFT{ x1(t) * x2(t) } = X1(f) conv X2(f)
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_cft(signal, t, f):
    cft = np.zeros(len(f), dtype=complex)
    for i, fi in enumerate(f):
        kernel = np.exp(-1j * 2 * np.pi * fi * t)
        cft[i] = np.trapezoid(signal * kernel, t) if hasattr(np, 'trapezoid') else np.trapz(signal * kernel, t)
    return cft

t = np.linspace(-5, 5, 2000)
f = np.linspace(-8, 8, 1000)
df = f[1] - f[0]

x1 = np.exp(-1.0 * t**2)
x2 = np.cos(2 * np.pi * 2.0 * t)
y = x1 * x2

X1_f = compute_cft(x1, t, f)
X2_f = compute_cft(x2, t, f)
y_f = compute_cft(y, t, f)
y_theory = np.convolve(X1_f, X2_f, mode='same') * df

mse_mag = np.mean((np.abs(y_f) - np.abs(y_theory)) ** 2)

print(f"[12. Multiplication] Magnitude MSE : {mse_mag:.6e}")

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs[0, 0].plot(t, y, 'b'); axs[0, 0].set_title(r'Product $y(t) = x_1(t)x_2(t)$')
axs[0, 1].plot(f, np.abs(y_f), 'b', label='Numerical'); axs[0, 1].plot(f, np.abs(y_theory), 'r--', label='Theoretical'); axs[0, 1].legend(); axs[0, 1].set_title('Frequency Convolution')
axs[1, 0].plot(f, np.angle(y_f), 'b'); axs[1, 0].set_title('Phase Spectrum')
axs[1, 1].plot(f, np.abs(np.abs(y_f) - np.abs(y_theory)), 'purple'); axs[1, 1].set_title('Magnitude Error')
for ax in axs.flat: ax.grid(True)
plt.tight_layout(); plt.show()
''',

    # -------------------------------------------------------------------------
    # 13. PARSEVAL'S THEOREM
    # -------------------------------------------------------------------------
    "13_parsevals_theorem.py": '''"""
===============================================================================
PROPERTY 13: PARSEVAL'S THEOREM (ENERGY CONSERVATION)
Formula: int |x(t)|^2 dt == int |X(f)|^2 df
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_cft(signal, t, f):
    cft = np.zeros(len(f), dtype=complex)
    for i, fi in enumerate(f):
        kernel = np.exp(-1j * 2 * np.pi * fi * t)
        cft[i] = np.trapezoid(signal * kernel, t) if hasattr(np, 'trapezoid') else np.trapz(signal * kernel, t)
    return cft

t = np.linspace(-5, 5, 2500)
f = np.linspace(-15, 15, 1500)

x = np.exp(-2.0 * t**2)
x_f = compute_cft(x, t, f)

energy_time = np.trapezoid(np.abs(x)**2, t) if hasattr(np, 'trapezoid') else np.trapz(np.abs(x)**2, t)
energy_freq = np.trapezoid(np.abs(x_f)**2, f) if hasattr(np, 'trapezoid') else np.trapz(np.abs(x_f)**2, f)
diff = np.abs(energy_time - energy_freq)

print(f"[13. Parseval] Energy Time Domain : {energy_time:.6f} J")
print(f"[13. Parseval] Energy Freq Domain : {energy_freq:.6f} J")
print(f"[13. Parseval] Difference         : {diff:.6e} J")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].plot(t, np.abs(x)**2, 'b'); axs[0].set_title(r'Instantaneous Energy $|x(t)|^2$')
axs[1].plot(f, np.abs(x_f)**2, 'r'); axs[1].set_title(r'Energy Spectral Density $|X(f)|^2$')
for ax in axs: ax.grid(True)
plt.tight_layout(); plt.show()
'''
}

# Write files into target directory
for fname, content in files.items():
    fpath = os.path.join(TARGET_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

# Create a zip archive right inside that directory as well
zip_path = os.path.join(TARGET_DIR, "fourier_properties_lab.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fname in files:
        z.write(os.path.join(TARGET_DIR, fname), arcname=fname)

print("=" * 70)
print(f"SUCCESS: All 13 scripts created in:")
print(f"--> {TARGET_DIR}")
print(f"--> Zip file: {zip_path}")
print("=" * 70)

# Automatically open the folder in Windows File Explorer
try:
    os.startfile(TARGET_DIR)
except Exception:
    pass