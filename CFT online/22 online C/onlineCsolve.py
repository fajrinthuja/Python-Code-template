import numpy as np
import matplotlib.pyplot as plt

def square(t, width):
    return np.where(np.abs(t) <= width / 2, 1, 0)

def triangle(t, width):
    return np.where(np.abs(t) <= width / 2.0, 1.0 - (np.abs(t) / (width / 2.0)), 0.0)

class SignalGenerator:
    
    def createSignal(t, f_0 = 0, a = 1):
        t_=t
        t = a * t
        x = square(t,2) + triangle(t,2)
        if f_0:
            x = x * np.exp(1j * 2 * np.pi * f_0 * t_)
        return x

def compute_cft(signal, t, f):
    cft = np.zeros(len(f), dtype=complex)
    for i, freq in enumerate(f):
        ex = np.exp(-1j * 2 * np.pi * freq * t)
        integral = np.trapezoid(signal * ex, t)
        cft[i] = integral
    return cft

def compute_mse_mag(spectrum_num, spectrum_theo):
    """Mean Squared Error between magnitude spectra."""
    mag_num = np.abs(spectrum_num)
    mag_theo = np.abs(spectrum_theo)
    return np.mean((mag_num - mag_theo) ** 2)

def compute_mse_phase(spectrum_num, spectrum_theo, mask=None):
    """Mean Squared Error between unwrapped phase spectra."""
    phase_num = np.unwrap(np.angle(spectrum_num))
    phase_theo = np.unwrap(np.angle(spectrum_theo))
    if mask is not None:
        return np.mean((phase_num[mask] - phase_theo[mask]) ** 2)
    return np.mean((phase_num - phase_theo) ** 2)

import numpy as np

def interp_complex(x_new, xp, yp, left=None, right=None):
    """
    One-dimensional linear interpolation for complex (or real) arrays.
    Drop-in replacement for np.interp with identical argument order:
    
    Parameters:
        x_new : array_like - The x-coordinates at which to evaluate the interpolated values.
        xp    : 1-D sequence of floats - The x-coordinates of the data points (must be increasing).
        yp    : 1-D sequence of complex/floats - The y-coordinates of the data points.
        left  : Value to return for x_new < xp[0] (default is yp[0]).
        right : Value to return for x_new > xp[-1] (default is yp[-1]).
        
    Returns:
        y_new : The interpolated values, same shape as x_new (dtype=complex).
    """
    x_new = np.asarray(x_new)
    xp = np.asarray(xp)
    yp = np.asarray(yp)

    # If already real, standard np.interp works directly
    if not np.iscomplexobj(yp):
        return np.interp(x_new, xp, yp, left=left, right=right)

    # Boundary handling for complex values
    left_real = left.real if left is not None else None
    left_imag = left.imag if left is not None else None
    right_real = right.real if right is not None else None
    right_imag = right.imag if right is not None else None

    # Interpolate real and imaginary parts independently
    y_real = np.interp(x_new, xp, yp.real, left=left_real, right=right_real)
    y_imag = np.interp(x_new, xp, yp.imag, left=left_imag, right=right_imag)

    return y_real + 1j * y_imag

sg = SignalGenerator
t = np.linspace(-5,5,2000)
f = np.linspace(-10,10,1000)
x = sg.createSignal(t)
y = sg.createSignal(t,10,10)

y_f = compute_cft(y, t, f)
x_f = compute_cft(x, t, f)

fnew = (f - 10) / 10
y_theory = interp_complex(fnew, f, x_f)

fig, axes= plt.subplots(2,2, figsize = (8,4))
axes[0,0].plot(f, abs(y_f), color = 'b')
axes[0,1].plot(f, abs(y_theory), color = 'r')

axes[1,0].plot(f, np.unwrap(np.angle(y_f)), color = 'b')
axes[1,1].plot(f, np.unwrap(np.angle(y_theory)), color = 'r')
fig.tight_layout()
plt.show()

print(compute_mse_mag(y_f, y_theory))
print(compute_mse_phase(y_f, y_theory))
