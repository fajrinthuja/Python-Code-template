import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Signal Generator Framework
# ==============================================================================

class SignalGenerator:
    def __init__(self, t):
        self.t = t

    def gaussian(self, a=1.0, t0=0.0):
        """Gaussian signal: x(t) = exp(-a * (t - t0)^2)"""
        return np.exp(-a * ((self.t - t0) ** 2))

    def rect(self, width=1.0, t0=0.0):
        """Rectangular pulse: 1 for |t - t0| <= width/2, 0 elsewhere"""
        return np.where(np.abs(self.t - t0) <= (width / 2.0), 1.0, 0.0)

    def square_period(self, duty=0.5):
        """One period of a square wave over the given t array"""
        t_span = self.t[-1] - self.t[0]
        t_norm = (self.t - self.t[0]) / t_span
        return np.where(t_norm < duty, 1.0, 0.0)


# ==============================================================================
# 2. 1D Fourier Analyzer (Fourier Series & Continuous Fourier Transform)
# ==============================================================================

class FourierAnalyzer1D:
    @staticmethod
    def _integrate(integrand, t):
        """Numerical integration compatible across NumPy versions."""
        if hasattr(np, 'trapezoid'):
            return np.trapezoid(integrand, t)
        return np.trapz(integrand, t)

    @classmethod
    def compute_cft(cls, signal, t, freq_axis):
        """
        Continuous Fourier Transform:
        X(f) = \int x(t) * e^(-j * 2*pi * f * t) dt
        """
        X_f = np.zeros(len(freq_axis), dtype=complex)
        for i, fi in enumerate(freq_axis):
            kernel = np.exp(-1j * 2 * np.pi * fi * t)
            X_f[i] = cls._integrate(signal * kernel, t)
        return X_f

    @classmethod
    def compute_fourier_series(cls, signal, t):
        """
        Computes Fourier Series for every point corresponding to the time array length.
        - Fundamental Period: T0 = t_max - t_min
        - Fundamental Frequency: f0 = 1 / T0
        - Frequency grid: centered around 0 with len(t) points.
        """
        N = len(t)
        T0 = t[-1] - t[0]
        f0 = 1.0 / T0

        # Symmetrical frequency array centered at 0 with length N (same as t)
        k_indices = np.arange(-N // 2, N // 2)
        freq_grid = k_indices * f0

        cn = np.zeros(N, dtype=complex)
        for i, fi in enumerate(freq_grid):
            kernel = np.exp(-1j * 2 * np.pi * fi * t)
            cn[i] = (1.0 / T0) * cls._integrate(signal * kernel, t)

        return freq_grid, cn, T0

    @classmethod
    def reconstruct_fourier_series(cls, cn, freq_grid, t_recon):
        """
        Reconstructs signal using computed FS coefficients and frequency grid:
        x(t) = \sum c_n * e^(j * 2*pi * f * t)
        """
        x_recon = np.zeros(len(t_recon), dtype=complex)
        for c, fi in zip(cn, freq_grid):
            x_recon += c * np.exp(1j * 2 * np.pi * fi * t_recon)
        return x_recon.real


# ==============================================================================
# 3. Error Metrics
# ==============================================================================

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

# ==============================================================================
# 4. Verification & Testing
# ==============================================================================

if __name__ == "__main__":
    
    # --------------------------------------------------------------------------
    # Part A: Fourier Series (Evaluated across the full time array)
    # --------------------------------------------------------------------------
    T0 = 2.0
    t_fs = np.linspace(-T0 / 2, T0 / 2, 200)  # 200 time samples
    sg_fs = SignalGenerator(t_fs)
    
    # Rectangular period: 1 for |t| <= 0.5, 0 elsewhere
    x_period = np.where(np.abs(t_fs) <= 0.5, 1.0, 0.0)

    # Compute FS directly across the length of t_fs
    fs_freqs, cn_num, T0_computed = FourierAnalyzer1D.compute_fourier_series(x_period, t_fs)

    # Theoretical Fourier Series: c(f) = (width/T0) * sinc(f * width) = 0.5 * sinc(f * 1.0)
    cn_theo = 0.5 * np.sinc(fs_freqs * 1.0)

    fs_mag_mse = compute_mse_mag(cn_num, cn_theo)
    print("=" * 65)
    print("FOURIER SERIES (EVALUATED ACROSS FULL TIME VECTOR LENGTH)")
    print(f"Number of Spectral Points: {len(fs_freqs)} (matches len(t))")
    print(f"Fundamental Period (T0) : {T0_computed:.2f} s")
    print(f"FS Magnitude MSE         : {fs_mag_mse:.6e}")

    # Reconstruct over 3 periods
    t_recon = np.linspace(-3, 3, 1000)
    x_recon = FourierAnalyzer1D.reconstruct_fourier_series(cn_num, fs_freqs, t_recon)

    # --------------------------------------------------------------------------
    # Part B: Continuous Fourier Transform (CFT)
    # --------------------------------------------------------------------------
    t_cft = np.linspace(-5, 5, 2000)
    sg_cft = SignalGenerator(t_cft)
    
    x_t = sg_cft.gaussian(a=1.0, t0=0.0)
    y_t = sg_cft.gaussian(a=1.0, t0=1.0)  # Shifted by t0 = 1.0

    f_cft = np.linspace(-4, 4, 1000)
    X_f = FourierAnalyzer1D.compute_cft(x_t, t_cft, f_cft)
    Y_f = FourierAnalyzer1D.compute_cft(y_t, t_cft, f_cft)

    # Theoretical time shift: Y(f) = X(f) * exp(-j * 2*pi * f * t0)
    t0 = 1.0
    Y_f_theo = X_f * np.exp(-1j * 2 * np.pi * f_cft * t0)

    cft_mag_mse = compute_mse_mag(Y_f, Y_f_theo)
    sig_mask = np.abs(X_f) > 1e-3
    cft_phase_mse = compute_mse_phase(Y_f, Y_f_theo, mask=sig_mask)

    print("-" * 65)
    print("CONTINUOUS FOURIER TRANSFORM (CFT)")
    print(f"CFT Magnitude MSE        : {cft_mag_mse:.6e}")
    print(f"CFT Phase MSE            : {cft_phase_mse:.6e}")
    print("=" * 65)

    # --------------------------------------------------------------------------
    # 5. Visualizations
    # --------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # FS Coefficients (Zoomed to central band)
    axs[0, 0].plot(fs_freqs, np.abs(cn_num), 'b-', label='Numerical $|c_n|$')
    axs[0, 0].plot(fs_freqs, np.abs(cn_theo), 'r--', label='Theoretical')
    axs[0, 0].set_xlim([-10, 10])
    axs[0, 0].set_title("Fourier Series Spectrum ($f = k / T_0$)")
    axs[0, 0].set_xlabel("Frequency (Hz)")
    axs[0, 0].set_ylabel("Magnitude")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # FS Signal Reconstruction
    axs[0, 1].plot(t_recon, x_recon, color='purple', label='Reconstructed $x(t)$')
    axs[0, 1].set_title("Reconstructed Periodic Signal")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # CFT Magnitude Spectra
    axs[1, 0].plot(f_cft, np.abs(X_f), 'b', label='$|X(f)|$')
    axs[1, 0].plot(f_cft, np.abs(Y_f), 'r--', label='$|Y(f)|$')
    axs[1, 0].set_title("CFT Magnitude Spectrum")
    axs[1, 0].set_xlabel("Frequency (Hz)")
    axs[1, 0].set_ylabel("Magnitude")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # CFT Phase Shift
    axs[1, 1].plot(f_cft, np.unwrap(np.angle(Y_f)), 'r', label=r'$\angle Y(f)$ (Numerical)')
    axs[1, 1].plot(f_cft, np.unwrap(np.angle(Y_f_theo)), 'g--', label=r'$\angle X(f) - 2\pi f t_0$ (Theoretical)')
    axs[1, 1].set_xlim([-2, 2])
    axs[1, 1].set_title("CFT Phase Shift Comparison")
    axs[1, 1].set_xlabel("Frequency (Hz)")
    axs[1, 1].set_ylabel("Phase (rad)")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()