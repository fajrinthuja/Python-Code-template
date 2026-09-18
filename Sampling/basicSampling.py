import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. PARAMETERS & TIME-DOMAIN SIGNALS
# ==========================================
f = 5          # Signal frequency (Hz)
fs = 30        # Sampling frequency (Hz)
duration = 1.0 # Duration (seconds)
N = 1000       # Number of points

t = np.linspace(0, duration, N, endpoint=False)
dt = t[1] - t[0]

# x(t) = cos(2*pi*f*t)
xt = np.cos(2 * np.pi * f * t)

# p(t) = sum(delta(t - nT_s))
T_s = 1 / fs
sample_indices = np.round(np.arange(0, duration, T_s) / dt).astype(int)
pt = np.zeros_like(t)
pt[sample_indices] = 1.0 

# x_s(t) = x(t) * p(t)
xs_t = xt * pt

# ==========================================
# 2. SPECTRA VIA FOURIER TRANSFORM
# ==========================================
freqs = np.fft.fftfreq(N, dt)
freqs_shifted = np.fft.fftshift(freqs)

X_w = np.fft.fft(xt)
P_w = np.fft.fft(pt)
X_s_w = np.fft.fft(xs_t)

X_w_shifted = np.fft.fftshift(X_w)
P_w_shifted = np.fft.fftshift(P_w)
X_s_shifted = np.fft.fftshift(X_s_w)

# ==========================================
# 3. RECONSTRUCTION WITH 1/T (or * T) SCALING
# ==========================================
# Ideal Low-Pass Filter (ILPF) mask: H(jw)
cutoff = fs / 2
lpf_mask = (np.abs(freqs) <= cutoff).astype(float)

# Since Xs is scaled by 1/T (fs), multiply by T (1/fs) during reconstruction
X_r_w = X_s_w * lpf_mask * (1 / fs)

# Final IFFT with standard discrete scaling
xr_t = np.real(np.fft.ifft(X_r_w)) * N

# ==========================================
# 4. VISUALIZATION (2 COLUMNS x 3 ROWS)
# ==========================================
fig, axes = plt.subplots(3, 2, figsize=(13, 7))

# --- Row 1: Signals x(t) and p(t) ---
axes[0, 0].plot(t, xt, color='blue', lw=1.5)
axes[0, 0].set_title('1. Continuous $x(t) = \\cos(2\\pi f t)$', fontsize=10)
axes[0, 0].grid(True)

axes[0, 1].plot(t, pt, color='orange', lw=1.2)
axes[0, 1].set_title('2. Impulse Train $p(t) = \\sum \\delta(t - nT_s)$', fontsize=10)
axes[0, 1].grid(True)

# --- Row 2: Spectra X(jw) and P(jw) ---
axes[1, 0].plot(freqs_shifted, np.abs(X_w_shifted), color='blue', lw=1.5)
axes[1, 0].set_title('3. Spectrum $X(j\\omega) = \\mathcal{F}\\{x(t)\\}$', fontsize=10)
axes[1, 0].set_xlim(-25, 25)
axes[1, 0].grid(True)

axes[1, 1].plot(freqs_shifted, np.abs(P_w_shifted), color='orange', lw=1.2)
axes[1, 1].set_title('4. Spectrum $P(j\\omega) = \\frac{2\\pi}{T}\\sum \\delta(\\omega - k\\omega_s)$', fontsize=10)
axes[1, 1].set_xlim(-45, 45)
axes[1, 1].grid(True)

# --- Row 3: Sampled Spectrum & Reconstruction ---
axes[2, 0].plot(freqs_shifted, np.abs(X_s_shifted), color='purple', lw=1.5)
axes[2, 0].set_title('5. Sampled Spectrum $X_s(j\\omega) = \\frac{1}{T} \\sum X(j\\omega - k\\omega_s)$', fontsize=10)
axes[2, 0].set_xlim(-45, 45)
axes[2, 0].grid(True)

axes[2, 1].plot(t, xt, label='Original $x(t)$', color='blue', alpha=0.4, lw=1.5)
axes[2, 1].plot(t, xr_t, label='Reconstructed $x_r(t)$', color='green', linestyle='--', lw=1.5)
axes[2, 1].set_title('6. Reconstruction: $x_r(t) = \\mathcal{F}^{-1}\\{T \\cdot X_s \\cdot H\\}$', fontsize=10)
axes[2, 1].legend(loc='upper right', fontsize=8)
axes[2, 1].grid(True)

plt.tight_layout()
plt.show()