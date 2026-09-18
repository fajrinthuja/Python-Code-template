"""
================================================================================
STEP-BY-STEP NUMERICAL DEMONSTRATION OF THE SAMPLING THEOREM
(implements exactly the formula shown on the slide)

    X_p(jw) = (1/2pi) X(jw) * (2pi/T) sum_k delta(w - k*ws)
            = (1/T)   sum_k X( j(w - k*ws) )

Pipeline:
  x(t)  --F-->  X(jw)
  p(t)  --F-->  P(jw)          (impulse train, spacing ws = 2*pi/T)
  x_p(t)=x(t)*p(t)  --F-->  X_p(jw) = shifted copies of X(jw), scaled by 1/T
  xr(t) = reconstruction of x(t) from the samples, done TWO independent ways:
      (a) DIRECT       -> Whittaker-Shannon sinc interpolation in the time domain
      (b) VIA SPECTRUM -> ideal low-pass filter on X_p(jw), then inverse FFT
Both are plotted together so we can visually confirm they agree with x(t).
================================================================================
"""

# ---- STEP 0: import the libraries we need ------------------------------------
import numpy as np                                # arrays, FFT, sinc
import matplotlib.pyplot as plt                    # plotting
from matplotlib.gridspec import GridSpec           # lay the 7 plots out in columns

# ---- STEP 1: build a dense "continuous-time" grid -----------------------------
N        = 4096                          # number of points in the dense (quasi-continuous) grid
dt       = 4.0 / N                       # time step of the dense grid (4-second window / N points)
t        = (np.arange(N) - N // 2) * dt  # dense time axis, centred at 0  (-2 s ... +2 s)
T_window = N * dt                        # total window length = 4 s
df       = 1.0 / T_window                # FFT frequency resolution = 0.25 Hz
f = np.fft.fftshift(np.fft.fftfreq(N, d=dt))   # dense frequency axis (Hz), centred at 0

# ---- STEP 2: define X(jw) directly, as a smooth band-limited "hump" -----------
fM = 6.0                                 # message bandwidth in Hz  ( = w_M / 2*pi on the slide )
# raised-cosine shape, exactly zero for |f| >= fM  ->  X(jw) is truly band-limited
X_f = np.where(np.abs(f) < fM, np.cos(np.pi * f / (2 * fM)), 0.0).astype(complex)

# ---- STEP 3: get x(t) itself by inverse-Fourier-transforming X(jw) ------------
x_t = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(X_f))) * (N * df)   # proper (scaled) inverse FT
x_t = x_t.real                            # X(f) is real & even -> x(t) must come out real

# ---- STEP 4: choose the sampling rate and build p(t), the impulse train -------
fs = 16.0                                 # sampling frequency in Hz  ( = w_s / 2*pi on the slide )
T  = 1.0 / fs                             # sampling period T
n_idx    = np.arange(-32, 32)             # sample indices n
t_sample = n_idx * T                      # impulse locations t = nT  (where p(t) "fires")

# ---- STEP 5: multiply x(t) by p(t)  ->  x_p(t) = x(nT) at every sample instant -
x_samples = np.interp(t_sample, t, x_t)   # x(t) evaluated exactly AT each sample instant

# ---- STEP 6: build P(jw): impulses spaced ws = 2*pi*fs apart, height 2*pi/T ----
k_p   = np.arange(-3, 4)
f_p   = k_p * fs
amp_p = np.full_like(f_p, 2 * np.pi / T, dtype=float)

# ---- STEP 7: build X_p(jw) = (1/T) * sum_k X(j(w - k*ws))  --------------------
shift_bins = int(round(fs / df))          # how many FFT bins one period ws corresponds to
Xp_f = np.zeros_like(X_f)
for k in range(-4, 5):                    # sum enough shifted copies to fill the plotted range
    Xp_f += np.roll(X_f, k * shift_bins)  # shifting X(f) by k*fs = convolving with impulse train
Xp_f = Xp_f / T                           # scale by 1/T, exactly as in the formula

# ---- STEP 8a: RECONSTRUCTION METHOD 1 - DIRECT (Whittaker-Shannon sinc) -------
# xr(t) = sum_n x(nT) * sinc( (t - nT)/T )   <- computed purely in TIME, no FFT involved
t_recon = t[(t > -1) & (t < 1)]                       # zoom to the centre, away from edge effects
xr_direct = np.zeros_like(t_recon)
for xn, tn in zip(x_samples, t_sample):
    xr_direct += xn * np.sinc((t_recon - tn) / T)     # np.sinc(x) = sin(pi x)/(pi x)

# ---- STEP 8b: RECONSTRUCTION METHOD 2 - VIA THE SPECTRUM (ideal LPF + IFFT) ---
H_f  = np.where(np.abs(f) < fs / 2, T, 0.0)           # ideal low-pass filter, gain T, cutoff fs/2
Xr_f = Xp_f * H_f                                     # keep only the baseband copy -> recovers X(jw)
xr_ifft_full = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Xr_f))) * (N * df)
xr_ifft_full = xr_ifft_full.real
xr_ifft = np.interp(t_recon, t, xr_ifft_full)         # same zoomed window as xr_direct

# quick numeric sanity check that both reconstructions agree with the original
max_err = np.max(np.abs(xr_direct - xr_ifft))
print(f"Max difference between the two reconstruction methods: {max_err:.2e}  (should be ~0)")

# ================================================================================
# STEP 9: PLOT ALL 7 GRAPHS, laid out in the SAME orientation as the slide
#         columns 1-3 = (x,p,xp) pairs [time row on top, spectrum row below]
#         column 4    = the two xr(t) reconstructions, overlaid for comparison
# ================================================================================
fig = plt.figure(figsize=(17, 8), dpi=100)            # sized to be clear on a monitor
gs  = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1.2], hspace=0.45, wspace=0.35)

tw = (t > -0.35) & (t < 0.35)                          # time-domain plotting window
fw = (np.abs(f) < 42)                                  # frequency-domain plotting window
nw = (t_sample > -0.35) & (t_sample < 0.35)             # which sample instants fall in tw
fpw = np.abs(f_p) < 42                                  # which impulses of P(jw) fall in fw

# --- column 1: x(t) and X(jw) --------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t[tw], x_t[tw], color="teal", lw=2)
ax1.axhline(0, color="gray", lw=0.6)
ax1.set_title("x(t)"); ax1.set_xlabel("t (s)")

ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(f[fw], np.abs(X_f[fw]), color="navy")
ax4.fill_between(f[fw], np.abs(X_f[fw]), color="steelblue", alpha=0.3)
ax4.set_title("X(jω)"); ax4.set_xlabel("f (Hz)")

# --- column 2: p(t) and P(jw) ---------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2.stem(t_sample[nw], np.ones(int(nw.sum())), basefmt=" ")
ax2.set_title("p(t)  (impulse train, spacing T)"); ax2.set_xlabel("t (s)")

ax5 = fig.add_subplot(gs[1, 1])
ax5.stem(f_p[fpw], amp_p[fpw], basefmt=" ")
ax5.set_title("P(jω)  (impulses spaced ωs)"); ax5.set_xlabel("f (Hz)")

# --- column 3: x_p(t) and X_p(jw) ------------------------------------------------
ax3 = fig.add_subplot(gs[0, 2])
ax3.stem(t_sample[nw], x_samples[nw], basefmt=" ")
ax3.set_title("x_p(t) = x(t)·p(t)"); ax3.set_xlabel("t (s)")

ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(f[fw], np.abs(Xp_f[fw]), color="darkgreen")
ax6.fill_between(f[fw], np.abs(Xp_f[fw]), color="mediumseagreen", alpha=0.3)
ax6.set_title("X_p(jω) = (1/T)Σ X(j(ω-kωs))"); ax6.set_xlabel("f (Hz)")

# --- column 4 (spans both rows): the two reconstructions of xr(t) ---------------
ax7 = fig.add_subplot(gs[:, 3])
ax7.plot(t[(t > -0.5) & (t < 0.5)], x_t[(t > -0.5) & (t < 0.5)],
         color="teal", lw=1, ls=":", label="original x(t)")
ax7.plot(t_recon, xr_direct, color="orange", lw=3, alpha=0.8,
         label="xr(t) — DIRECT (sinc interp.)")
ax7.plot(t_recon, xr_ifft, color="black", lw=1.3, ls="--",
         label="xr(t) — via IFFT of filtered X_p")
ax7.set_title("Reconstruction xr(t)\ndirect vs. IFFT — both match x(t)")
ax7.set_xlabel("t (s)")
ax7.legend(fontsize=8, loc="upper right")

fig.suptitle(
    "Sampling Theorem: every impulse becomes a copy of X(jω)  —  reconstruction recovers x(t)",
    fontsize=13, fontweight="bold"
)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# ---- STEP 10: just display the figure (no file saving) -------------------------
plt.show()