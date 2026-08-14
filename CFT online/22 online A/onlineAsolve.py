import numpy as np
import matplotlib.pyplot as plt

tmax = 16 * np.pi  
tmin = -16 * np.pi 
dt = 0.005
t = np.arange(tmin, tmax, dt)

f = np.arange(-2,2, 0.005)

def compute_cft(signal, t_array, f_array):
    cft = np.zeros(len(f_array), dtype = complex)
    for i, freq in enumerate(f_array):
        ex = -1j * 2 * np.pi * freq * t_array
        ex = np.exp(ex)
        integrand = signal * ex
        cft[i] = np.trapezoid(integrand, t_array)
    return cft

x = 0.5 * np.cos(4 * t) + 0.5 * np.sin(6 * t)
y1 = - 2 * np.sin(4 * t) + 3 * np.cos(6 * t)
y2 = -8 * np.cos(4 * t) - 18 * np.sin(6 * t)
y3 = 32 * np.sin(4 * t) - 108 * np.cos(6 * t)

# -------------------------------------------------------------------------
# THE FIX: Apply a Hann Window to force the boundaries to exactly 0
# -------------------------------------------------------------------------
window = np.hanning(len(t))

x  = x  * window
y1 = y1 * window
y2 = y2 * window
y3 = y3 * window
# -------------------------------------------------------------------------

# Now compute the CFTs using these windowed signals
x_f = compute_cft(x, t, f)
y1_f = compute_cft(y1, t, f)
y2_f = compute_cft(y2, t, f)
y3_f = compute_cft(y3, t, f)

prop1_f = (1j * 2 * np.pi * f) ** 1 * x_f
prop2_f = (1j * 2 * np.pi * f) ** 2 * x_f
prop3_f = (1j * 2 * np.pi * f) ** 3 * x_f

fig, axes = plt.subplots(3,2,figsize = (14,10))

derivatives = [
    ("1st Derivative", y1_f, prop1_f),
    ("2nd Derivative", y2_f, prop2_f),
    ("3rd Derivative", y3_f, prop3_f)
]
# Loop through the 3 derivatives. 'idx' tells us which row (0, 1, or 2) to plot on.
for idx, (title, direct_cft, prop_cft) in enumerate(derivatives):
    
    # --- Column 0: Magnitude Plot ---
    # np.abs() calculates the magnitude (length) of the complex numbers.
    # We plot the direct integration as a solid blue line ('b-').
    axes[idx, 0].plot(f, np.abs(direct_cft), 'b-', label='Direct Integration', linewidth=1.5)
    
    # We plot the theoretical property as a dashed red line ('r--') right on top of it.
    # If the math works, the red dashes will perfectly overlap the blue line.
    axes[idx, 0].plot(f, np.abs(prop_cft), 'r--', label='Property: |(j2πf)^n X(f)|', linewidth=1.5)
    
    # Add labels and a grid for readability
    axes[idx, 0].set_title(f'{title}: Magnitude Comparison')
    axes[idx, 0].set_xlabel('Frequency (Hz)')
    axes[idx, 0].set_ylabel('Magnitude')
    axes[idx, 0].legend(loc='upper right')
    axes[idx, 0].grid(True)

    # --- Column 1: Phase Plot ---
    # np.angle() calculates the phase (angle in radians) of the complex numbers.
    axes[idx, 1].plot(f, np.angle(direct_cft), 'b-', label='Direct Integration Phase', linewidth=1.5)
    axes[idx, 1].plot(f, np.angle(prop_cft), 'r--', label='Property Phase', linewidth=1.5)
    
    # Add labels and a grid
    axes[idx, 1].set_title(f'{title}: Phase Comparison')
    axes[idx, 1].set_xlabel('Frequency (Hz)')
    axes[idx, 1].set_ylabel('Phase (radians)')
    axes[idx, 1].legend(loc='upper right')
    axes[idx, 1].grid(True)

# Adjust spacing so the graphs don't overlap, then show the window.
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# 6. Mean Squared Error (MSE) Analysis
# -------------------------------------------------------------------------

# This function measures how different two arrays are.
# It subtracts them, squares the difference (so all errors are positive), and averages them.
def compute_mse(arr1, arr2):
    return np.mean(np.abs(arr1 - arr2)**2)

# Calculate the MSE for the magnitude of all three derivatives.
# We compare the direct integration (Y_f) to the theoretical property (prop_f).
mse_mag_1 = compute_mse(np.abs(y1_f), np.abs(prop1_f))
mse_mag_2 = compute_mse(np.abs(y2_f), np.abs(prop2_f))
mse_mag_3 = compute_mse(np.abs(y3_f), np.abs(prop3_f))

# Print the final error numbers to the console.
print(f"--- Mean Squared Error (MSE) Analysis ---")
print(f"1st Derivative | Magnitude MSE: {mse_mag_1:.6e}")
print(f"2nd Derivative | Magnitude MSE: {mse_mag_2:.6e}")
print(f"3rd Derivative | Magnitude MSE: {mse_mag_3:.6e}")