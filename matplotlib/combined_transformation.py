import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------
# Time axis
# ------------------------------------
T_MIN = -4
T_MAX = 4
N = 4001

# ------------------------------------
# Original signal
# (Replace this if another signal is given)
# ------------------------------------
def x_of_t(t):
    return (
        np.sin(2 * np.pi * 0.5 * t)
        + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    )

# ------------------------------------
# Interpolation
# ------------------------------------
def interpolate_signal(t_original, x_original, t_query):
    return np.interp(t_query, t_original, x_original)

# ------------------------------------
# Time Shift
# y(t)=x(t-shift)
# shift>0 -> right
# shift<0 -> left
# ------------------------------------
def time_shift(t, x, shift):
    t_query = t - shift
    return interpolate_signal(t, x, t_query)

# ------------------------------------
# Time Scaling
# y(t)=x(scale*t)
#
# scale>1   -> compression
# 0<scale<1 -> expansion
# ------------------------------------
def time_scale(t, x, scale):
    t_query = scale * t
    return interpolate_signal(t, x, t_query)

# ------------------------------------
# Time Reversal
# y(t)=x(-t)
# ------------------------------------
def time_reverse(t, x):
    t_query = -t
    return interpolate_signal(t, x, t_query)

# ------------------------------------
# Combined Transformation
# y(t)=x(a*t+b)
#
# Examples:
# a=-2,b=3  -> x(-2t+3)
# a=3,b=-6  -> x(3t-6)
# a=1,b=-2  -> x(t-2)
# ------------------------------------
def combined_transform(t, x, a, b):
    t_query = a * t + b
    return interpolate_signal(t, x, t_query)

# ------------------------------------
# Plot
# ------------------------------------
def plot_pair(t, x, y, title):
    plt.figure(figsize=(8,5))

    plt.plot(t, x, label="Original x(t)", linewidth=2)
    plt.plot(t, y, label="Transformed", linewidth=2)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()

# ------------------------------------
# Main
# ------------------------------------
def main():

    t = np.linspace(T_MIN, T_MAX, N)
    x = x_of_t(t)

    # -----------------------------
    # Shift
    # -----------------------------
    y = time_shift(t, x, 2)
    plot_pair(t, x, y, "y(t)=x(t-2)")

    # -----------------------------
    # Scaling
    # -----------------------------
    y = time_scale(t, x, 2)
    plot_pair(t, x, y, "y(t)=x(2t)")

    # -----------------------------
    # Reversal
    # -----------------------------
    y = time_reverse(t, x)
    plot_pair(t, x, y, "y(t)=x(-t)")

    # -----------------------------
    # Combined
    # x(-2t+3)
    # -----------------------------
    y = combined_transform(t, x, -2, 3)
    plot_pair(t, x, y, "y(t)=x(-2t+3)")


if __name__ == "__main__":
    main()