"""
================================================================================
NUMPY SYNTAX CHEAT SHEET — for the sampling lab
Run this file directly; every section prints its own output so you can see
exactly what each line produces.
================================================================================
"""
import numpy as np

# ============================================================================
# 1. CREATING ARRAYS
# ============================================================================

# --- np.arange(start, stop, step) ---
# Like Python's range(), but works with decimals and returns a numpy array.
# stop is EXCLUDED (never reached), step is INCLUDED (the spacing).
T = 0.25                              # spacing you want between values
a = np.arange(0, 1, T)                # 0, 0.25, 0.5, 0.75   (1.0 is NOT included)
print("arange(0,1,0.25):", a)

# --- T-spaced sample instants: t_n = n*T ---
# This is exactly how you build sample times for a sampling frequency fs:
fs = 40.0
T = 1 / fs                            # sampling period
t_n = np.arange(0, 1, T)              # sample instants 0, T, 2T, ... up to (not incl.) 1 second
print("t_n has", len(t_n), "samples, spaced", T, "s apart")

# --- np.linspace(start, stop, num) ---
# Gives you exactly `num` points, evenly spaced, and BOTH endpoints ARE included.
# Use this when you care about the total COUNT of points, not the exact spacing.
t_dense = np.linspace(0, 1, 2000)     # 2000 points from 0 to 1, inclusive of both ends
print("linspace step size:", t_dense[1] - t_dense[0])

# --- np.zeros / np.ones ---
z = np.zeros(5)                       # array of 5 zeros: [0. 0. 0. 0. 0.]
o = np.ones(5)                        # array of 5 ones:  [1. 1. 1. 1. 1.]
print("zeros:", z, " ones:", o)

# --- np.zeros_like(x) / np.ones_like(x) ---
# Same SHAPE (and dtype) as an existing array x, but filled with 0s or 1s.
# Very handy as an "empty accumulator" you add values into in a loop.
x_example = np.array([1.0, 2.0, 3.0])
acc = np.zeros_like(x_example)        # array([0., 0., 0.]) — same length as x_example
print("zeros_like:", acc)


# ============================================================================
# 2. SETTING VALUES INSIDE AN ARRAY (this is how you build an impulse train)
# ============================================================================

# --- basic indexing: arr[i] = value ---
p = np.zeros(10)
p[3] = 1.0                            # set just ONE element (index 3) to 1
print("single index set:", p)

# --- slicing with a step: arr[start:stop:step] = value ---
# This sets EVERY step-th element in one line — no loop needed.
p = np.zeros(20)
p[::4] = 1.0                          # every 4th element, starting at index 0, becomes 1
print("step-slice set (every 4th):", p)
# This exact pattern is how you build an impulse train p(t):
#   step = (dense grid rate) / (sampling rate fs)
#   p[::step] = 1.0        <-- a "1" (impulse) every T seconds

# --- boolean mask indexing: arr[condition] ---
# Comparing an array to a number gives back a True/False array of the same size.
# Using that True/False array inside [ ] keeps only the True positions.
vals = np.array([1, 5, 2, 8, 3, 9])
mask = vals > 4                       # array([False, True, False, True, False, True])
print("mask:", mask)
print("vals where mask is True:", vals[mask])    # array([5, 8, 9])

# combine conditions with & (and) / | (or) — NOT Python's `and`/`or`, those don't
# work elementwise on arrays. Always wrap each condition in ( ) when combining.
t = np.linspace(0, 1, 11)
window = (t > 0.2) & (t < 0.7)        # True only where BOTH conditions hold
print("t inside window:", t[window])

# --- np.nonzero(arr) ---
# Returns the INDICES where an array is nonzero (e.g. where your impulse train fired).
# It gives back a tuple (one array per dimension) — for a 1-D array, take [0].
p = np.zeros(10)
p[2] = 1
p[6] = 1
idx = np.nonzero(p)[0]                # array([2, 6]) — the positions of the impulses
print("nonzero indices:", idx)


# ============================================================================
# 3. ELEMENTWISE MATH & BROADCASTING (no loops needed)
# ============================================================================

# numpy math functions apply to EVERY element of an array at once
t = np.linspace(0, 1, 5)
x = np.cos(2 * np.pi * 5 * t)         # cos(2*pi*f*t) computed for all 5 points in one call
print("cos values:", x)

# multiplying two arrays of the SAME shape multiplies them elementwise (not matrix mult.)
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
print("elementwise multiply:", a * b)            # array([10, 40, 90])

# BROADCASTING: array (op) single number applies to every element
# e.g. this is how (t - t0) works for a whole array of times t against one instant t0
t = np.array([0.0, 0.1, 0.2, 0.3])
t0 = 0.1
print("t - t0 (broadcast):", t - t0)             # subtracts 0.1 from every entry


# ============================================================================
# 4. FFT-RELATED SYNTAX
# ============================================================================

fs = 500.0
n = np.arange(0, 500)                 # sample indices 0..499
t_n = n / fs                          # convert indices to actual seconds
x_n = np.cos(2 * np.pi * 50 * t_n)    # a 50 Hz signal, sampled

X = np.fft.fft(x_n)                   # DFT: returns a COMPLEX array (real + imaginary parts)
freqs = np.fft.fftfreq(len(x_n), d=1 / fs)   # matching frequency (Hz) label for each entry of X
print("X dtype:", X.dtype)                    # complex128
print("first 3 freqs:", freqs[:3])

mag = np.abs(X)                       # magnitude of each complex number: sqrt(real^2+imag^2)
print("first 3 magnitudes:", mag[:3])


# ============================================================================
# 5. OTHER SYNTAX USED A LOT IN THE LAB
# ============================================================================

# np.sinc(u) — NORMALIZED sinc: sin(pi*u)/(pi*u), and exactly 1 at u=0 (no divide-by-zero)
print("np.sinc(0):", np.sinc(0))
print("np.sinc([0, 0.5, 1]):", np.sinc(np.array([0, 0.5, 1])))

# zip(a, b) — walk through two arrays/lists together, in matched pairs
a = [1, 2, 3]
b = ["x", "y", "z"]
for num, letter in zip(a, b):
    print("zip pair:", num, letter)

# len(arr) and arr.shape — how big is the array
arr = np.zeros((5,))
print("len:", len(arr), " shape:", arr.shape)

# np.max / np.min / np.abs on arrays
diff = np.array([-3.0, 1.0, 5.0, -0.2])
print("max abs diff:", np.max(np.abs(diff)))

# round() and abs() — plain Python builtins, for SINGLE numbers (not arrays)
print("round(2.6):", round(2.6), " abs(-3):", abs(-3))

# f-strings with format specs — {value:.2e} = scientific notation, 2 decimal places
err = 0.0000431
print(f"formatted error: {err:.2e}")