import numpy as np
import matplotlib.pyplot as plt

INF = 8

# =====================================================
# Plot Function
# =====================================================

def plot(
        signal,
        title=None,
        y_range=(-1, 3),
        figsize=(8,3),
        x_label='n (Time Index)',
        y_label='x[n]'
):

    plt.figure(figsize=figsize)

    plt.xticks(np.arange(-INF, INF+1))

    y_range = (
        y_range[0],
        max(np.max(signal), y_range[1]) + 1
    )

    plt.ylim(*y_range)

    plt.stem(np.arange(-INF, INF+1), signal)

    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()


# =====================================================
# Sample Signal
# =====================================================

def init_signal():

    x = np.zeros(2*INF+1)

    x[INF] = 1
    x[INF+1] = .5
    x[INF-1] = 2
    x[INF+2] = 1
    x[INF-2] = .5

    return x


# =====================================================
# Time Shift
# y[n]=x[n-k]
# =====================================================

def time_shift(x, k):

    y = np.roll(x, k)

    if k > 0:
        y[:k] = 0
    elif k < 0:
        y[k:] = 0

    return y


# =====================================================
# Time Reversal
# y[n]=x[-n]
# =====================================================

def time_reverse(x):

    return np.flip(x)


# =====================================================
# Time Compression
# y[n]=x[kn]
# k=2 -> x[2n]
# =====================================================

def time_compress(x, k):

    c = len(x)//2

    n = np.arange(-c, c+1)

    query = k*n

    y = np.interp(query, n, x)

    return y


# =====================================================
# Time Expansion
# y[n]=x[n/k]
# k=2 -> x[n/2]
# (zeros between samples)
# =====================================================

def time_expand(x, k):

    y = np.zeros_like(x)

    c = len(x)//2

    idx = np.arange(-c, c+1)

    valid = (idx % k == 0)

    new_idx = c + idx[valid]
    old_idx = (c + idx[valid]//k).astype(int)

    y[new_idx] = x[old_idx]

    return y


# =====================================================
# Time Expansion with Interpolation
# y[n]=x[n/k]
# =====================================================

def time_expand_interpolate(x, k):

    c = len(x)//2

    n = np.arange(-c, c+1)

    query = n/k

    return np.interp(query, n, x)


# =====================================================
# General Continuous-Time Transform
# y=x(an+b)
# =====================================================

def transform(x, a, b):

    c = len(x)//2

    n = np.arange(-c, c+1)

    query = a*n + b

    return np.interp(query, n, x)


# =====================================================
# Main
# =====================================================

def main():

    x = init_signal()

    plot(x, "Original")

    plot(time_shift(x, 2), "x[n-2]")

    plot(time_shift(x, -2), "x[n+2]")

    plot(time_reverse(x), "x[-n]")

    plot(time_compress(x, 2), "x[2n]")

    plot(time_expand(x, 2), "x[n/2]")

    plot(time_expand_interpolate(x, 2), "x[n/2] (Interpolation)")

    plot(transform(x, -2, 3), "x[-2n+3]")

    plot(transform(x, 2, -1), "x[2n-1]")

    plot(transform(x, 0.5, 2), "x[n/2+2]")


if __name__ == "__main__":
    main()