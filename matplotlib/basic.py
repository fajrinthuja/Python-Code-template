import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.setrecursionlimit(300000)

inf = 10**18
M = 10**9 + 7
mod = 998244353
eps = 1e-7
pi = 2 * math.acos(0)


# ----------------------------------------------------------------------
# Helper: read whitespace-separated tokens line by line from stdin
# ----------------------------------------------------------------------
def token_generator():
    for line in sys.stdin:
        for token in line.split():
            yield token


# ----------------------------------------------------------------------
# 1. Single points (scatter-style)
#    plt.plot(x, y, marker, linestyle) with linestyle='' or fmt 'o'/'x'
#    draws ONLY markers, no connecting line.
# ----------------------------------------------------------------------
def demo_points():
    plt.figure()
    plt.plot(1, 1, marker='o', color='tab:blue', label='point (1,1)')
    plt.plot(2, 2, marker='x', color='tab:red', label='point (2,2)')
    # a list of points plotted as markers only (no line):
    xs = [0, 1, 2, 3, 4]
    ys = [3, 1, 4, 1, 5]
    plt.plot(xs, ys, 'o', color='tab:green', label='point cloud')
    plt.title('Plotting individual points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    # plt.show()


# ----------------------------------------------------------------------
# 2. Horizontal and vertical lines
#    plt.axhline(y=..) / plt.axvline(x=..) draw lines across the WHOLE
#    axes. plt.hlines / plt.vlines draw a line segment between limits.
# ----------------------------------------------------------------------
def demo_hv_lines():
    plt.figure()
    # full-width horizontal line at y = 2
    plt.axhline(y=2, color='tab:orange', linestyle='--', label='axhline y=2')
    # full-height vertical line at x = 3
    plt.axvline(x=3, color='tab:purple', linestyle=':', label='axvline x=3')
    # a horizontal SEGMENT from x=0 to x=5 at y=0
    plt.hlines(y=0, xmin=0, xmax=5, color='black', label='hlines segment')
    # a vertical SEGMENT from y=-1 to y=1 at x=1
    plt.vlines(x=1, ymin=-1, ymax=1, color='gray', label='vlines segment')
    plt.xlim(-1, 6)
    plt.ylim(-2, 3)
    plt.title('Horizontal / vertical lines')
    plt.legend()
    plt.grid(True)


# ----------------------------------------------------------------------
# 3. Sine / cosine curves
#    Build a dense x-array with np.linspace so the curve looks smooth.
# ----------------------------------------------------------------------
def demo_sin_cos():
    x = np.linspace(0, 2 * np.pi, 400)   # 400 samples for a smooth curve
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    plt.figure()
    plt.plot(x, y_sin, label='sin(x)', color='tab:blue')
    plt.plot(x, y_cos, label='cos(x)', color='tab:red')
    plt.title('Sine and Cosine')
    plt.xlabel('x (radians)')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.8)  # x-axis reference
    plt.legend()
    plt.grid(True)


# ----------------------------------------------------------------------
# 4. Polynomial functions
#    Evaluate a polynomial (given by coefficients) over a range with
#    np.polyval, or build it manually with numpy vectorized ops.
# ----------------------------------------------------------------------
def demo_polynomial():
    x = np.linspace(-3, 3, 200)
    # coefficients highest degree first: 1*x^3 - 2*x^2 + 0*x - 1
    coeffs = [1, -2, 0, -1]
    y = np.polyval(coeffs, x)

    plt.figure()
    plt.plot(x, y, color='tab:green', label='x^3 - 2x^2 - 1')
    # also show a plain quadratic for comparison
    y2 = x ** 2 - 4
    plt.plot(x, y2, color='tab:orange', linestyle='--', label='x^2 - 4')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title('Polynomial functions')
    plt.xlim(-5,5)
    plt.ylim(-100,100)
    y3 = 3*x - 5
    plt.plot(x,y3,color = 'red', label = 'linear')

    plt.legend()
    plt.grid(True)


# ----------------------------------------------------------------------
# 5. Line styles, markers and colors cheat-sheet in one figure
# ----------------------------------------------------------------------
def demo_styles():
    x = np.linspace(0, 10, 20)
    plt.figure()
    plt.plot(x, x, linestyle='-',  marker='',  label="'-'  solid")
    plt.plot(x, x + 2, linestyle='--', marker='', label="'--' dashed")
    plt.plot(x, x + 4, linestyle='-.', marker='', label="'-.' dash-dot")
    plt.plot(x, x + 6, linestyle=':',  marker='', label="':'  dotted")
    plt.plot(x, x + 8, linestyle='',   marker='o', label="'o'  markers only")
    plt.title('Line style / marker reference')
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True)


# ----------------------------------------------------------------------
# 6. Scatter plot (explicit) with variable size / color
# ----------------------------------------------------------------------
def demo_scatter():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 60)
    y = rng.uniform(0, 10, 60)
    sizes = rng.uniform(20, 200, 60)
    colors = rng.uniform(0, 1, 60)

    plt.figure()
    sc = plt.scatter(x, y, s=sizes, c=colors, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='color value')
    plt.title('Scatter plot with size & color mapping')
    plt.grid(True)


# ----------------------------------------------------------------------
# 7. Bar chart
# ----------------------------------------------------------------------
def demo_bar():
    categories = ['A', 'B', 'C', 'D']
    values = [3, 7, 2, 5]
    plt.figure()
    plt.bar(categories, values, color='tab:cyan')
    plt.title('Bar chart')
    plt.ylabel('value')

def demo_vertical_bar():
    categories = ['A', 'B', 'C', 'D']
    values = [3, 7, 2, 5]
    plt.figure()
    plt.barh(categories, values, color='tab:cyan')
    plt.title('Bar chart')
    plt.xlabel('value')


# ----------------------------------------------------------------------
# 8. Subplots: combine several plots into one figure (grid layout)
# ----------------------------------------------------------------------
def demo_subplots():
    x = np.linspace(0, 2 * np.pi, 200)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    axes[0, 0].plot(x, np.sin(x), color='tab:blue', label = 'sinx')
    axes[0, 0].set_title('sin(x)')
    axes[0, 0].legend(loc = 'upper right')

    axes[0, 1].plot(x, np.cos(x), color='tab:red', label = 'cosx')
    axes[0, 1].set_title('cos(x)')
    axes[0,1].legend(loc = 'upper right')

    axes[1, 0].plot(x, x ** 2, color='tab:green', label = 'x ^ 2')
    axes[1, 0].set_title('x^2')
    axes[1,0].legend(loc = 'upper right')

    axes[1, 1].plot([1, 2, 3], [3, 1, 4], 'o--', color='tab:purple', label = 'random')
    axes[1, 1].set_title('points + dashed line')
    axes[1,1].legend(loc = 'upper right')
    # fig.legend()
    fig.suptitle('Subplot grid demo')
    fig.tight_layout()

# ----------------------------------------------------------------------
# 9. Stem Plot (Discrete-Time Signals)
#    stem() is mainly used in DSP for discrete-time signals.
#    linefmt   -> color/style of stems
#    markerfmt -> marker style/color
#    basefmt   -> baseline style
# ----------------------------------------------------------------------
def demo_stem():
    n = np.arange(-8, 9)
    x1 = np.cos(np.pi * n / 4)
    x2 = 0.7 * np.sin(np.pi * n / 3)

    plt.figure(figsize=(8,4))

    # First discrete signal
    markerline1, stemlines1, baseline1 = plt.stem(
        n,
        x1,
        linefmt='b-',
        markerfmt='bo',
        basefmt=' '
    )
    markerline1.set_label("cos($\\pi n/4$)")

    # Second discrete signal
    markerline2, stemlines2, baseline2 = plt.stem(
        n,
        x2,
        linefmt='r-',
        markerfmt='rs',
        basefmt=' '
    )
    markerline2.set_label("0.7 sin($\\pi n/3$)")

    plt.title("Stem Plot (Discrete-Time Signals)")
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()


# ----------------------------------------------------------------------
# 10. Multiple / Grouped Bar Charts
#    Demonstrates
#      1. Side-by-side bars
#      2. Bars at arbitrary x positions
# ----------------------------------------------------------------------
def demo_multiple_bars():

    labels = ['A', 'B', 'C', 'D']

    set1 = [5, 8, 4, 6]
    set2 = [6, 4, 7, 5]
    set3 = [3, 5, 6, 8]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(8,4))

    # Three bars beside each other
    plt.bar(x - width, set1,
            width=width,
            color='tab:blue',
            label='Set 1')

    plt.bar(x,
            set2,
            width=width,
            color='tab:red',
            label='Set 2')

    plt.bar(x + width,
            set3,
            width=width,
            color='tab:green',
            label='Set 3')

    plt.xticks(x, labels)
    plt.title("Grouped Bar Chart")
    plt.xlabel("Category")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(axis='y')

    # ----------------------------------------------------------
    # Another example:
    # Bars placed at arbitrary x-locations
    # ----------------------------------------------------------

    plt.figure(figsize=(8,4))

    xpos = [1, 3, 6, 8]

    plt.bar(xpos[0], 5, width=0.5, color='tab:blue')
    plt.bar(xpos[1], 3, width=0.5, color='tab:red')
    plt.bar(xpos[2], 7, width=0.5, color='tab:green')
    plt.bar(xpos[3], 4, width=0.5, color='tab:purple')

    plt.title("Bars at Different Positions")
    plt.xlabel("x")
    plt.ylabel("Height")
    plt.grid(axis='y')
# ----------------------------------------------------------------------
# Original-style solve(): reads n points from stdin and plots them,
# with a horizontal reference line and a sine curve overlaid.
# ----------------------------------------------------------------------
def solve(iterator):
    n = int(next(iterator))
    a = [int(next(iterator)) for _ in range(n)]

    y_const = [5] * n  # constant y-level for the input points

    plt.figure()
    # points only, no connecting line
    plt.plot(a, y_const, 'o', color='tab:blue', label='input points')

    # horizontal reference line across the input points' y-level
    plt.axhline(y=5, color='gray', linestyle='--', label='y = 5 (axhline)')

    # a vertical line marking the mean of the input
    if n > 0:
        mean_a = sum(a) / n
        plt.axvline(x=mean_a, color='tab:orange', linestyle=':', label=f'mean x = {mean_a:.2f}')

    # overlay a sine curve scaled/shifted to sit near the points
    if n > 0:
        lo, hi = min(a), max(a)
        pad = max(1, (hi - lo) * 0.1)
        xs = np.linspace(lo - pad, hi + pad, 400)
        ys = 3 * np.sin(xs) + 5  # amplitude 3, shifted up to y=5
        plt.plot(xs, ys, color='tab:red', label='3*sin(x)+5')

    plt.title('Input points with reference lines and sine overlay')
    plt.xlabel('a[i]')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Run all the standalone demos (safe to comment out any of these)
    # demo_points()
    # demo_hv_lines()
    # demo_sin_cos()
    # demo_polynomial()
    # demo_styles()
    # demo_scatter()
    # demo_bar()
    # demo_vertical_bar()
    # demo_subplots()
    # demo_stem()
    demo_multiple_bars()
    plt.show()  # shows all demo figures at once

    # Uncomment to also run the stdin-driven solve() on real input:
    # iterator = token_generator()
    # t = 1
    # for _ in range(t):
    #     solve(iterator)


if __name__ == '__main__':
    main()