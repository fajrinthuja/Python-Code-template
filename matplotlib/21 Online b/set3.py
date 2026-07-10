import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

INF = 8

def plot(
        signal, 
        title=None, 
        y_range=(-1, 3), 
        figsize = (8, 3),
        x_label='n (Time Index)',
        y_label='x[n]',
        saveTo=None
    ):
    plt.figure(figsize=figsize)
    plt.xticks(np.arange(-INF, INF + 1, 1))
    
    y_range = (y_range[0], max(np.max(signal), y_range[1]) + 1)
    # set y range of 
    plt.ylim(*y_range)
    plt.stem(np.arange(-INF, INF + 1, 1), signal)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if saveTo is not None:
        plt.savefig(saveTo)
    plt.show()

def init_signal():
    return np.zeros(2 * INF + 1)


def time_scale_signal(x : np.ndarray, k) -> np.ndarray:
    # implement this function
    k = 1 / k
    n = len(x);
    y = np.zeros(n)
    i = n // 2 
    j = 0
    n //= 2
    while j < n:
        if i + k * j < len(x) and int(i + k * j) == i + k * j:
            print(i + k * j)
            y[i + j] = x[int(i + k * j)]
        if i - k * j >= 0 and int(i + k * j) == i + k * j:
            print(i - k * j)
            y[i - j] = x[int(i - k * j)]
        j += 1
    # return y
    k = 1 / k
    y = np.zeros_like(x)

    c = len(x) // 2
    idx = np.arange(-c, c + 1)

    valid = (idx % k == 0)

    new_idx = c + idx[valid]
    old_idx = c + (idx[valid] // k).astype(int)

    y[new_idx] = x[old_idx]
    return y
    # None

def time_scale_signal_interpolate(x : np.ndarray, k) -> np.ndarray:
    # implement this function
    k = 1 / k
    n = len(x);
    y = np.zeros(n)
    i = n // 2 
    j = 0
    n //= 2
    while j < n:
        if i + k * j < len(x) and int(i + k * j) == i + k * j:
            print(i + k * j)
            y[i + j] = x[int(i + k * j)]
        elif i + k * j < len(x):
            num = i + k * j;
            num += 1
            num = int(num)
            id = num - 1
            y[i + j] = (x[num] + x[id]) / 2
        if i - k * j >= 0 and int(i + k * j) == i + k * j:
            print(i - k * j)
            y[i - j] = x[int(i - k * j)]
        elif i - k * j >= 0:
            num = i - k * j;
            # num += 1
            num = int(num)
            id = num + 1
            y[i - j] = (x[num] + x[id]) / 2
        j += 1
    # return y
    c = len(x) // 2

    n = np.arange(-c, c + 1)

    query = n * k

    return np.interp(query, n, x)
    None


def main():
    img_root = '.'
    signal = init_signal()
    signal[INF] = 1
    signal[INF+1] = .5
    signal[INF-1] = 2
    signal[INF + 2] = 1
    signal[INF - 2] = .5

    plot(signal, title='Original Signal(x[n])', saveTo=f'{img_root}/x[n].png')
    plot(time_scale_signal(signal, 3), title='x[n/3]', saveTo=f'{img_root}/x[n divided by 3].png')
    plot(time_scale_signal(signal, 1), title='x[n/1]', saveTo=f'{img_root}/x[n divided by 1].png')
    plot(time_scale_signal_interpolate(signal, 3), title='x[n/3] with interpolation', saveTo=f'{img_root}/x[n divided by 3]_with_interpolation.png')
    plot(time_scale_signal_interpolate(signal, 1), title='x[n/1] with interpolation', saveTo=f'{img_root}/x[n divided by 1]_with_interpolation.png')

main()
