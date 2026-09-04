
import sys
import math
import numpy as np

sys.setrecursionlimit(300000)

inf = 10**18
M = 10**9 + 7
mod = 998244353
eps = 1e-7
pi = 2 * math.acos(0)

# Helper function to get individual tokens line-by-line
def token_generator():
    for line in sys.stdin:
        for token in line.split():
            yield token

def solve(iterator):
    # print("degree of p : ")
    pdeg = int(next(iterator))
    p = [0] * (pdeg + 1)
    for i in range(pdeg + 1):
        p[i] = int(next(iterator))
    p = p[::-1]
    # print("degree of q : ")     
    qdeg = int(next(iterator))
    q = [0] * (qdeg + 1)
    for i in range(qdeg + 1):
        q[i] = int(next(iterator))
    q = q[::-1]

    w = [0] * (pdeg + 1)
    for i in range(pdeg + 1):
        w[i] = int(next(iterator))
    w = w[::-1]

    a = [0] * (pdeg + 1)
    for i in range(pdeg + 1):   
        print("a[i] : ", a[i])
        print(str(i) + "  " + str(pdeg + 1))
        a[i] = w[i] * p[i]


    anslen = pdeg + qdeg + 1

    anslen = next_power_of_two(anslen)

    aa = [0] * anslen
    aa[0:pdeg + 1] = a
    bb = [0] * anslen
    bb[0 : qdeg + 1] = q
    dft = DFTAnalyzer()
    xa = dft.transform(aa)
    xb = dft.transform(bb)

    xc = xa * xb
    print("xc : ", xc)

    cc = dft.inverse(xc)
    print("cc : ", cc)

    kitty = [0] * anslen

    for i in range(anslen):
        kitty[i] = round(cc[i].real)
    print("asnwer : ", kitty)

def next_power_of_two(n):
    """
    Return the smallest power of two that is >= ``n`` (and at least 1).

    Both tasks need this to choose a transform length for the radix-2 FFT.
    """
    if n <= 1:
        return 1
    return 1 << int(np.ceil(np.log2(n)))


class DFTAnalyzer:
    """
    The Discrete Fourier Transform, computed straight from its definition.

        Analysis:   X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
        Synthesis:  x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2j*pi*k*n/N)

    How you write it is up to you -- a literal double loop, a precomputed
    table of twiddle factors indexed by (k*n) % N, or a NumPy expression --
    as long as it computes these sums directly and is not secretly an FFT.
    """

    name = "dft"

    def transform(self, x):
        """
        Forward DFT.

        Parameters
        ----------
        x : 1D array_like, length N (real or complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
        """
        x_arr = np.asarray(x, dtype=np.complex128)
        N = x_arr.shape[0]
        if N == 0:
            return np.array([], dtype=np.complex128)
        k = np.arange(N).reshape((N, 1))
        n = np.arange(N).reshape((1, N))
        w = np.exp(-2j * np.pi * k * n / N)
        return np.dot(w, x_arr)

    def inverse(self, spectrum):
        """
        Inverse DFT, including the 1/N factor.

        Parameters
        ----------
        spectrum : 1D array_like, length N (complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
            Do NOT discard the imaginary part here -- the caller decides when
            it is safe to take .real.
        """
        x_arr = np.asarray(spectrum, dtype=np.complex128)
        N = x_arr.shape[0]
        if N == 0:
            return np.array([], dtype=np.complex128)
        k = np.arange(N).reshape((N, 1))
        n = np.arange(N).reshape((1, N))
        w = np.exp(2j * np.pi * k * n / N)
        return np.dot(w, x_arr) / N

    

def main():
    iterator = token_generator()

    t = 1
    
    try:
        t = int(next(iterator))
    except StopIteration:
        return
        
    for _ in range(t):
        solve(iterator)

if __name__ == '__main__':
    main()

# 1
# 4
# 1 3 2 6 7
# 1
# 4 1
# 3 2 1 5 6