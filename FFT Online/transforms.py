"""
transforms.py  --  YOUR CODE GOES HERE.

The shared transform core used by BOTH tasks. Write it once; bigmul.py
(Task A) and image_conv.py (Task B) import it.

Nothing in this file may call numpy.fft, scipy.fft, numpy.convolve,
scipy.signal, or any other library routine that performs a Fourier
transform, a convolution or a correlation for you. NumPy is for array
arithmetic only.

A quick self-test you should run before touching either application:

    import numpy as np
    from transforms import DFTAnalyzer, FFTTransformer
    x = np.random.randn(64) + 1j * np.random.randn(64)
    d, f = DFTAnalyzer(), FFTTransformer()
    assert np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9
    assert np.max(np.abs(d.inverse(d.transform(x)) - x)) < 1e-9
"""

import numpy as np


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


class FFTTransformer(DFTAnalyzer):
    """
    Radix-2 decimation-in-time (Cooley-Tukey) FFT, in O(N log N).

    It inherits from DFTAnalyzer so that both applications can treat the two
    interchangeably: they call ``engine.transform(...)`` and
    ``engine.inverse(...)`` without caring which engine they hold.

    Requirements:
      * Recursive or iterative (with bit-reversal permutation) -- your choice.
      * N must be a power of two; raise ValueError for any other length.
        The caller is responsible for zero-padding up to next_power_of_two.
      * The inverse must reuse the same butterfly machinery (conjugated
        twiddles, or conjugate-transform-conjugate), not a second copy of it.
      * Twiddle factors for a stage are computed once per stage, never once
        per butterfly.
    """

    name = "fft"

    def _fft_core(self, a, invert=False):
        N = a.shape[0]
        if N == 0:
            return np.array([], dtype=np.complex128)

        if (N & (N - 1)) != 0:
            raise ValueError(f"Array length {N} must be a power of two.")

        if N == 1:
            return a.astype(np.complex128)

        num_bits = int(np.log2(N))
        indices = np.arange(N)
        rev_indices = np.zeros(N, dtype=int)
        for bit in range(num_bits):
            rev_indices = (rev_indices << 1) | ((indices >> bit) & 1)

        res = a[rev_indices].astype(np.complex128)

        length = 2
        sign = 1.0 if invert else -1.0
        while length <= N:
            half = length // 2
            k = np.arange(half)
            w = np.exp(sign * 2j * np.pi * k / length)

            for i in range(0, N, length):
                u = res[i : i + half].copy()
                v = res[i + half : i + length] * w
                res[i : i + half] = u + v
                res[i + half : i + length] = u - v

            length <<= 1

        if invert:
            res /= N

        return res

    def transform(self, x):
        """Forward FFT. Same contract as DFTAnalyzer.transform."""
        x_arr = np.asarray(x, dtype=np.complex128)
        return self._fft_core(x_arr, invert=False)

    def inverse(self, spectrum):
        """Inverse FFT, including the 1/N factor."""
        x_arr = np.asarray(spectrum, dtype=np.complex128)
        return self._fft_core(x_arr, invert=True)


# ---------------------------------------------------------------------------
# BONUS (optional) -- arbitrary-length FFT.
#
# Delete this class if you are not attempting the bonus. If you do attempt it,
# run both tasks with --engine arbitrary and leave those output directories in
# your submission as the evidence.
# ---------------------------------------------------------------------------
class ArbitraryLengthFFT(FFTTransformer):
    """
    Bonus: an O(N log N) transform for ANY length N, not just powers of two.

    Bluestein's chirp-z algorithm is the usual route: rewrite the DFT as a
    convolution of two chirp sequences, and evaluate that convolution with a
    radix-2 FFT of length >= 2N-1. A mixed-radix Cooley-Tukey that factorises
    N is equally acceptable.

    With this engine, Task A no longer has to pad the digit arrays up to a
    power of two, and Task B no longer has to pad the image up to one.
    """

    name = "arbitrary"

    def transform(self, x):
        # TODO (bonus): implement this method
        raise NotImplementedError("Bonus: implement ArbitraryLengthFFT.transform")

    def inverse(self, spectrum):
        # TODO (bonus): implement this method
        raise NotImplementedError("Bonus: implement ArbitraryLengthFFT.inverse")