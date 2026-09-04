"""
bigmul.py  --  TASK A: multiplying huge integers with your own transform.

YOUR CODE GOES HERE. The provided modules io_utils.py (file reading, random
operands) and bench_utils.py (timing, runtime plots) are already imported;
everything mathematical is yours.

Usage (the command line is already wired up for you):

    python3 bigmul.py inputs/1.txt --engine fft --out-dir outputs/1
    python3 bigmul.py inputs/3.txt --engine dft --out-dir outputs/3
    python3 bigmul.py --benchmark --out-dir outputs/benchmark

Restrictions: no numpy.fft / scipy.fft / numpy.convolve / scipy.signal, and
no Python big-integer multiplication of the operands themselves. Python's
own integers may be used ONLY to check your answer at the end.
"""

import argparse
import os
import sys

import numpy as np

from bench_utils import plot_runtime_curve, time_best, timing_table_lines
from io_utils import random_decimal, read_operands, write_report, write_text
from transforms import DFTAnalyzer, FFTTransformer, next_power_of_two

# Python 3.11+ refuses to print integers longer than 4300 digits unless this
# limit is raised, and the verification step below prints one.
sys.set_int_max_str_digits(2_000_000)

# Number of decimal digits packed into one "limb" (one polynomial
# coefficient). The specification explains why 4 is a safe choice and what
# breaks if you raise it too far.
BASE_DIGITS = 4
BASE = 10 ** BASE_DIGITS


def to_limbs(text, base_digits=BASE_DIGITS):
    """
    Convert a decimal string into polynomial coefficients.

    "123456789" with base_digits = 4 becomes the little-endian limb array
    [6789, 2345, 1] -- that is, 1*BASE^2 + 2345*BASE^1 + 6789*BASE^0.

    Parameters
    ----------
    text : str
        A decimal integer, possibly with a leading '+' or '-'.
    base_digits : int
        Decimal digits per limb.

    Returns
    -------
    (int, numpy.ndarray)
        The sign (+1 or -1) and the little-endian limb array (dtype int64).
        Handle the sign separately from the magnitude: the transform never
        sees it.
    """
    text = text.strip()
    sign = 1
    if text.startswith("-"):
        sign = -1
        text = text[1:].lstrip("0")
    elif text.startswith("+"):
        text = text[1:].lstrip("0")
    else:
        text = text.lstrip("0")

    if not text:
        return 1, np.zeros(1, dtype=np.int64)

    rev = text[::-1]
    n = len(rev)
    limbs = []
    
    for i in range(0, n, base_digits):
        chunk = rev[i : i + base_digits]
        limbs.append(int(chunk[::-1]))

    return sign, np.array(limbs, dtype=np.int64)


def from_limbs(sign, limbs, base_digits=BASE_DIGITS):
    """
    Convert limbs back into a decimal string, propagating carries.

    The limbs handed to this function are the convolution result, so they are
    NOT yet reduced: an entry may be far larger than BASE. Sweep from the
    least significant limb upwards, carrying the overflow into the next one,
    then strip leading zeros and re-attach the sign.

    Returns
    -------
    str
        The decimal representation. "0" must come out as "0", not "-0" or "".
    """
    base = 10 ** base_digits
    carried = []
    carry = 0

    for x in limbs:
        total = int(x) + carry
        carry = total // base
        carried.append(total % base)

    while carry > 0:
        carried.append(carry % base)
        carry //= base

    while len(carried) > 1 and carried[-1] == 0:
        carried.pop()

    if len(carried) == 1 and carried[0] == 0:
        return "0"
    parts = [str(carried[-1])]
    fmt = f"{{:0{base_digits}d}}"
    for val in reversed(carried[:-1]):
        parts.append(fmt.format(val))
    result = "".join(parts)
    return f"-{result}" if sign < 0 else result


def multiply_transform(a, b, engine):
    """
    Multiply two limb arrays through the frequency domain.

    The product of two polynomials is the LINEAR convolution of their
    coefficients, and the transform gives you convolution as a pointwise
    product -- but a DFT of length N gives you CIRCULAR convolution of period
    N. Choose N large enough that the linear result fits, or the high-order
    coefficients wrap around and silently corrupt the answer.

    Steps:
      1. Choose the transform length N (see above; for FFTTransformer it must
         also be a power of two -- next_power_of_two is in transforms.py).
      2. Zero-pad both limb arrays to length N.
      3. Transform both, multiply the two spectra pointwise, inverse-transform.
      4. The result is real up to rounding error: take the real part and round
         to the nearest integer.

    Parameters
    ----------
    a, b : numpy.ndarray of int64
        Little-endian limb arrays.
    engine : DFTAnalyzer or FFTTransformer

    Returns
    -------
    (numpy.ndarray of int64, int)
        The un-carried convolution coefficients, and the transform length N
        you used (report.txt has to state it).
    """
    linear_len = len(a) + len(b) - 1
    if engine.name == "fft":
        N = next_power_of_two(linear_len)
    else:
        N = linear_len
    padded_a = np.zeros(N, dtype=np.complex128)
    padded_b = np.zeros(N, dtype=np.complex128)
    padded_a[: len(a)] = a
    padded_b[: len(b)] = b
    spec_a = engine.transform(padded_a)
    spec_b = engine.transform(padded_b)
    spec_c = spec_a * spec_b
    conv = engine.inverse(spec_c)
    res = np.round(conv.real).astype(np.int64)[:linear_len]
    return res, N


def multiply_schoolbook(a, b):
    """
    OPTIONAL baseline: the O(n^2) method everyone learns at school, on limbs.

    Only needed if you want a third curve on your runtime plot. Expect it to
    be competitive for a long time: a NumPy-assisted schoolbook multiply has a
    very small constant factor, and constant factors decide who wins at small
    sizes.
    """
    n, m = len(a), len(b)
    res = np.zeros(n + m - 1, dtype=np.int64)
    for i in range(n):
        res[i : i + m] += a[i] * b
    return res, n + m - 1


def multiply(text_a, text_b, method):
    """
    Multiply two decimal strings and return (product_string, N, limbs_a, limbs_b).

    ``method`` is one of "dft", "fft", "schoolbook" (optional) or "arbitrary"
    (bonus). Pick the engine, convert to limbs, convolve, carry, re-sign.
    """
    sign_a, limbs_a = to_limbs(text_a, BASE_DIGITS)
    sign_b, limbs_b = to_limbs(text_b, BASE_DIGITS)
    sign = sign_a * sign_b

    if method == "dft":
        engine = DFTAnalyzer()
        raw_conv, N = multiply_transform(limbs_a, limbs_b, engine)
    elif method == "fft":
        engine = FFTTransformer()
        raw_conv, N = multiply_transform(limbs_a, limbs_b, engine)
    elif method == "schoolbook":
        raw_conv, N = multiply_schoolbook(limbs_a, limbs_b)
    else:
        raise ValueError(f"Unknown multiplication method: {method}")

    product = from_limbs(sign, raw_conv, BASE_DIGITS)
    return product, N, limbs_a, limbs_b

import sys
import math

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
    a = [9,9,9]
    b = [9,9]
    s = ""
    for i in a:
        s += str(i)
    s = s[::-1]
    t = ""
    for i in b:
        t += str(i)
    t = t[::-1]

    ans = multiply(s, t, "fft")
    print(ans)

    s = ans[0]

    n = len(s)
    answer = [0] * n


    for i in range(len(s)):
        answer[i] = int(s[n - 1 - i])

    print(answer)


def main():
    iterator = token_generator()

    t = 1
    
    # try:
    #     t = int(next(iterator))
    # except StopIteration:
    #     return
        
    for _ in range(t):
        solve(iterator)

if __name__ == '__main__':
    main()