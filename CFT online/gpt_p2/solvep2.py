import numpy as np

class FourierEpicycles:
    def __init__(self, t, signal, n_harmonics):
        """
        Step 1: Store the sampled signal and set up everything the other
        methods will need.

        Parameters
        ----------
        t : 1D numpy array, shape (M,)
            Uniformly spaced sample times covering ONE FULL PERIOD of the
            signal, as a *closed* interval: t[0] == 0 and t[-1] == T (the
            period). This is exactly what svg_utils.load_svg_path(...)
            returns.
        signal : 1D complex numpy array, shape (M,)
            signal[i] = f(t[i]) = x(t[i]) + 1j * y(t[i]). Periodic, so
            signal[-1] == signal[0].
        n_harmonics : int (call it N)
            The series will use every integer harmonic n with
            -N <= n <= N (i.e. 2N+1 terms in total -- do not forget the
            negative harmonics).

        You must set at least the following attributes, since the rest of
        this class (and the provided plotting/animation code) expects
        them to exist:
            self.t, self.signal, self.N
            self.T      -- the period (a float)
            self.omega  -- the fundamental angular frequency, 2*pi/T
            self.coeffs -- an (initially empty) dict that will map
                           n -> c_n once calculate_all_coefficients() has
                           been called
        """
        # TODO: implement this method
        self.t = np.asarray(t)
        self.signal = np.asarray(signal)
        self.N = int(n_harmonics)
        self.T = float(self.t[-1] - self.t[0])
        self.omega = 2 * np.pi / self.T
        self.coeffs = {}
        # raise NotImplementedError("Implement __init__")


    def calculate_cn(self, n):
        """
        Step 2: Compute a single complex Fourier coefficient c_n using
        numerical integration (np.trapezoid) over the stored samples
        self.t, self.signal.

            c_n = (1/T) * integral_0^T  f(t) * exp(-j*n*omega*t)  dt

        n may be zero, positive, or negative.
        """
        # TODO: implement this method
        integrand = self.signal * np.exp(-1j * n * self.omega * self.t)
        integral = np.trapezoid(integrand, self.t)
        return (1 / self.T) * integral
        # raise NotImplementedError("Implement calculate_cn")

    def calculate_all_coefficients(self):
        """
        Step 3: Populate self.coeffs with c_n for every harmonic
        n = -N, ..., -1, 0, 1, ..., N by repeatedly calling calculate_cn(n).
        """
        # TODO: implement this method
        self.coeffs = {n : self.calculate_cn(n) for n in range(-self.N, self.N + 1)}
        # raise NotImplementedError("Implement calculate_all_coefficients")

    def approximate(self, t):
        """
        Step 4: Reconstruct (an approximation of) the signal at time(s) t
        from the coefficients already stored in self.coeffs:

            f_hat(t) = sum_{n=-N}^{N} c_n * exp(j*n*omega*t)

        t may be a single number or a numpy array of times -- your
        implementation must support both, since the provided
        plotting/animation code calls this both ways.
        """
        # TODO: implement this method
        t = np.asarray(t)
        f = np.zeros_like(t, dtype = complex)
        for n, c_n in self.coeffs.items():
            f += c_n * np.exp(1j * n * self.omega * t)
        if np.ndim(t) == 0:
            return f.item()
        return f
        # raise NotImplementedError("Implement approximate")

t = np.linspace(-2,2,1000)
N = {3,7,25,50}
x = np.where(np.abs(t) <= 1.0, 1.0 - np.abs(t), 0.0)



for i in N:
    n_harmonics = np.linspace(-i,i+1,2 * i)
    sg = FourierEpicycles(t,x,i)
    sg.calculate_all_coefficients()
    approx = sg.approximate(t)
    mse_mag = np.mean((np.abs(x) - np.abs(approx)) ** 2)
    print(mse_mag)