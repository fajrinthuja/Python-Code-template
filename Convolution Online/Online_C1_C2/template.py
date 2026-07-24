import numpy as np
import matplotlib.pyplot as plt

# Todo: Define Signal class


def readable_time_ticks(time_values, max_labels=18):
    if len(time_values) <= max_labels:
        return time_values

    step = int(np.ceil(len(time_values) / max_labels))
    ticks = time_values[::step]

    if ticks[-1] != time_values[-1]:
        ticks.append(time_values[-1])

    return ticks


class DiscreteSignal:
    """Finite discrete-time signal with integer indices."""

    # Create a finite discrete-time signal over the given integer range.
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        rng = end_time - start_time + 1
        if rng <= 0:
            raise ValueError("Array should contain at least one value")
        self.values = np.zeros(rng, dtype = float)
        # raise NotImplementedError("Complete the DiscreteSignal constructor")

    # Return the number of stored samples in the signal.
    def __len__(self):
        return len(self.values)
        # raise NotImplementedError("Complete __len__")

    # Return the integer time indices covered by the signal.
    def times(self):
        return np.arange(self.start_time, self.end_time + 1)
        # raise NotImplementedError("Complete times")

    # Return the signal value at the given time index.
    def get_value_at_time(self, t):
        if self.start_time <= t and self.end_time >= t:
            return self.values[t - self.start_time]
        return 0
        # raise NotImplementedError("Complete get_value_at_time")

    # Set the signal value at the given time index.
    def set_value_at_time(self, t, value):
        if self.start_time <= t and self.end_time >= t:
            self.values[t - self.start_time] = float(value)
        else:
            raise IndexError("out of bound")
        # raise NotImplementedError("Complete set_value_at_time")

    # Return a shifted copy of the signal.
    def shift(self, k):
        shifted = DiscreteSignal(self.start_time + k, self.end_time + k)
        for i in range (self.start_time, self.end_time + 1):
            shifted.set_value_at_time(i + k, self.get_value_at_time(i))
        return shifted
        # raise NotImplementedError("Complete shift")

    # Return the sum of this signal and another signal.
    def add(self, other):
        start = min(self.start_time,other.start_time)
        end = max(self.end_time, other.end_time)
        added = DiscreteSignal(start, end)
        for i in range(start, end + 1):
            now = self.get_value_at_time(i) + other.get_value_at_time(i)
            added.set_value_at_time(i,now)
        return added
        # raise NotImplementedError("Complete add")

    # Return a scaled copy of the signal.
    def multiply(self, scalar):
        scaled = DiscreteSignal(self.start_time, self.end_time)
        scaled.values = self.values * scalar
        return scaled
        # raise NotImplementedError("Complete multiply")

    # Return the nonzero samples of the signal.
    def nonzero_samples(self, tolerance=1e-12):
        samples = []
        for i in range(self.start_time, self.end_time + 1):
            if abs(self.values[i - self.start_time]) >= tolerance:
                samples.append((i,self.values[i - self.start_time]))
        return samples
        # raise NotImplementedError("Complete nonzero_samples")

    def plot(self, title, save_path=None, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        time_values = list(self.times())
        markerline, stemlines, baseline = ax.stem(time_values, self.values)
        markerline.set_markersize(6)
        baseline.set_color("black")
        baseline.set_linewidth(1)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.35)
        ax.set_xticks(readable_time_ticks(time_values))
        ax.tick_params(axis="x", labelsize=9)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        return ax
class SuperSignal:
    def __init__(self):
        self.components = []

    def add(self, signal: DiscreteSignal, coefficient=1.0):
        self.components.append((coefficient, signal))

class LTISystem:
    """Discrete-time LTI system described by a finite impulse response."""

    # Store the impulse response that defines the LTI system.
    def __init__(self, impulse_response:DiscreteSignal):
        self.h = impulse_response
        # raise NotImplementedError("Complete the LTISystem constructor")

    # Return the output time range for the convolution result.
    def output_range(self, input_signal):
        start = self.h.start_time + input_signal.start_time
        end = self.h.end_time + input_signal.end_time
        return (start, end)
        # raise NotImplementedError("Complete output_range")

    # Return all shifted and scaled impulse-response components for the input.
    def get_response_components(self, input_signal : DiscreteSignal):
        comp = []
        for k, x_k in input_signal.nonzero_samples():
            now = self.h.shift(k);
            now = now.multiply(x_k)
            comp.append(now)
        return comp
        # raise NotImplementedError("Complete get_response_components")

    # Return the system output using superposition of response components.
    def output_by_superposition(self, input_signal: DiscreteSignal):
        rng = self.output_range(input_signal)
        start = rng[0]
        end = rng[1]
        output_signal = DiscreteSignal(start,end)
        comp = self.get_response_components(input_signal)
        for c in comp:
            output_signal = output_signal.add(c)
        return output_signal
        # raise NotImplementedError("Complete output_by_superposition")

    # Return the nonzero product terms that contribute to one output sample.
    def get_contributions_at_time(self, input_signal : DiscreteSignal, n):
        comp = []
        for k, x_k in input_signal.nonzero_samples():
            now = self.h.get_value_at_time(n - k)
            now = now * x_k
            if abs(now) > 1e-12:
                comp.append((k,x_k,self.h.get_value_at_time(n-k),now))
        return comp
        # raise NotImplementedError("Complete get_contributions_at_time")

    # Return one output sample of the LTI system.
    def output_at_time(self, input_signal, n):
        comp = self.get_contributions_at_time(input_signal, n)
        ans = 0
        for i in comp:
            ans += i[3]
        return ans
        # raise NotImplementedError("Complete output_at_time")

    # Return the complete output signal of the LTI system.
    def output(self, input_signal):
        rng = self.output_range(input_signal)
        start = rng[0]
        end = rng[1]
        output_signal = DiscreteSignal(start, end)
        for i in range(start, end + 1):
            now = self.output_at_time(input_signal,i)
            output_signal.set_value_at_time(i,now)
        return output_signal
        # raise NotImplementedError("Complete output")
    def output_super(self, input_superSignal : SuperSignal):
        start = 1e9
        end = -1e9
        for i in input_superSignal.components:
            rng = self.output_range(i[1])
            st = rng[0]
            en = rng[1]
            start = min(start, st)
            end = max(en, end)
        out = DiscreteSignal(start, end)
        for i in input_superSignal.components:
            now = self.output(i[1])
            out = out.add(now.multiply(i[0]))
        return out

        

        
# Todo: Define LTI class

if __name__ == "__main__":
    INF = 10

    # Component signals
    x1 = DiscreteSignal(-INF,INF)
    x1.set_value_at_time(0, 1)

    x2 = DiscreteSignal(-INF,INF)
    x2.set_value_at_time(2, 1)

    # Todo: Create SuperSignal: x(n) = 2*x1(n) - x2(n)
    x = SuperSignal()
    x.add(x1)
    x.add(x2)

    # Impulse response
    h = DiscreteSignal(-INF,INF)
    h.set_value_at_time(0, 1)
    h.set_value_at_time(1, 0.5)

    system = LTISystem(h)

    # Todo: Output using superposition

    out = system.output_super(x)
    plt = out.plot("thuja")
    # plt.show()
    
