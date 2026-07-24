import os
import matplotlib.pyplot as plt
# from signallti import DiscreteSignal, LTISystem


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



def parse_signal_file(filename):
    """
    Parses signal file containing:
    Line 1: start_time end_time
    Line 2: space-separated values
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    start_time, end_time = map(int, lines[0].split())
    values = list(map(float, lines[1].split()))
    
    sig = DiscreteSignal(start_time, end_time)
    for idx, t in enumerate(range(start_time, end_time + 1)):
        sig.set_value_at_time(t, values[idx])
        
    return sig


def compute_impulse_response(step_response):
    """
    Computes impulse response h[n] from step response s[n]:
    h[n] = s[n] - s[n-1]
    """
    # s[n-1] is shifted by +1
    s_shifted = step_response.shift(1)
    
    # h[n] = s[n] + (-1)*s[n-1]
    h = step_response.add(s_shifted.multiply(-1.0))
    return h


def main():
    # 1. Read input signal x[n] and step response s[n] from files
    input_file = "D:\2-2\Python Code Template\Python-Code-template\Convolution Online\Conv_Online_A1_A2\input_signal.txt"
    step_file = "D:\2-2\Python Code Template\Python-Code-template\Convolution Online\Conv_Online_A1_A2\step_response.txt"
    
    x = parse_signal_file(input_file)
    s = parse_signal_file(step_file)
    
    # 2. Derive Impulse Response h[n] from Step Response s[n]
    h = compute_impulse_response(s)
    
    # 3. Create the LTI System using impulse response h[n]
    system = LTISystem(h)
    
    # 4. Compute output y[n] = x[n] * h[n]
    x = compute_impulse_response(x)
    y = system.output(x)
    
    # 5. Plot and save results
    os.makedirs("output_plots", exist_ok=True)
    
    x.plot("Input Signal x[n]", save_path="output_plots/input_signal.png")
    plt.close()
    
    s.plot("Step Response s[n]", save_path="output_plots/step_response.png")
    plt.close()
    
    h.plot("Impulse Response h[n]", save_path="output_plots/impulse_response.png")
    plt.close()
    
    y.plot("Output Signal y[n]", save_path="output_plots/output_signal.png")
    plt.close()

    # 6. Print Non-zero samples of Output
    print("Non-zero samples of Output y[n]:")
    for t, val in y.nonzero_samples():
        print(f"y[{t}] = {val:.4f}")


if __name__ == "__main__":
    main()