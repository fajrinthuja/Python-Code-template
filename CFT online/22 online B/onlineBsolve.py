import numpy as np
import matplotlib.pyplot as plt

class SignalGenerator:
    def generate(a,t):
        ex = np.exp(-a * t ** 2)
        return ex
    def shift(signal,t, to):
        t = t - to
        y = np.exp(-a * t ** 2)
        return y

class CFTAnalyser:
    def compute_cft(signal, freq, time):
        cft = np.zeros(len(freq), dtype=complex)
        for id, i in enumerate(freq):
            ex = np.exp(-1j * 2 * np.pi * i * time)
            cft[id] = np.trapezoid(signal * ex, time)
        return cft

t = np.linspace(-5,5,2000)
sg = SignalGenerator 
a = 1
x = sg.generate(a,t)
y = sg.shift(x,t,1)

ca = CFTAnalyser

f = np.linspace(-10,10,1000)
xf = ca.compute_cft(x,f,t)

yf = ca.compute_cft(y,f,t)

y_theory = xf * np.exp(-1j * 2 * np.pi * f * 1)

def MSE_mag(x,y):
    n = len(x)
    sum = 0
    for i in range(0,len(x)):
        sum += (x[i] - y[i]) ** 2
    sum /= n
    return sum

# def MSE_phase(x,y):

fig, axes = plt.subplots(4,1,figsize = (8,4))
axes[0].plot(x)
axes[1].plot(xf, color = 'b')
axes[2].plot(yf, color = 'r')
axes[3].plot(y_theory, color = 'g')

fig.tight_layout()
plt.show()

print(MSE_mag(yf,y_theory))
    