"""
===============================================================================
2D PROPERTY 13: 2D PARSEVAL'S / PLANCHEREL'S THEOREM
===============================================================================
Mathematical Formula:
    \iint |I(x, y)|^2 dx dy = \iint |F(u, v)|^2 du dv
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

class CFT2D:
    def __init__(self, I, x, y, u, v):
        self.I = I; self.x = x; self.y = y; self.u = u; self.v = v

    def compute_cft(self):
        cos_ux = np.cos(2 * np.pi * self.u[None, :] * self.x[:, None])
        sin_ux = np.sin(2 * np.pi * self.u[None, :] * self.x[:, None])
        trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        A = trap(self.I[:, :, None] * cos_ux[None, :, :], x=self.x, axis=1)
        B = trap(self.I[:, :, None] * sin_ux[None, :, :], x=self.x, axis=1)
        cos_vy = np.cos(2 * np.pi * self.v[:, None] * self.y[None, :])
        sin_vy = np.sin(2 * np.pi * self.v[:, None] * self.y[None, :])
        term1 = trap(A[:, None, :] * cos_vy.T[:, :, None], x=self.y, axis=0)
        term2 = trap(B[:, None, :] * sin_vy.T[:, :, None], x=self.y, axis=0)
        term3 = trap(A[:, None, :] * sin_vy.T[:, :, None], x=self.y, axis=0)
        term4 = trap(B[:, None, :] * cos_vy.T[:, :, None], x=self.y, axis=0)
        return (term1 - term2) + 1j * (-(term3 + term4))

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

u = np.linspace(-3, 3, 80)
v = np.linspace(-3, 3, 80)

I = np.exp(-2.0 * (X**2 + Y**2))
F_uv = CFT2D(I, x, y, u, v).compute_cft()

trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# Energy in Spatial Domain
energy_space = trap(trap(np.abs(I)**2, x, axis=1), y, axis=0)
# Energy in Frequency Domain
energy_freq = trap(trap(np.abs(F_uv)**2, u, axis=1), v, axis=0)

diff = np.abs(energy_space - energy_freq)

print(f"[13. 2D Parseval] Energy Space Domain : {energy_space:.6f} J")
print(f"[13. 2D Parseval] Energy Freq Domain  : {energy_freq:.6f} J")
print(f"[13. 2D Parseval] Absolute Difference : {diff:.6e} J")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(np.abs(I)**2, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='viridis')
axs[0].set_title(r'Spatial Energy Density $|I(x, y)|^2$')
axs[1].imshow(np.abs(F_uv)**2, extent=[u[0], u[-1], v[0], v[-1]], origin='lower', cmap='inferno')
axs[1].set_title(r'Spectral Energy Density $|F(u, v)|^2$')
plt.tight_layout(); plt.show()
