import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameter Model
N = 1000           # Total populasi
I0 = 1             # Kasus infeksi awal
R0 = 0             # Kasus pulih awal
S0 = N - I0 - R0   # Individu yang rentan
beta = 0.3         # Laju transmisi infeksi (contoh)
gamma = 0.1        # Laju pemulihan (contoh)
days = 160         # Durasi hari

# Model SIR
def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Kondisi Awal
y0 = S0, I0, R0

# Rentang Waktu
t = np.linspace(0, days, days)

# Solusi ODE
result = odeint(sir_model, y0, t, args=(N, beta, gamma))
S, I, R = result.T

# Plot Hasil
plt.figure(figsize=(10,6))
plt.plot(t, S, 'b', label='Rentan')
plt.plot(t, I, 'r', label='Terinfeksi')
plt.plot(t, R, 'g', label='Pulih')
plt.xlabel('Waktu (hari)')
plt.ylabel('Jumlah Individu')
plt.legend()
plt.title("Pemodelan Penyebaran COVID-19 (Model SIR)")
plt.grid(True)
plt.show()
 