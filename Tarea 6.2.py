import numpy as np
import matplotlib.pyplot as plt

# Parámetros del circuito
V = 10          # voltios
R = 1000        # ohmios
C = 0.001       # faradios

# EDO: dq/dt = (V - q/C) / R
def f(t, q):
    return (V - q / C) / R

# Método de Runge-Kutta de cuarto orden
def runge_kutta_4(f, t0, q0, t_end, h):
    t_vals = [t0]
    q_vals = [q0]

    t = t0
    q = q0

    print(f"{'t':>6} {'q(t)':>15}")
    print(f"{t:6.2f} {q:15.6f}")

    while t < t_end:
        k1 = f(t, q)
        k2 = f(t + h/2, q + h/2 * k1)
        k3 = f(t + h/2, q + h/2 * k2)
        k4 = f(t + h, q + h * k3)

        q += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h

        t_vals.append(t)
        q_vals.append(q)

        print(f"{t:6.2f} {q:15.6f}")

    return t_vals, q_vals

# Condiciones iniciales
t0 = 0
q0 = 0
t_end = 1
h = 0.05

# Resolver la EDO
t_vals, q_vals = runge_kutta_4(f, t0, q0, t_end, h)

# Graficar q(t)
plt.figure(figsize=(8, 5))
plt.plot(t_vals, q_vals, 'bo-', label='q(t) - Carga del capacitor')
plt.xlabel("Tiempo t (s)")
plt.ylabel("Carga q(t) (C)")
plt.title("Carga del capacitor en un circuito RC")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("carga_capacitor.png")
plt.show()
