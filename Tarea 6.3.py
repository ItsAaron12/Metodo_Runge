import numpy as np
import matplotlib.pyplot as plt

# Sistema de EDOs:
# dy1/dt = y2
# dy2/dt = -2*y2 - 5*y1

def f(t, y):
    y1, y2 = y
    dy1_dt = y2
    dy2_dt = -2 * y2 - 5 * y1
    return np.array([dy1_dt, dy2_dt])

# Método de Runge-Kutta para sistemas de ecuaciones
def runge_kutta_4_system(f, t0, y0, t_end, h):
    t_vals = [t0]
    y1_vals = [y0[0]]
    y2_vals = [y0[1]]
    
    t = t0
    y = np.array(y0)

    print(f"{'t':>6} {'y1 (posición)':>15} {'y2 (velocidad)':>17}")
    print(f"{t:6.2f} {y[0]:15.6f} {y[1]:17.6f}")

    while t < t_end:
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)

        y = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t = t + h

        t_vals.append(t)
        y1_vals.append(y[0])
        y2_vals.append(y[1])

        print(f"{t:6.2f} {y[0]:15.6f} {y[1]:17.6f}")

    return t_vals, y1_vals, y2_vals

# Condiciones iniciales
t0 = 0
t_end = 5
h = 0.1
y0 = [1, 0]  # y1(0) = 1, y2(0) = 0

# Resolver el sistema
t_vals, y1_vals, y2_vals = runge_kutta_4_system(f, t0, y0, t_end, h)

# Graficar la posición y1(t)
plt.figure(figsize=(8, 5))
plt.plot(t_vals, y1_vals, 'bo-', label='Posición (y1)')
plt.xlabel("Tiempo t (s)")
plt.ylabel("Posición de la masa")
plt.title("Dinámica de un resorte amortiguado")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("resorte_amortiguado.png")
plt.show()
