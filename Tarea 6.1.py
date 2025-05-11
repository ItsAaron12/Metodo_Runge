import numpy as np
import matplotlib.pyplot as plt

# EDO: dT/dx = -0.25 * (T - 25)
def f(x, T):
    return -0.25 * (T - 25)

# Solución exacta
def T_exacta(x):
    return 25 + 75 * np.exp(-0.25 * x)

# Método de Runge-Kutta de cuarto orden
def runge_kutta_4(f, x0, T0, x_end, h):
    x_vals = [x0]
    T_vals = [T0]
    T_exact_vals = [T_exacta(x0)]

    x = x0
    T = T0

    print(f"{'x':>6} {'T_aprox':>15} {'T_exacta':>15}")
    print(f"{x:6.2f} {T:15.6f} {T_exacta(x):15.6f}")

    while x < x_end:
        k1 = f(x, T)
        k2 = f(x + h/2, T + h/2 * k1)
        k3 = f(x + h/2, T + h/2 * k2)
        k4 = f(x + h, T + h * k3)

        T += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h

        x_vals.append(x)
        T_vals.append(T)
        T_exact_vals.append(T_exacta(x))

        print(f"{x:6.2f} {T:15.6f} {T_exacta(x):15.6f}")

    return x_vals, T_vals, T_exact_vals

# Parámetros iniciales
x0 = 0
T0 = 100
x_end = 2
h = 0.1

# Llamada al método de Runge-Kutta
x_vals, T_aprox, T_exact_vals = runge_kutta_4(f, x0, T0, x_end, h)

# Gráfica comparativa
plt.figure(figsize=(8, 5))
plt.plot(x_vals, T_aprox, 'bo-', label='RK4 (Aproximada)')
plt.plot(x_vals, T_exact_vals, 'r--', label='Solución Exacta')
plt.xlabel("Distancia x (m)")
plt.ylabel("Temperatura T (°C)")
plt.title("Transferencia de Calor en un Tubo")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("transferencia_calor_tubo.png")
plt.show()