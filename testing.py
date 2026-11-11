import rk23 as rk23
import numpy as np
import matplotlib.pyplot as plt

t_curr = 0.0
t_end = 1.0
y_curr = [1.0]
h_curr = 0.1
eps_tol = 0.3
lam = 0.5

def f(t, y):
    return -10*y

y_data = [y_curr]
t_data = [t_curr]
while t_curr < t_end:
    if t_curr + h_curr > t_end:
        h_curr = t_end - t_curr
    h_new, y_new, success = rk23.iterate(t_curr, h_curr, y_curr, eps_tol, f, lam)
    if success:
        y_curr = y_new
        t_curr += h_curr
        y_data.append(y_curr)
        t_data.append(t_curr)
    h_curr = h_new


t_anal = np.linspace(0, 1, 100)
y_anal = np.exp(-10 * t_anal)

plt.figure(figsize=(8, 6)) 
plt.plot(t_data, y_data, marker='o', linestyle='-', label='RK23 (adaptive)')
plt.plot(t_anal, y_anal, 'k--', label='analytical')
plt.title('Task 1, adaptive, tol=0.3, safety=0.5')
plt.legend()
plt.show()



