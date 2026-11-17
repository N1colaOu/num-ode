"""Aufgabe a): y' = -10 y, y(0) = 1, t in [0, 1]. Analytisch: y(t) = exp(-10 t)."""
import numpy as np
import matplotlib.pyplot as plt
from solvers import explicit_euler, rk4, rk23_adaptive

f = lambda y, t: -10.0 * y
y0, t0, t_end = 1.0, 0.0, 1.0
exact = lambda t: np.exp(-10.0 * t)
t_fine = np.linspace(t0, t_end, 1000)

#plot 1/2: of explicit euler and rk4 for  h = 0.1 and h = 0.01
for h in (0.1, 0.01):
    n = int(round((t_end -t0) / h))
    t_e, y_e = explicit_euler(f, y0, t0, t_end, n)
    t_r, y_r = rk4(f, y0, t0, t_end, n)
    plt.figure(figsize=(6, 4.5))
    plt.plot(t_e, y_e[:, 0], "o-", ms=4, label="expl. Euler")
    plt.plot(t_r, y_r[:, 0], "o-", ms=4, label="RK4")
    plt.plot(t_fine, exact(t_fine), "k--", label="analytical")
    plt.title(f"Task 1, h = {h}")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"a_euler_rk4_h{h}.png", dpi=150)

#plot 3/4: adaptive rk23 for etol = 0.3, 0.01 and lambda = 0.5
for tol in (0.3, 0.01):
    t_a, y_a, h_a = rk23_adaptive(f, y0, t0, t_end, h0=0.1, tol=tol, safety=0.5)
    plt.figure(figsize=(6, 4.5))
    plt.plot(t_a, y_a[:, 0], "o-", ms=4, label="adaptive RK23")
    plt.plot(t_fine, exact(t_fine), "k--", label="analytical")
    plt.title(f"Task 1, adaptive, tol={tol}, safety= 0.5")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"a_rk23_tol{tol}.png", dpi=150)

plt.show()


    