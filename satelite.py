"""Aufgabe b): Arenstorf-Orbit (eingeschränktes Dreikörperproblem)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from solvers import explicit_euler, rk4, rk23_adaptive

mu = 0.012277471
mu_hat = 1.0 - mu
T = 17.06521656015796


def f(u, t):
    """System 1. Ordnung, u = (x1, x2, v1, v2) mit v = x'."""
    x1, x2, v1, v2 = u
    N1 = ((x1 + mu) ** 2 + x2 ** 2) ** 1.5
    N2 = ((x1 - mu_hat) ** 2 + x2 ** 2) ** 1.5
    a1 = x1 + 2 * v2 - mu_hat * (x1 + mu) / N1 - mu * (x1 - mu_hat) / N2
    a2 = x2 - 2 * v1 - mu_hat * x2 / N1 - mu * x2 / N2
    return np.array([v1, v2, a1, a2])


u0 = np.array([0.994, 0.0, 0.0, -2.001585106376082])

# --- RK23 adaptiv ---
t, u, h = rk23_adaptive(f, u0, 0.0, T, h0=0.1, tol=1e-3, lam=0.5)
n_rk23 = len(h)
err_rk23 = np.max(np.abs(u[-1, :2] - u0[:2]))
print(f"RK23: {n_rk23} Schritte, |x(T)-x(0)|_inf = {err_rk23:.3e}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot(u[:, 0], u[:, 1], label="RK23 (adaptive)")
ax[0].set_xlabel("x1"); ax[0].set_ylabel("x2"); ax[0].legend()
ax[1].plot(h, label="time step size")
ax[1].set_xlabel("step number"); ax[1].set_ylabel("dt"); ax[1].legend()
plt.tight_layout(); plt.savefig("b_rk23_orbit.png", dpi=150); plt.close()


# --- Wie viele Schritte brauchen RK4 / Euler für vergleichbare Genauigkeit? ---
# Maß: Schließungsfehler der Bahn |x(T) - x(0)|_inf (die Bahn ist periodisch).
def closure_error(method, n):
    _, uu = method(f, u0, 0.0, T, n)
    return np.max(np.abs(uu[-1, :2] - u0[:2]))


def min_steps(method, target, n_start=100, n_max=5_000_000):
    n = n_start
    while n <= n_max:
        if closure_error(method, n) <= target:
            return n
        n *= 2
    return None


for name, method in (("RK4", rk4), ("expl. Euler", explicit_euler)):
    n = min_steps(method, err_rk23)
    print(f"{name}: ca. {n} Schritte nötig (Fehler <= {err_rk23:.2e})")

# Plots für je ein paar Schrittzahlen
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
for a, (name, method, ns) in zip(
        ax, (("expl. Euler", explicit_euler, (1000, 10000, 100000)),
             ("RK4", rk4, (200, 500, 1000)))):
    for n in ns:
        _, uu = method(f, u0, 0.0, T, n)
        a.plot(uu[:, 0], uu[:, 1], label=f"n={n}")
    a.plot(u[:, 0], u[:, 1], "k--", lw=0.8, label="RK23")
    a.set_title(name); a.set_xlabel("x1"); a.legend()
ax[0].set_ylabel("x2")
plt.tight_layout(); plt.savefig("b_euler_rk4_orbit.png", dpi=150); plt.close()
