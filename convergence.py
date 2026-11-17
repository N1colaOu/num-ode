"""Aufgabe c): Konvergenzraten.

y' = 3 exp(-4t) - 2y, y(0) = 1.
Ansatz y = a e^{-2t} + b e^{-4t}: Einsetzen liefert -4b = 3 - 2b => b = -3/2,
y(0) = a + b = 1 => a = 5/2.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from solvers import explicit_euler, rk4, rk23_adaptive

f = lambda y, t: 3 * np.exp(-4 * t) - 2 * y
exact = lambda t: 2.5 * np.exp(-2 * t) - 1.5 * np.exp(-4 * t)
y0, t0, tend = 1.0, 0.0, 1.0


def linf_error(t, y):
    return np.max(np.abs(y[:, 0] - exact(t)))


# --- feste Schrittweiten ---
ns = np.unique(np.round(np.logspace(np.log10(2), 3, 10)).astype(int))
err_euler = [linf_error(*explicit_euler(f, y0, t0, tend, n)) for n in ns]
err_rk4 = [linf_error(*rk4(f, y0, t0, tend, n)) for n in ns]

# --- adaptives RK2(3) für verschiedene Toleranzen ---
tols = np.logspace(-1, -9, 9)
n_ad, err_ad = [], []
for tol in tols:
    t, y, h = rk23_adaptive(f, y0, t0, tend, h0=0.1, tol=tol, safety=0.5)
    n_ad.append(len(h)); err_ad.append(linf_error(t, y))
n_ad = np.array(n_ad)

# Referenzgeraden (durch den ersten Punkt der jeweiligen Kurve gelegt)
nn = np.logspace(np.log10(ns[0]), 3, 50)
ref = lambda p, n_pts, e_pts: e_pts[0] * (nn / n_pts[0]) ** (-p)

plt.figure(figsize=(7, 5))
plt.loglog(ns, err_rk4, "o-", label="error RK4")
plt.loglog(ns, err_euler, "o-", label="error euler")
plt.loglog(n_ad, err_ad, "o-", label="error adaptive RK23")
plt.loglog(nn, ref(1, ns, err_euler) * 2, "k-", label="slope 1")
plt.loglog(nn, ref(3, n_ad, err_ad) * 2, "k-.", label="slope 3")
plt.loglog(nn, ref(4, ns, err_rk4), "k--", label="slope 4")
plt.xlabel("n_steps"); plt.ylabel("Linf error"); plt.legend()
plt.tight_layout(); plt.savefig("c_convergence.png", dpi=150); plt.close()

# Empirische Ordnungen (Steigung im log-log-Plot)
slope = lambda n, e: -np.polyfit(np.log(n), np.log(e), 1)[0]
print(f"Euler:  Ordnung ~ {slope(ns, err_euler):.2f}")
print(f"RK4:    Ordnung ~ {slope(ns[:-2], err_rk4[:-2]):.2f} (ohne Rundungsbereich)")
print(f"RK2(3): Ordnung ~ {slope(n_ad, err_ad):.2f} (Fehler vs. Schrittzahl)")

# --- RK2(3): Fehler über der Toleranz ---
p = np.polyfit(np.log(tols), np.log(err_ad), 1)[0]
print(f"RK2(3): Fehler ~ tol^{p:.2f}")
plt.figure(figsize=(6, 4.5))
plt.loglog(tols, err_ad, "o-", label="error adaptive RK23")
plt.loglog(tols, tols * err_ad[0] / tols[0], "k--", label="slope 1")
plt.xlabel("eps_tol"); plt.ylabel("Linf error"); plt.legend()
plt.tight_layout(); plt.savefig("c_rk23_tolerance.png", dpi=150); plt.close()
