"""Einschrittverfahren: explizites Euler, RK4, adaptives RK2(3).

Alle Verfahren lösen y'(t) = f(y, t), y(t0) = y0 (y darf ein Vektor sein).
"""
import numpy as np


def explicit_euler(f, y0, t0, tend, n_steps):
    """Expliziter Euler mit n_steps gleichgroßen Schritten, Rückgabe (t, y)."""
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    t = np.linspace(t0, tend, n_steps + 1)
    h = (tend - t0) / n_steps
    y = np.zeros((n_steps + 1, y0.size))
    y[0] = y0
    for n in range(n_steps):
        y[n + 1] = y[n] + h * f(y[n], t[n])          # Gl. (2)
    return t, y


def rk4(f, y0, t0, tend, n_steps):
    """Klassisches Runge-Kutta 4. Ordnung (Butcher-Schema (3))."""
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    t = np.linspace(t0, tend, n_steps + 1)
    h = (tend - t0) / n_steps
    y = np.zeros((n_steps + 1, y0.size))
    y[0] = y0
    for n in range(n_steps):
        tn, yn = t[n], y[n]
        k1 = f(yn, tn)
        k2 = f(yn + h / 2 * k1, tn + h / 2)
        k3 = f(yn + h / 2 * k2, tn + h / 2)
        k4 = f(yn + h * k3, tn + h)
        y[n + 1] = yn + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t, y


def rk23_adaptive(f, y0, t0, tend, h0, tol, lam=0.5, max_steps=10**6):
    c = np.array([0.0, 1.0, 0.5])
    A = np.array([[0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.25, 0.25, 0.0]])
    b = np.array([1 / 6, 1 / 6, 4 / 6])      # Ordnung 3
    b_hat = np.array([0.5, 0.5, 0.0])        # Ordnung 2

    y_cur = np.atleast_1d(np.asarray(y0, dtype=float))
    t_cur = float(t0)
    h = float(h0)
    ts, ys, hs = [t_cur], [y_cur.copy()], []
    steps = 0
    while tend > t_cur:
        h = min(h, tend - t_cur)             # genau auf tend landen
        while True:                          # Schritt wiederholen bis eps < tol
            k = np.zeros((3, y_cur.size))
            for i in range(3):
                k[i] = f(y_cur + h * (A[i, :i] @ k[:i]), t_cur + c[i] * h)
            y_new = y_cur + h * (b @ k)
            y_hat = y_cur + h * (b_hat @ k)
            eps = np.max(np.abs(y_new - y_hat))
            # Vorschlag für die nächste Schrittweite (eps=0 -> maximal vergrößern)
            factor = (lam * tol / eps) ** (1 / 3) if eps > 0 else 10.0
            if eps < tol:
                break
            h *= factor                      # Schritt abgelehnt, h verkleinern
        t_cur += h
        y_cur = y_new
        ts.append(t_cur); ys.append(y_cur.copy()); hs.append(h)
        h *= factor                         
        steps += 1
        if steps > max_steps:
            raise RuntimeError("zu viele Schritte")
    return np.array(ts), np.array(ys), np.array(hs)
