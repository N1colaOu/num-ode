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
