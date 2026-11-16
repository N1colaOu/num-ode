import numpy as np

"runge-kutta 4"
def rk4(f, y0, t0, t_end, n_steps):
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    t = np.linspace(t0, t_end, n_steps + 1)
    h = (t_end - t0) / n_steps
    y = np.zeros((n_steps + 1, y0.size))
    y[0] = y0
    for n in range (n_steps):
        tn, yn = t[n], y[n]
        k1 = f(yn, tn)
        k2 = f(yn + (h / 2) * k1, tn + h / 2)
        k3 = f(yn + (h / 2) * k2, tn + h / 2)
        k4 = f(yn + h * k3, tn + h)
        y[n + 1] = yn + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y