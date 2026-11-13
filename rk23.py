from collections.abc import Set

import numpy as np

A = np.array([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.25,0.25,0.0]], dtype=float)
b = np.array([0.5, 0.5, 0.0], dtype=float)
b_star = np.array([1/6, 1/6, 2/3], dtype=float)
c = np.array([0.0 ,1.0 ,0.5], dtype=float)
#constants for the method, the butcher schema

def get_k(A, c, hn, tn, yn, func):
   #Compute stage derivatives; each column stores one vector derivative.
    yn = np.atleast_1d(np.asarray(yn, dtype=float))
    n_coeff = len(c)
    n_func = len(yn)
    k = np.zeros((n_func, n_coeff))
    for i in range(n_coeff):
        increment = np.zeros(n_func)
        for j in range(i):
            increment += A[i, j] * k[:, j]
        k[:, i] = func(tn + c[i] * hn, yn + hn * increment)
    return k


def step(hn, yn, k, b):
    #Combine shared stage derivatives with either set of weights.
    yn = np.atleast_1d(np.asarray(yn, dtype=float))
    psi = np.zeros(len(yn))
    for i in range(k.shape[1]):
        psi += b[i] * k[:, i]
    return yn + hn * psi


def iterate(tn, hn, yn, eps_tol, func, lam, *, return_error=False):
    #Attempt one step without changing the caller's current time or state.
    # The default return value is (h_new, y3, success), matching the original API.
    # Set return_error=True to append the embedded error estimate to that tuple.
    # A rejected candidate must not replace the caller's current state.
    
    if not (np.isfinite(hn) and hn > 0 and np.isfinite(eps_tol)
            and eps_tol > 0 and np.isfinite(lam) and 0 < lam <= 1):
        raise ValueError("Require finite hn > 0, eps_tol > 0, and 0 < lam <= 1.")
    yn = np.atleast_1d(np.asarray(yn, dtype=float))
    k = get_k(A, c, hn, tn, yn, func)
    y_2 = step(hn, yn, k, b)
    y_3 = step(hn, yn, k, b_star)
    eps = float(np.max(np.abs(y_2 - y_3)))
    if not np.isfinite(eps):
        raise FloatingPointError("The embedded error estimate is not finite.")
    # Avoid division by zero when both embedded values coincide.
    hn1 = 2.0 * hn if eps == 0.0 else hn * (lam * eps_tol / eps)**(1/3)
    result = (hn1, y_3, eps < eps_tol)
    return result + (eps,) if return_error else result