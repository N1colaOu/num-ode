import numpy as np

A = np.array([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.25,0.25,0.0]], np.double)
b = np.array([0.5, 0.5, 0.0], np.double)
b_star = np.array([1/6, 1/6, 2/3], np.double)
c = np.array([0.0 ,1.0 ,0.5], np.double)

def get_k(A, c, hn, tn, yn, func):
    n_coeff = len(c)
    n_func = len(yn)
    k = np.zeros([n_func, n_coeff])
    for i in range(n_coeff):
        Ki = np.zeros(n_func)
        for j in range(0, i):
            Ki += A[i][j]*k[:, j]
        k[:, i] = func(tn+c[i]*hn, yn + hn*Ki)
    return k 

def step(hn, yn, k, b):
    n_func = len(yn)
    n_coeff = k.shape[1]
    psi = np.zeros(n_func)
    for i in range(n_coeff):
        psi += b[i]*k[:, i]
    return yn + hn*psi 

def iterate(tn, hn, yn, eps_tol, func, lam):
    k = get_k(A, c, hn, tn, yn, func)
    y_2 = step(hn, yn, k, b)
    y_3 = step(hn, yn, k, b_star)
    eps = np.max(np.absolute(y_2-y_3))
    hn1 = hn*(lam*eps_tol/eps)**(1.0/3.0)
    if eps < eps_tol:
        return hn1, y_3, True
    return hn1, y_3, False