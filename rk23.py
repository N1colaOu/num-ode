import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.25,0.25,0.0]], np.double)
b = np.array([0.5, 0.5, 0.0], np.double)
b_star = np.array([1/6, 1/6, 2/3], np.double)
c = np.array([0.0 ,1.0 ,0.5], np.double)
eps_tol = 1e-4


def f(t, y):
    return y

def get_k(A, b, c, hn, tn, yn, func):
    n_coeff = len(b)
    n_func = len(yn)
    assert(n_coeff == len(c))
    k = np.zeros([n_func, n_coeff])
    for i in range(n_coeff):
        Ki = np.zeros(n_func)
        for j in range(0, i):
            Ki += A[i][j]*k[:, j]
        k[:, i] = func(tn+c[i]*hn, yn + hn*Ki)
    return k #works   

def step(tn, hn, yn, func):
    k = get_k(A, b, c, hn, tn, yn, func)
    n_func = len(yn)
    n_coeff = k.shape[1]
    psi = np.zeros(n_func)
    for i in range(n_coeff):
        psi += b[i]*k[:, i]
    return yn + hn*psi #works

def step_star(tn, hn, yn, func):
    k = get_k(A, b_star, c, hn, tn, yn, func)
    n_func = len(yn)
    n_coeff = k.shape[1]
    psi = np.zeros(n_func)
    for i in range(n_coeff):
        psi += b_star[i]*k[:, i]
    return yn + hn*psi #works