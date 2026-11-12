import rk23 as rk23
import numpy as np
import matplotlib.pyplot as plt

miu = 0.012277471
T = 17.06521656015796
miu_hat = 1 - miu

# u, v, z, k
# u = x1_dot , v = x1, z = x2_dot, k = x2
# => y = (v, u, k, z), y_dot = f(t, y)
v0 = 0.994
k0 = 0.0
u0 = 0.0
z0 = -2.001585106376082
y_curr_rk23 = np.array([v0, u0, k0, z0])
t_curr_rk23 = 0.0
eps_tol = 0.001
lam = 0.5
h_curr_rk23 = 0.1

def N1(v, k):
    return ((v+miu)**2 + k**2)**1.5
def N2(v, k):
    return ((v-miu_hat)**2 + k**2)**1.5

def f(t, y): #the whole system right hand side
    v, u, k, z = y
    v_dot = u
    u_dot = v + 2*z - miu_hat*(v+miu)/N1(v, k) - miu*(v-miu_hat)/N2(v, k)
    k_dot = z
    z_dot = k - 2*u - miu_hat*k/N1(v, k) - miu*k/N2(v, k)
    return np.array([v_dot, u_dot, k_dot, z_dot])

y_data_rk23 = [y_curr_rk23.copy()]
t_data_rk23 = [t_curr_rk23]
h_data_rk23 = [h_curr_rk23]
while t_curr_rk23 < T: #same while loop as the other file
    if t_curr_rk23 + h_curr_rk23 > T:
        h_curr_rk23 = T - t_curr_rk23
    h_new, y_new, success = rk23.iterate(t_curr_rk23, h_curr_rk23, y_curr_rk23, eps_tol, f, lam)
    if success:
        y_curr_rk23 = y_new
        t_curr_rk23 += h_curr_rk23
        y_data_rk23.append(y_curr_rk23.copy())
        t_data_rk23.append(t_curr_rk23)
        h_data_rk23.append(h_curr_rk23)
    h_curr_rk23 = h_new


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot([state[0] for state in y_data_rk23], [state[2] for state in y_data_rk23], label='RK23 (adaptive)')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.legend()

ax2.plot(range(len(h_data_rk23)), h_data_rk23, label='time step size')
ax2.set_xlabel('step number')
ax2.set_ylabel('dt')
ax2.legend()

plt.tight_layout()
plt.show()