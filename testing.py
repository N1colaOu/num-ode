import rk23 as rk23
import numpy as np
import matplotlib.pyplot as plt

t_curr_rk23 = 0.0
t_end = 1.0
y_curr_rk23 = [1.0] #!!!1 has to be initialized as vector, as to function, bc all the code is for vectors
h_curr_rk23 = 0.1
eps_tol = 0.3
lam = 0.5

def f(t, y):
    return -10*y

y_data_rk23 = [y_curr_rk23] #plotting data arrays
t_data_rk23 = [t_curr_rk23]
while t_curr_rk23 < t_end:
    if t_curr_rk23 + h_curr_rk23 > t_end: #if we are about to overshoot, we correct the step so its fits perfect
        h_curr_rk23 = t_end - t_curr_rk23
    h_new, y_new, success = rk23.iterate(t_curr_rk23, h_curr_rk23, y_curr_rk23, eps_tol, f, lam) #get the new values
    if success: # if we made the step we update
        y_curr_rk23 = y_new
        t_curr_rk23 += h_curr_rk23
        y_data_rk23.append(y_curr_rk23)
        t_data_rk23.append(t_curr_rk23)
    h_curr_rk23 = h_new # the step has to always be updated no matter the success


t_anal = np.linspace(0, 1, 100)
y_anal = np.exp(-10 * t_anal)

plt.figure(figsize=(8, 6)) 
plt.plot(t_data_rk23, y_data_rk23, marker='o', linestyle='-', label='RK23 (adaptive)')
plt.plot(t_anal, y_anal, 'k--', label='analytical')
plt.title('Task 1, adaptive, tol=0.3, safety=0.5')
plt.legend()
plt.show()



