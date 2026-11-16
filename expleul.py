import numpy as np 

def explicit(f, y0, t0, t_end, h):   
     #edge cases h<=0, t_end < t0
    if h <= 0:
        print("h must be positive.")
        return None, None

    if t_end < t0:
        print("t_end must be greater than or equal to t0.")
        return None, None

   
    step_number = int(np.ceil((t_end - t0)/h))
    #round up for min number of steps h needed to get from t0 to t_end
    t = np.zeros(step_number + 1)

    if isinstance(y0, (int, float)):
        y = np.zeros(step_number + 1)
    else: 
        y = np.zeros((step_number + 1, len(y0)))

    t[0] = t0
    y[0] = y0

    for n in range(step_number):
     t[n + 1] = t[n] + h
     y[n + 1] = y[n] + h * f(t[n], y[n])

    return t, y