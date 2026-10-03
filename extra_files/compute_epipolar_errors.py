import numpy as np

def compute_epipolar_errors(F, x1s, x2s):
    l = F @ x1s

    a = l[0, :]
    b = l[1, :]
    c = l[2, :]

    u = x2s[0, :]
    v = x2s[1, :]

    num = np.abs(a*u + b*v + c)
    den = np.sqrt(a*a + b*b)

    return num / den
