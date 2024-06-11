import numpy as np

def normval(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def is_in(val, b1, b2) -> bool:
    if b1 < b2: return val >= b1 and val <= b2
    return val >= b2 and val <= b1