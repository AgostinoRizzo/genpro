import numpy as np
import math
import dataset

def __get_slope(dp_i: dataset.DataPoint, dp_j: dataset.DataPoint) -> float:
    return (dp_j.y - dp_i.y) / (dp_j.x - dp_i.x)

"""
requires S.data as mesh
"""
def compute_smoothness(S: dataset):  # TODO: kernel based computation
    n = len(S.data)
    assert(n >= 2)

    slopes = []
    for i in range(1, n):
        slopes.append(__get_slope(S.data[i-1], S.data[i]))
    
    return np.average(slopes), np.std(slopes)


"""
Sum Operator: a+b=k
k: target image
a: alpha value
return: beta value
"""
def __sum_opt_beta(k:float, a:float) -> float:
    return k - a

"""
Product Operator: a*b=k
k: target image
a: alpha value
return: beta value
"""
def __prod_opt_beta(k:float, a:float) -> float:
    return k / a

"""
Power Operator: a^b=k
k: target image
a: alpha value
return: beta value
"""
def __pow_opt_beta(k:float, a:float) -> float:
    return math.log(k, a)

"""
Division Operator: a/b=k
k: target image
a: alpha value
return: beta value
"""
def __div_opt_beta(k:float, a:float) -> float:
    return a / k

"""
requires S.data as mesh
"""
def infer_operator(S:dataset.Dataset, opt:str='sum'):
    __opt_beta = __sum_opt_beta
    if   opt == 'sum' : pass
    elif opt == 'prod': __opt_beta = __prod_opt_beta
    elif opt == 'pow' : __opt_beta = __pow_opt_beta
    elif opt == 'div' : __opt_beta = __div_opt_beta
    else: raise RuntimeError('Invalid operator.')

    n = len(S.data)
    i = 0
    eps = 0.0001
    y_radius = 0.2

    alphas = []
    betas  = []
    n_iters = 0

    while i < n and n_iters < 100:
        k = S.data[i].y

        if i == 0:
            alphas.append(S.yl)# + eps
            betas  = [__opt_beta(k, alphas[0])]
            if alphas[0] > S.yu:
                break
        
        else:
            l = max(alphas[i-1] - y_radius, S.yl)
            u = min(alphas[i-1] + y_radius, S.yu)

            a_best = None
            b_best = None
            diff_min = y_radius
            for a in np.linspace(l, u, int((u - l) / eps)):
                b = __opt_beta(k, a)
                diff = abs(b - betas[i-1]) #+ abs(a - alphas[i-1])) / 2
                if diff < diff_min:
                    a_best = a
                    b_best = b
                    diff_min = diff
            
            if diff_min == y_radius:
                i = -1
                n_iters += 1
                alphas = []
                betas = []
            else:
                alphas.append(a_best)
                betas.append(b_best)

        i += 1
    
    S_alphas = dataset.Dataset()
    S_betas  = dataset.Dataset()
    
    if len(alphas) == n and len(betas) == n:
        for i in range(n):
            S_alphas.data.append(dataset.DataPoint(S.data[i].x, alphas[i]))
            S_betas .data.append(dataset.DataPoint(S.data[i].x, betas [i]))
    else:
        alphas = []
        betas  = []
        print('No solution found.')

    return alphas, betas, S_alphas, S_betas