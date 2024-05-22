import numpy as np


def check_qp_sol(G:np.array, h:np.array,
                 A:np.array, b:np.array,
                 lb:np.array, ub:np.array,
                 x:np.array) -> bool:
    
    if G is not None:
        actual_h = np.matmul(G, x)
        if not np.less_equal(actual_h, h).all(): return False

    if A is not None:
        actual_b = np.matmul(A, x)
        if not np.equal(actual_b, b).all(): return False

    if lb is not None and np.less_equal(x, lb).any(): return False
    if ub is not None and not np.less_equal(x, ub).all(): return False

    return True