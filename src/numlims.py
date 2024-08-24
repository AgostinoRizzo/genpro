import numpy as np


INFTY        = 10     #sys.float_info.max  # TODO: fix it (consider overflow for qp).
EPSILON      = 1e-3  #1e-16  # TODO: consider it differently for qp?!
INEQ_EPSILON = 1e-50
STEPSIZE     = 1e-6


def tostr(n:float) -> str:
    if abs(n) == INFTY:
        return f"{'-' if n < 0 else ''}INFTY"
    if abs(n) == EPSILON:
        return f"{'-' if n < 0 else ''}EPSILON"
    return str(n)


class NumericLimits:
    def __init__(self,
                 infty:float=INFTY,
                 epsilon:float=EPSILON,
                 ineq_epsilon:float=INEQ_EPSILON,
                 stepsize:float=STEPSIZE):
        self.INFTY = infty
        self.EPSILON = epsilon
        self.INEQ_EPSILON = ineq_epsilon
        self.STEPSIZE = stepsize
    
    def set_bounds(self, xl, xu):
        isscalar = np.isscalar(xl)
        assert isscalar == np.isscalar(xu)
        max_idx = np.argmax(xu - xl)
        (max_xl, max_xu) = (xl, xu) if isscalar else (xl[max_idx], xu[max_idx])
        infty_step = (max_xu - max_xl) * 2
        self.INFTY = max( abs(max_xl-infty_step), abs(max_xu+infty_step) )
    
    def tostr(self, n:float) -> str:
        if abs(n) == self.INFTY:
            return f"{'-' if n < 0 else ''}INFTY"
        if abs(n) == self.EPSILON:
            return f"{'-' if n < 0 else ''}EPSILON"
        return str(n)
    
    def __str__(self) -> str:
        out_str = f"INFTY = {self.INFTY}\n" + \
            f"EPSILON = {self.EPSILON}\n" + \
            f"INEQ_EPSILON = {self.INEQ_EPSILON}\n" + \
            f"STEPSIZE = {self.STEPSIZE}\n"
        return out_str