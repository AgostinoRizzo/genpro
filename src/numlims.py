INFTY        = 10     #sys.float_info.max  # TODO: fix it (consider overflow for qp).
EPSILON      = 1e-12  #1e-16  # TODO: consider it differently for qp?!
INEQ_EPSILON = 1e-50
STEPSIZE     = 1e-10

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
    
    def set_bounds(self, xl:float, xu:float):  # TODO: mutlivar extension.
        infty_step = (xu - xl) * 2
        self.INFTY = max( abs(xl-infty_step), abs(xu+infty_step) )
    
    def tostr(self, n:float) -> str:
        if abs(n) == self.INFTY:
            return f"{'-' if n < 0 else ''}INFTY"
        if abs(n) == self.EPSILON:
            return f"{'-' if n < 0 else ''}EPSILON"
        return str(n)