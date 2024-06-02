import dataset

INFTY        = 10     #sys.float_info.max  # TODO: fix it (consider overflow for qp).
EPSILON      = 1e-50  #1e-16  # TODO: consider it differently for qp?!
INEQ_EPSILON = 1e-50
STEPSIZE     = 1e-10

def tostr(n:float) -> str:
    if abs(n) == INFTY:
        return f"{'-' if n < 0 else ''}INFTY"
    if abs(n) == EPSILON:
        return f"{'-' if n < 0 else ''}EPSILON"
    return str(n)

def init(S):
    global INFTY
    infty_step = (S.xu - S.xl) * 2
    INFTY = max( abs(S.xl-infty_step), abs(S.xu+infty_step) )