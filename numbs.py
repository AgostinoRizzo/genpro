import sys

INFTY = 10 #sys.float_info.max  # TODO: fix it (consider overflow for qp).
EPSILON = 1e-5 #1e-16  # TODO: consider it differently for qp?!

def tostr(n:float) -> str:
    if abs(n) == INFTY:
        return f"{'-' if n < 0 else ''}INFTY"
    if abs(n) == EPSILON:
        return f"{'-' if n < 0 else ''}EPSILON"
    return str(n)