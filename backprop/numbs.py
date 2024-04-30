import sys

INFTY = sys.float_info.max

def tostr(n:float) -> str:
    if abs(n) == INFTY:
        return f"{'-' if n < 0 else ''}INFTY"
    return str(n)