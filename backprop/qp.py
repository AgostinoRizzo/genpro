import sys
sys.path.append('..')

import numpy as np
from qpsolvers import solve_ls as qpsolvers_solve_ls
import dataset
import backprop
import lpbackprop
import numbs


class Constraints:
    def __init__(self):
        self.eq_ineq:list[dataset.DataPoint, backprop.Relopt] = []
        self.symm:list[x1:float, x2:float, iseven:bool] = []


# TODO: open/closed interval (now open) + same constraints?! + interval with INFTY.
def get_constraints(K:dataset.DataKnowledge, derivdeg:int=0, sample_size=10) -> Constraints:
    constrs = Constraints()

    # eq constrs.
    if derivdeg in K.derivs.keys():
        for dp in K.derivs[derivdeg]:
            constrs.eq_ineq.append( (dataset.DataPoint(dp.x,dp.y), backprop.Relopt('=')) )
    
    # ieq constrs.
    if derivdeg in K.sign.keys():
        for (_l,_u,sign,th) in K.sign[derivdeg]:
            l = _l + numbs.EPSILON  # open interval
            u = _u - numbs.EPSILON
            if l < u:  # avoid l==u and l>u
                X = np.linspace(l, u, sample_size).tolist()
                for x in X:
                    constrs.eq_ineq.append( (dataset.DataPoint(x,th), backprop.Relopt( '>' if sign == '+' else '<' )) )
    
    # symm constrs.
    if derivdeg in K.symm.keys():
        (x_0, iseven) = K.symm[derivdeg]
        X = np.linspace(x_0 + numbs.EPSILON, numbs.INFTY, sample_size).tolist()
        for x in X:
            constrs.symm.append( (x, x_0-(x-x_0), iseven) )

    return constrs


# derivdeg: constraints about the derivdeg-th derivative
def get_qp_constraints(constrs:Constraints, polydeg:int, derivdeg:int):
    assert polydeg >= derivdeg
    G = []; h = []
    A = []; b = []

    P = np.polyder( np.poly1d([1] * (polydeg+1)), m=derivdeg )
    for i in range(P.c.size): P.c[i] = 0

    # define ineq constraints (G, h) as G <= h (using G <= h-epsilon for G < h)
    for (dp, relopt) in constrs.eq_ineq:
        if relopt.opt not in ['>', '<', '>=', '<=']: continue
        flip_sign = 1 if relopt.opt in ['<', '<='] else -1
        G_row = []
        for p in range(polydeg, -1, -1):
            if p < derivdeg:  # TODO: factorize with (*)
                G_row.append(0)
                continue
            P.c[polydeg-p] = 1
            G_row.append( P(dp.x) * flip_sign )
            P.c[polydeg-p] = 0
        h.append( (dp.y * flip_sign) - (numbs.EPSILON if relopt.opt in ['>', '<'] else 0) )
        G.append(G_row)

    # define eq constraints (A, b)
    for (dp, relopt) in constrs.eq_ineq:
        if relopt.opt != '=': continue
        A_row = []
        for p in range(polydeg, -1, -1):
            if p < derivdeg:  # TODO: (*)
                A_row.append(0)
                continue
            P.c[polydeg-p] = 1
            A_row.append( P(dp.x) )
            P.c[polydeg-p] = 0
        b.append(dp.y)
        A.append(A_row)
    
    # define symmetry constraints as eq constraints (A, b)
    for (x1, x2, iseven) in constrs.symm:
        # iseven==True : P(x1) ==  dp_inputs(x2)
        # iseven==False: P(x1) == -dp_inputs(x2)
        A_row = []
        for p in range(polydeg, -1, -1):
            if p < derivdeg:  # TODO: (*)
                A_row.append(0)
                continue
            P.c[polydeg-p] = 1
            A_row.append( P(dp.x1) - P(dp.x2) if iseven else P(dp.x1) + P(dp.x2) )
            P.c[polydeg-p] = 0
        b.append(0)
        A.append(A_row)
    
    return G, h, A, b


# constrs: map (key=deriv degree) of Constraints
# returns np.array of coefficients (in decreasing powers)
def qp_solve(constrs:dict[Constraints], polydeg:int):
    assert polydeg >= 0
    R = []; s = []
    G = []; h = []
    A = []; b = []
    nvars = polydeg + 1
    
    # define the objective function (R, s) - minimize usage of coefficients.
    for i in range(nvars):
        R_row = [0] * nvars
        R_row[i] = 1
        R.append(R_row)
    s = [0] * nvars
    
    # define ineq (G, h) and eq (A, b) constraints.
    for derivdeg in range(3):
        if derivdeg not in constrs.keys(): continue
        __G, __h, __A, __b = get_qp_constraints(constrs[derivdeg], polydeg, derivdeg)
        G += __G; h += __h
        A += __A; b += __b
    
    """print(f"R = {R}")
    print(f"s = {s}")
    print(f"G = {G}")
    print(f"h = {h}")
    print(f"A = {A}")
    print(f"b = {b}")"""

    R = np.array(R, dtype=float); s = np.array(s, dtype=float)
    G = np.array(G, dtype=float) if len(G) > 0 else None
    h = np.array(h, dtype=float) if len(h) > 0 else None
    A = np.array(A, dtype=float) if len(A) > 0 else None
    b = np.array(b, dtype=float) if len(b) > 0 else None

    sol = qpsolvers_solve_ls(R, s, G, h, A, b, solver='cvxopt', verbose=True)  # returns optimal sol if found, None otherwise.
    print(f"QP solution: {sol}")

    return np.zeros(nvars) if sol is None else sol # TODO: manage error or no solution + rows(A)>|x|