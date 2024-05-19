import sys
sys.path.append('..')

import numpy as np
import sympy
from qpsolvers import solve_ls as qpsolvers_solve_ls
import dataset
import backprop
import lpbackprop
import numbs
import utils


class Constraints:
    def __init__(self):
        self.eq_ineq:list[dataset.DataPoint, backprop.Relopt] = []
        self.symm:list[x1:float, x2:float, iseven:bool] = []


# TODO: open/closed interval (now open) + same constraints?! + interval with INFTY.
def get_constraints(K:dataset.DataKnowledge, break_points:set, derivdeg:int=0, sample_size=10) -> Constraints:
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
    if derivdeg in K.symm.keys() and derivdeg == 0:
        (x_0, iseven) = K.symm[derivdeg]
        X = [] #np.linspace(x_0 + numbs.EPSILON, numbs.INFTY, sample_size).tolist()
        for bp in break_points:
            if bp > x_0:
                X.append(bp)
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
        # iseven==True : P(x1) ==  P(x2)
        # iseven==False: P(x1) == -P(x2)
        A_row = []
        for p in range(polydeg, -1, -1):
            if p < derivdeg:  # TODO: (*)
                A_row.append(0)
                continue
            P.c[polydeg-p] = 1
            A_row.append( P(x1) - P(x2) if iseven else P(x1) + P(x2) )
            P.c[polydeg-p] = 0
        b.append(0)
        A.append(A_row)
    print()
    
    return G, h, A, b


# coeffs_mask: (0,>0,<0).
def add_qp_coeffs_mask_constraints(polydeg:int, coeffs_mask:list[float]):
    ncoeffs = len(coeffs_mask)
    assert polydeg == ncoeffs - 1

    nvars = polydeg + 1
    lb = np.full(nvars, -numbs.INFTY)
    ub = np.full(nvars,  numbs.INFTY)

    for i in range(ncoeffs):

        if coeffs_mask[i] == 0:
            lb[i] = 0
            ub[i] = 0
        
        elif coeffs_mask[i] > 0:
            lb[i] = numbs.EPSILON
        
        else:
            ub[i] = -numbs.EPSILON
    
    return lb, ub


# constrs: map (key=deriv degree) of Constraints
# returns np.array of coefficients (in decreasing powers)
def qp_solve(constrs:dict[Constraints], polydeg:int, S:dataset.Dataset=None, coeffs_mask:list[float]=None):
    assert polydeg >= 0
    R = []; s = []; W = None
    G = []; h = []
    A = []; b = []
    lb = None; ub = None
    nvars = polydeg + 1
    
    # define the objective function (R, s):
    #   - minimize usage of coefficients, if no data provided.
    #   - fit data, otherwise.
    if S is None:
        R = np.identity(nvars, dtype=float)
        s = np.zeros(nvars)
    else:
        for dp in S.data:
            R.append( [dp.x ** p for p in range(polydeg, -1, -1)] )
            s.append( dp.y )
        R = np.array(R, dtype=float)
        s = np.array(s, dtype=float)
        W = np.diag( [1 / S.compute_yvar(dp.x) for dp in S.data] )
    
    # define ineq (G, h), eq (A, b) and bounds (lb, ub) constraints.
    for derivdeg in range(3):
        if derivdeg not in constrs.keys(): continue
        __G, __h, __A, __b = get_qp_constraints(constrs[derivdeg], polydeg, derivdeg)
        G += __G; h += __h
        A += __A; b += __b
    if coeffs_mask is not None:  # TODO: we should remove the variable in case of coeffs[i] == 0.
        lb, ub = add_qp_coeffs_mask_constraints(polydeg, coeffs_mask)

    
    """print(f"R = {R}")
    print(f"s = {s}")
    print(f"G = {G}")
    print(f"h = {h}")
    print(f"A = {A}")
    print(f"b = {b}")"""

    G = np.array(G, dtype=float) if len(G) > 0 else None
    h = np.array(h, dtype=float) if len(h) > 0 else None
    A = np.array(A, dtype=float) if len(A) > 0 else None
    b = np.array(b, dtype=float) if len(b) > 0 else None

    # TODO: manage this!
    print(f"Rank of A: {np.linalg.matrix_rank(A)}")
    _, inds = sympy.Matrix(A).T.rref()
    print(f"Inds of A: {inds}")
    idx_to_remove = []
    for i in range(np.shape(A)[0]):
        if i not in inds: idx_to_remove.append(i)
    print(f"Idx to remove from A: {idx_to_remove}")
    A = np.delete(A, idx_to_remove, 0)
    b = np.delete(b, idx_to_remove, 0)

    sol = qpsolvers_solve_ls(R, s, G, h, A, b, W=W, lb=lb, ub=ub, solver='cvxopt', verbose=False)  # returns optimal sol if found, None otherwise.
    print(f"QP solution: {sol}")

    return np.zeros(nvars) if sol is None else sol # TODO: manage error or no solution + rows(A)>|x|