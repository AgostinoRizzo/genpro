import numpy as np
import sympy
from qpsolvers import solve_ls as qpsolvers_solve_ls
import logging

import dataset
import numlims
from backprop import backprop
from backprop import lpbackprop
from backprop import utils
from backprop import models
from backprop.test import test_qp


class Constraints:
    def __init__(self):
        self.eq_ineq:list[dataset.DataPoint, backprop.Relopt] = []
        self.symm:list[x1:float, x2:float, iseven:bool] = []
        self.noroot = False
    
    def isempty(self) -> bool:
        return len(self.eq_ineq) == 0 and len(self.symm) == 0 and not self.noroot


# TODO: open/closed interval (now open) + same constraints?! + interval with INFTY.
def get_constraints(K:dataset.DataKnowledge, break_points:set, derivdeg:int=0, sample_size=10) -> Constraints:
    EPSILON = K.numlims.EPSILON
    constrs = Constraints()

    # eq constrs.
    if derivdeg in K.derivs.keys():
        for dp in K.derivs[derivdeg]:
            constrs.eq_ineq.append( (dataset.DataPoint(dp.x,dp.y), backprop.Relopt('=')) )
    
    # ieq constrs.
    if derivdeg in K.sign.keys():
        for (_l,_u,sign,th) in K.sign[derivdeg]:
            l = _l + EPSILON  # open interval
            u = _u - EPSILON
            if l < u:  # avoid l==u and l>u
                X = np.linspace(l, u, sample_size).tolist()
                for x in X:
                    constrs.eq_ineq.append( (dataset.DataPoint(x,th), backprop.Relopt( '>' if sign == '+' else '<' )) )
    
    # symm constrs.
    if derivdeg in K.symm.keys() and derivdeg == 0:
        (x_0, iseven) = K.symm[derivdeg]
        X = [] #np.linspace(x_0 + EPSILON, INFTY, sample_size).tolist()
        for bp in break_points:
            if bp > x_0:
                X.append(bp)
        for x in X:
            constrs.symm.append( (x, x_0-(x-x_0), iseven) )
    
    # noroot constr.
    if derivdeg in K.noroot:
        constrs.noroot = True

    return constrs


# derivdeg: constraints about the derivdeg-th derivative
def get_qp_constraints(constrs:Constraints, polydeg:int, derivdeg:int, lb:np.array, ub:np.array, limits:numlims.NumericLimits):
    if not (polydeg >= derivdeg):
        logging.debug(f"polydeg = {polydeg}")
        logging.debug(f"derivdeg = {derivdeg}")
    #assert polydeg >= derivdeg TODO: maybe it is not necessary to assert this.
    G = []; h = []
    A = []; b = []

    P = models.ModelFactory.create_poly(polydeg)
    P.set_coeffs(1.)
    P = P.get_deriv( (0,)*derivdeg )
    P_c = np.copy(P.get_coeffs())
    P.set_coeffs(0.)

    # define ineq constraints (G, h) as G <= h (using G <= h-epsilon for G < h)
    some_positiv_constr = False
    for (dp, relopt) in constrs.eq_ineq:
        if relopt.opt not in ['>', '<', '>=', '<=']: continue
        if relopt.opt in ['>', '>='] and dp.y >= 0: some_positiv_constr = True
        flip_sign = 1 if relopt.opt in ['<', '<='] else -1
        G_row = []
        for p in range(polydeg, -1, -1):
            if p < derivdeg:  # TODO: factorize with (*)
                G_row.append(0)
                continue
            c_idx = polydeg-p
            P.set_coeff(c_idx, P_c[c_idx])
            G_row.append( P(dp.x) * flip_sign )
            P.set_coeff(c_idx, 0)
        h.append( (dp.y * flip_sign) - (limits.INEQ_EPSILON if relopt.opt in ['>', '<'] else 0) )
        G.append(G_row)

    # define eq constraints (A, b)
    for (dp, relopt) in constrs.eq_ineq:
        if relopt.opt != '=': continue
        A_row = []
        for p in range(polydeg, -1, -1):
            if p < derivdeg:  # TODO: (*)
                A_row.append(0)
                continue
            c_idx = polydeg-p
            P.set_coeff(c_idx, P_c[c_idx])
            A_row.append( P(dp.x) )
            P.set_coeff(c_idx, 0)
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
            c_idx = polydeg-p
            P.set_coeff(c_idx, P_c[c_idx])
            A_row.append( P(x1) - P(x2) if iseven else P(x1) + P(x2) )
            P.set_coeff(c_idx, 0)
        b.append(0)
        A.append(A_row)
    
    # define noroot constraints as lb or ub in the last coefficient (TODO: it is a soft check).
    if constrs.noroot:
        if some_positiv_constr: lb[-1] = max(lb[-1], limits.INEQ_EPSILON)
        else: ub[-1] = min(ub[-1], -limits.INEQ_EPSILON)

    return G, h, A, b


# coeffs_mask: (0,>0,<0).
def add_qp_coeffs_mask_constraints(polydeg:int, coeffs_mask:list[float], lb:np.array, ub:np.array, limits:numlims.NumericLimits):
    ncoeffs = len(coeffs_mask)
    assert polydeg == ncoeffs - 1

    for i in range(ncoeffs):

        if coeffs_mask[i] == 0:
            lb[i] = 0
            ub[i] = 0
        
        elif coeffs_mask[i] > 0:
            lb[i] = max(lb[i], limits.INEQ_EPSILON)
        
        else:
            ub[i] = min(ub[i], -limits.INEQ_EPSILON)


# constrs: map (key=deriv degree) of Constraints
# weak_constrs: (derivdeg => Constraints) not utilized when S is None
# returns np.array of coefficients (in decreasing powers)
def qp_solve(constrs:dict[Constraints],
             polydeg:int,
             limits:numlims.NumericLimits,
             S:dataset.NumpyDataset=None,
             data_W:np.array=None,
             coeffs_mask:list[float]=None,
             s_val:float=1,
             weak_constrs:dict[int,Constraints]=None):
    
    assert polydeg >= 0
    R = []; s = []; W = None
    G = []; h = []
    A = []; b = []

    nvars = polydeg + 1
    lb = np.full(nvars, -limits.INFTY)
    ub = np.full(nvars,  limits.INFTY)

    Sy_scale_factor = None
    
    # define the objective function (R, s):
    #   - minimize usage of coefficients, if no data provided.
    #   - fit data, otherwise.
    if S is None:
        R = np.identity(nvars, dtype=float)
        s = np.full(nvars, s_val) #np.zeros(nvars)
    else:
        # add data point errors.
        R = np.array( [S.X ** p for p in range(polydeg, -1, -1)] ).T
        s = S.Y
        
        # add weak constraints errors (when provided).
        n_weak_constrs = 0
        if weak_constrs is not None:
            for derivdeg in weak_constrs.keys():
                __G, __h, __A, __b = get_qp_constraints(weak_constrs[derivdeg], polydeg, derivdeg, lb, ub, limits)
                if len(__A) > 0:
                    R = np.append(R, __A, axis=0)
                    s = np.append(s, __b, axis=0)
                    n_weak_constrs += len(__A)
        
        s, Sy_scale_factor = utils.scale_y(s)

        # add error weights (data points + weak constraints).
        w_data       = np.array([1 / S.compute_yvar(x0) for x0 in S.X])  # TODO: optimize even more using numpy.
        w_weakconstr = np.full( n_weak_constrs, utils.compute_weakconstr_weight(w_data, n_weak_constrs) )
        w            = np.concatenate( (w_data, w_weakconstr) )
        W = np.diag( utils.scale_data_weight( w ) )
        #W = np.diag( utils.compute_origin_data_weight(S) )
        #if data_W is not None:
        #    W = np.diag( utils.scale_data_weight(data_W) )
    
    # define ineq (G, h), eq (A, b) and bounds (lb, ub) constraints.
    for derivdeg in constrs.keys():
        __G, __h, __A, __b = get_qp_constraints(constrs[derivdeg], polydeg, derivdeg, lb, ub, limits)
        G += __G; h += __h
        A += __A; b += __b
    if coeffs_mask is not None:  # TODO: we should remove the variable in case of coeffs[i] == 0.
        add_qp_coeffs_mask_constraints(polydeg, coeffs_mask, lb, ub)

    G = np.array(G, dtype=float) if len(G) > 0 else None
    h = np.array(h, dtype=float) if len(h) > 0 else None
    A = np.array(A, dtype=float) if len(A) > 0 else None
    b = np.array(b, dtype=float) if len(b) > 0 else None

    if Sy_scale_factor is not None:
        if h is not None: h /= Sy_scale_factor
        if b is not None: b /= Sy_scale_factor

    # TODO: manage this!
    if A is not None:
        #print(f"Rank of A: {np.linalg.matrix_rank(A)}")
        _, inds = sympy.Matrix(A).T.rref()
        #print(f"Inds of A: {inds}")
        idx_to_remove = []
        for i in range(np.shape(A)[0]):
            if i not in inds: idx_to_remove.append(i)
        #print(f"Idx to remove from A: {idx_to_remove}")
        A = np.delete(A, idx_to_remove, 0)
        b = np.delete(b, idx_to_remove, 0)

    import warnings
    warnings.filterwarnings("ignore")

    #print(f"A = {A}")
    #print(f"b = {b}")

    sol = qpsolvers_solve_ls(R, s, G, h, A, b, W=W, lb=None, ub=None, solver='clarabel', verbose=False)  # returns optimal sol if found, None otherwise.
    #print(f"QP solution: {sol}\n")

    sol = np.zeros(nvars) if sol is None else sol # TODO: manage error or no solution + rows(A)>|x|
    #print(f"QP solution check: {test_qp.check_qp_sol(G, h, A, b, lb, ub, sol)}")

    if Sy_scale_factor is not None:
        sol = utils.unscale_polycoeffs(sol, Sy_scale_factor, S.xl, S.xu)
        #print(f"QP solution unscaled: {sol}\n")
    
    """
    print(f"R = {R}")
    print(f"s = {s}")
    print(f"G = {G}")
    print(f"h = {h}")
    print(f"A = {A}")
    print(f"b = {b}")
    print(f"lb = {lb}")
    print(f"ub = {ub}")
    print()
    """

    return sol