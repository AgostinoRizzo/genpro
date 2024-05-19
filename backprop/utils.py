import sys
sys.path.append('..')

import numpy as np
import dataset
import numbs
import lpbackprop
import qp


def map_break_points(K:dataset.DataKnowledge) -> dict:
    break_points = set()
    break_points.add(0)
    break_points.add( numbs.INFTY)
    break_points.add(-numbs.INFTY)

    for vals in K.derivs.values():
        for dp in vals:
            break_points.add(dp.x)
    
    for vals in K.sign.values():
        for (l,u,sign,th) in vals:
            break_points.add(l)
            break_points.add(u)
    
    for (x, iseven) in K.symm.values():
        break_points.add(x)
    
    abs_break_points = set()
    for p in break_points:
        abs_break_points.add(abs(p))
    abs_break_points = sorted(abs_break_points)

    idx_break_points_map = {}
    for idx in range(len(abs_break_points)):
        idx_break_points_map[abs_break_points[idx]] = idx
    
    break_points_map    = {}
    break_points_invmap = {}
    for p in break_points:
        val = idx_break_points_map[abs(p)] * (1 if p > 0 else -1)
        break_points_map   [p  ] = val
        break_points_invmap[val] = p
        #print(f"Mapping {p} -> {val}")
            
    return break_points_map, break_points_invmap


# constrs: map (key=deriv degree) of Constraints.
def check_constrs_sat(P:np.poly1d, constrs:dict[qp.Constraints]) -> bool:
    for derivdeg in constrs.keys():
        _P = np.polyder( P, m=derivdeg ) if derivdeg > 0 else P

        for (dp, relopt) in constrs[derivdeg].eq_ineq:
            if relopt.opt != '=': continue
            print(f"Checking (derivdeg={derivdeg}; x={dp.x}) {_P(dp.x)} {relopt.opt} {dp.y}")
            if not relopt.check( _P(dp.x), dp.y):
                return False
        
        for (x1, x2, iseven) in constrs[derivdeg].symm:
            if iseven:
                if _P(x1) !=  _P(x2): return False
            else:
                if _P(x1) != -_P(x2): return False
    
    return True


# constrs: map (key=deriv degree) of Constraints.
# returns a simplified version of P by removing (=0) some coefficients
#       while keeping the satisfaction of constrs
# it is assumed that P is compliant w.r.t. constrs
def simplify_poly(P:np.poly1d, constrs:dict[qp.Constraints]) -> np.poly1d:
    coeffs_mask = []
    for i in range(P.c.size):
        if abs(P.c[i]) < 1e-16:  # TODO: fix epsilon.
            P.c[i] = 0
            coeffs_mask.append(0)
        else:
            coeffs_mask.append(1 if P.c[i] > 0 else -1)
    return P, coeffs_mask

    # TODO:
    if not check_constrs_sat(P, constrs):
        print('P is not compliant!')
        return P
    for i in range(P.c.size):
        ci_val = P.c[i]
        P.c[i] = 0
        if not check_constrs_sat(P, constrs):
            P.c[i] = ci_val
    return P


def find_matrix_row(A:list[list[float]], row:list[float]) -> bool:
    for A_i in A:
        found = True
        for j in range(len(A_i)):
            if A_i[j] != row[j]:
                found = False
                break
        if found: return True
    return False


def coeffs_softmax(c:np.array) -> np.array:
    c_norm = np.abs(c)
    c_norm = (c - c.min()) / (c.max() - c.min())
    return np.exp(c_norm) / np.sum(np.exp(c_norm))
