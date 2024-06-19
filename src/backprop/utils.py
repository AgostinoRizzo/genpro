from scipy.interpolate import lagrange as scipy_interp_lagrange
import numpy as np

import dataset


def map_break_points(K:dataset.DataKnowledge) -> dict:
    break_points = set()
    break_points.add(0)
    break_points.add( K.numlims.INFTY)
    break_points.add(-K.numlims.INFTY)

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


# constrs:dict[qp.Constraints] map (key=deriv degree) of Constraints.
def check_constrs_sat(P:np.poly1d, constrs:dict) -> bool:
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


# constrs:dict[qp.Constraints] map (key=deriv degree) of Constraints.
# returns a simplified version of P by removing (=0) some coefficients
#       while keeping the satisfaction of constrs
# it is assumed that P is compliant w.r.t. constrs
def simplify_poly(P:np.poly1d, constrs:dict) -> np.poly1d:
    canBeZeroCoeffs = [True] * P.c.size
    for derivdeg, constrs in constrs.items():
        if constrs.noroot: canBeZeroCoeffs[-derivdeg-1] = False
    
    coeffs_mask = []
    for i in range(P.c.size):
        if abs(P.c[i]) < 1e-8 and canBeZeroCoeffs[i]:  # TODO: fix epsilon.
            P.c[i] = 0
            coeffs_mask.append(0)
        else:
            coeffs_mask.append(1 if P.c[i] > 0 else -1)
    return P, None #coeffs_mask

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


def compute_data_weight(data:dataset.NumpyDataset,
                        local_stree, #:backprop.UnknownSyntaxTree,
                        global_stree #:backprop.SyntaxTree
                       ) -> np.array:
    
    local_model = local_stree.model
    
    local_stree.model = lambda X: data.y
    dy = global_stree(data.X)

    local_stree.model = lambda X: data.y + data.numlims.STEPSIZE
    dy = np.absolute(global_stree(data.X) - dy)

    local_stree.model = local_model

    return dy


def scale_data_weight(data_W:np.array, u:float=1., l:float=0.) -> np.array:
    w_min = data_W.min()
    w_max = data_W.max()
    return l + ( (data_W - w_min) * (u - l) / (w_max - w_min) )


def compute_origin_data_weight(S:dataset.Dataset) -> np.array:
    max_dist = max( abs(S.xl), abs(S.xu) )
    W = np.empty(len(S.data))
    
    for i, dp in enumerate(S.data):
        #W[i] = 1 / ((abs(dp.x) / max_dist) ** 4)
        #print(f"Weight at {dp.x} is {W[i]}")
        W[i] = 0. if abs(dp.x) > 0.5 else 1.
    
    return W


def scale_y(Y:np.array) -> tuple[np.array,float]:
    Y_abs = np.abs(Y)
    y_max = Y_abs.max()
    return Y / y_max, y_max


def unscale_polycoeffs(coeffs:np.array, scale_factor:float, xl:float=-1, xu:float=1) -> np.array:
    P = np.poly1d( [1.] * coeffs.size )
    for i, c in enumerate(coeffs):
        P.c[i] = c

    X = np.linspace(xl, xu, P.c.size)
    Y = P(X)
    Y_unscaled = Y * scale_factor
    unscaled_coeffs = scipy_interp_lagrange(X, Y_unscaled).c
    return np.zeros(coeffs.size) if unscaled_coeffs.size == 1 and unscaled_coeffs[0] == 0 else unscaled_coeffs


# pressure: percentage of the total weight (data + weak constraints) given by all weak constraints.
# returns weight of a single weak constraint.
def compute_weakconstr_weight(w_data:np.array, n_weak_constrs:int, pressure:float=0.2) -> float:
    return ( (pressure/(1-pressure)) * w_data.sum() ) / n_weak_constrs


# returns:
#   True  : a better than b.
#   False : a is equal b or b is better than a.
#   When r2 score under 0.1, knowledge comparison is ignored.
def compare_fit(k_mse_a, r2_a, k_mse_b, r2_b) -> bool:
    if k_mse_a < k_mse_b and r2_a >= 0.1: return True
    if k_mse_b < k_mse_a and r2_b >= 0.1: return False
    return r2_a > r2_b