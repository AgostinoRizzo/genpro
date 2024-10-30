from scipy.interpolate import lagrange as scipy_interp_lagrange
import numpy as np
import math

import dataset


def map_break_points(K:dataset.DataKnowledge) -> dict:
    break_points = set()
    break_point_coords = set()
    break_point_coords.add(0)
    break_point_coords.add( K.numlims.INFTY)
    break_point_coords.add(-K.numlims.INFTY)

    for vals in K.derivs.values():
        for dp in vals:
            if np.isscalar(dp.x):
                break_point_coords.add(dp.x)
                break_points.add(dp.x)
            else:
                break_point_coords.update(dp.x)
                break_points.add(tuple(dp.x))
    
    for vals in K.sign.values():
        for (l,u,sign,th) in vals:
            if np.isscalar(l):  # u is scalar too!
                break_point_coords.add(l)
                break_point_coords.add(u) 
                break_points.add(l)
                break_points.add(u)
            else:  # u is non-scalar too!
                break_point_coords.update(l)
                break_point_coords.update(u)
                break_points.add(tuple(l))
                break_points.add(tuple(u))
    
    for (x, iseven) in K.symm.values():
        if np.isscalar(x):
            break_point_coords.add(x)
            break_points.add(x)
        else:
            break_point_coords.update(x)
            break_points.add(tuple(x))
    
    abs_break_point_coords = set()
    for p in break_point_coords:
        abs_break_point_coords.add(abs(p))
    abs_break_point_coords = sorted(abs_break_point_coords)

    idx_break_point_coords_map = {}
    for idx in range(len(abs_break_point_coords)):
        idx_break_point_coords_map[abs_break_point_coords[idx]] = idx
    
    break_point_coords_map    = {}
    break_point_coords_invmap = {}
    for c in break_point_coords:
        val = idx_break_point_coords_map[abs(c)] * (1 if c > 0 else -1)
        break_point_coords_map   [c  ] = val
        break_point_coords_invmap[val] = c
        #print(f"Mapping {p} -> {val}")

    return break_points, break_point_coords_map, break_point_coords_invmap


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
    if y_max == 0.:
        return Y, None
    return Y / y_max, y_max


def unscale_polycoeffs(coeffs:np.array, scale_factor:float, xl:float=-1, xu:float=1) -> np.array:
    P = np.poly1d( [1.] * coeffs.size )
    for i, c in enumerate(coeffs):
        P.c[i] = c

    X = np.linspace(xl, xu, P.c.size)
    Y = P(X)
    Y_unscaled = Y * scale_factor
    unscaled_coeffs = scipy_interp_lagrange(X, Y_unscaled).c
    unscaled_coeffs = np.zeros(coeffs.size) if unscaled_coeffs.size == 1 and unscaled_coeffs[0] == 0 else unscaled_coeffs

    if unscaled_coeffs.size < coeffs.size:
        return np.insert(unscaled_coeffs, slice(0, unscaled_coeffs.size-coeffs.size), 0)
    return unscaled_coeffs


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


def deriv_to_string(deriv:tuple[int]) -> str:
    deriv_str = ''
    for varidx in deriv:
        deriv_str += f"d{varidx}"
    return deriv_str

def parse_deriv(deriv_str:str, parsefunc:bool=False) -> tuple[int]|tuple[tuple[int],str]:
    toskip = " \n\t"
    varidx_str = ''
    deriv = []
    i = 0
    for c in deriv_str:
        if c == 'd':
            if len(varidx_str) > 0:
                deriv.append(int(varidx_str))
                varidx_str = ''
        elif c.isdigit():
            varidx_str += c
        elif c not in toskip:
            break
        i += 1
    if len(varidx_str) > 0:
        deriv.append(int(varidx_str))
        varidx_str = ''
    
    if parsefunc: return tuple(deriv), deriv_str[i:]
    return tuple(deriv)


def squarify(M, paddingval=0.):
    maxdimsize = max(M.shape)
    if maxdimsize == min(M.shape): return M
    padding = []
    for dimsize in M.shape:
        padding.append((0,maxdimsize-dimsize))
    return np.pad(M, padding, mode='constant', constant_values=paddingval)


def flatten(v):
    return np.sign(v)


def random_test(y) -> float:
    """
    returns the value of |Z-statistic|
        * The higher the value the more the null hypo (random values) is rejected.
    """

    y_median = np.median(y)

    runs, n1, n2 = 0, 0, 0
      
    # Checking for start of new run 
    for i in range(y.size): 
          
        # no. of runs 
        if (y[i] >= y_median and y[i-1] < y_median) or \
                (y[i] < y_median and y[i-1] >= y_median): 
            runs += 1  
          
        # no. of positive values 
        if(y[i]) >= y_median: 
            n1 += 1   
          
        # no. of negative values 
        else: 
            n2 += 1   
  
    runs_exp = ((2*n1*n2)/(n1+n2))+1
    stan_dev = math.sqrt((2*n1*n2*(2*n1*n2-n1-n2))/ \
                       (((n1+n2)**2)*(n1+n2-1))) 
    
    if stan_dev == 0: return np.inf
    z = (runs-runs_exp)/stan_dev 
  
    return abs(z)


SYMMETRIC_RTOL = 0.2e-15
SYMMETRIC_R2TH = 1.0 - 1e-50

def is_symmetric(y, Y_Ids) -> bool:
    global SYMMETRIC_RTOL
    
    y_0 = y[Y_Ids[0]]
    #y_0_sst = np.sum( (y_0 - y_0.mean()) ** 2 )
    
    for i in range(1, Y_Ids.shape[0]):
        #if not np.allclose(y[Y_Ids[i]], y_0, rtol=SYMMETRIC_RTOL, atol=0.0, equal_nan=True):  #if not np.array_equal(y[Y_Ids[i]], y_0, equal_nan=True):
        if not np.allclose(y[Y_Ids[i]], y_0, equal_nan=True):
            return False

        #ssr = np.sum( (y[Y_Ids[i]] - y_0) ** 2 )
        #r2  = 1 - ((ssr / y_0_sst) if y_0_sst > 0. else 1.)
        #if r2 < SYMMETRIC_R2TH: return False

    return True

def count_symmetric(y, Y_Ids) -> int:
    global SYMMETRIC_RTOL
    n = 0
    
    y_0 = y[Y_Ids[0]]
    #y_0_sst = np.sum( (y_0 - y_0.mean()) ** 2 )

    for i in range(1, Y_Ids.shape[0]):
        #if np.allclose(y[Y_Ids[i]], y_0, rtol=SYMMETRIC_RTOL, atol=0.0, equal_nan=True):  #if np.array_equal(y[Y_Ids[i]], y_0, equal_nan=True):
        if np.allclose(y[Y_Ids[i]], y_0, equal_nan=True):
            n += 1

        #ssr = np.sum( (y[Y_Ids[i]] - y_0) ** 2 )
        #r2  = 1 - ((ssr / y_0_sst) if y_0_sst > 0. else 1.)
        #if r2 >= SYMMETRIC_R2TH:
        #    n += 1

    return n