from scipy.optimize import fmin_slsqp
import numpy as np
import random
import tree_search
import dataset


__stree:tree_search.SyntaxTree = None
__interc_dpx:np.array = None
__interc_dpy:np.array = None
__pos_dpx:list = None
__neg_dpx:list = None
__pos_deriv_dpx:list = None
__neg_deriv_dpx:list = None
__S = None


def get_func(coeffs, stree):
    global __interc_dpx
    global __interc_dpy
    stree.set_coeffs(coeffs)
    return np.sum( (stree.evaluate(__interc_dpx) - __interc_dpy) ** 2 )


def setcoeffs_n_evaluate(coeffs, stree, x):
    
    return 


def get_ieqcons(coeffs, stree) -> list:
    global __pos_dpx
    global __neg_dpx
    global __pos_deriv_dpx
    global __neg_deriv_dpx

    ieqcons = []
    stree.set_coeffs(coeffs)

    for x in __pos_dpx:
        ieqcons.append( stree.evaluate(x) )
    for x in __neg_dpx:
        ieqcons.append( -stree.evaluate(x) )
    
    for x in __pos_deriv_dpx:
        ieqcons.append( stree.evaluate_deriv(x) )
    for x in __neg_deriv_dpx:
        ieqcons.append( -stree.evaluate_deriv(x) )
    
    #print(f"INEQS: {ieqcons}")
    #print(f"POS_DPX: {__pos_dpx}")
    #print(f"NEG_DPX: {__neg_dpx}")
    return np.array(ieqcons)


def get_eqcons(coeffs, stree) -> list:
    global __S

    eqcons = []
    stree.set_coeffs(coeffs)

    if 0 in __S.knowledge.derivs.keys():
        for dp in __S.knowledge.derivs[0]:
            eqcons.append( stree.evaluate(dp.x) - dp.y )
    
    return np.array(eqcons)
    


def infer_syntaxtree(stree:tree_search.SyntaxTree, S:dataset.Dataset):
    global __stree
    global __interc_dpx
    global __interc_dpy
    global __pos_dpx
    global __neg_dpx
    global __pos_deriv_dpx
    global __neg_deriv_dpx
    global __S

    sample_size = 50

    dp_X = [dp.x for dp in S.data]
    dp_Y = [dp.y for dp in S.data]

    pos_X = []
    neg_X = []
    if 0 in S.knowledge.sign.keys():
        for (l,u,sign,th) in S.knowledge.sign[0]:
            x_vals = np.linspace(l, u, sample_size).tolist()
            if sign == '+': pos_X += x_vals
            else: neg_X += x_vals
    
    pos_deriv_X = []
    neg_deriv_X = []
    if 1 in S.knowledge.sign.keys():
        for (l,u,sign,th) in S.knowledge.sign[1]:
            x_vals = np.linspace(l, u, sample_size).tolist()
            if sign == '+': pos_deriv_X += x_vals
            else: neg_deriv_X += x_vals

    __stree = stree
    __interc_dpx = np.array(dp_X)
    __interc_dpy = np.array(dp_Y)
    __pos_dpx = pos_X
    __neg_dpx = neg_X
    __pos_deriv_dpx = pos_deriv_X
    __neg_deriv_dpx = neg_deriv_X
    __S = S
    
    tot_coeffs = stree.get_ncoeffs()
    coeffs_0 = np.array( [random.uniform(1000., -1000.) for _ in range(tot_coeffs)] )

    res = fmin_slsqp(get_func, coeffs_0, f_ieqcons=get_ieqcons, f_eqcons=get_eqcons, args=(__stree,), iprint=2, iter=500)
    stree.set_coeffs(res)
    print(res)
