import dataset
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def __y_kernel(x:float, xl:float, xu:float, dp:dataset.DataPoint) -> float:
     
    if dp.x < xl or dp.x > xu: return 0.
        
    h = (xu - x) if dp.x >= x else (x - xl)
    u = (dp.x - x) / h
    u_abs = abs(u)

    # Tukey's tri-weight kernel function
    if u_abs > 1: return 0.
    return (1- (u_abs ** 3)) ** 3


def __propagate_bound(bin_descr:dict, l:float, u:float, threshold:float, sign:str='+'):
    for x in bin_descr.keys():
        if x < l or x > u: continue
        if sign == '+': bin_descr[x]['lower'] = max( threshold, bin_descr[x]['lower'] )
        else: bin_descr[x]['upper'] = min( threshold, bin_descr[x]['upper'] )


def __get_bound(x:float, bin_descr:dict, bound:str='lower') -> float:
    if x in bin_descr.keys(): return bin_descr[x][bound]

    closest_xbin = None
    closest_dist = None
    for xbin in bin_descr.keys():
        dist = abs(x - xbin)
        if closest_xbin is None or dist < closest_dist:
            closest_xbin = xbin
            closest_dist = dist
    
    if closest_xbin is None:
        raise RuntimeError('No closest bin found.')
    return bin_descr[closest_xbin][bound]


def __get_xbin_range(bin_descr:dict, x1:float, x2:float) -> list:
    l = min(x1, x2)
    u = max(x1, x2)
    xrng = []
    for x in bin_descr.keys():
        if x >= l and x <= u: xrng.append(x)
    xrng.sort(reverse=x1>x2)
    return xrng


def __integrate_concavity_to_bounds(bin_descr:dict, xl:float, xu:float, sign:str='+'):
    if xl == xu: return

    bound = 'upper' if sign == '+' else 'lower'
    yl = __get_bound(xl, bin_descr, bound)
    yu = __get_bound(xu, bin_descr, bound)
    m_line = ((yu-yl)/(xu-xl))  # line slope
    line = lambda x: m_line * (x-xl) + yl  # y = m(x-xl) + yl
    line_dist = lambda x, y: abs(m_line * x - y + (yl - m_line*xl)) / math.sqrt(m_line**2 + 1)  # mx - y + (yl-mxl) = 0

    xbinrgn = __get_xbin_range(bin_descr, xl, xu)
    for x in xbinrgn:
        if sign == '+': bin_descr[x][bound] = min( bin_descr[x][bound], line(x) )
        else:           bin_descr[x][bound] = max( bin_descr[x][bound], line(x) )
    
    best_x = None
    best_y_val = None
    best_dist = None
    for x in xbinrgn:
        if x == xl or x == xu: continue
        
        y_val = bin_descr[x][bound]
        dist = line_dist(x, y_val)
        if dist == 0 or (y_val >= line(x) if sign == '+' else y_val <= line(x)): continue

        if best_x is None or dist > best_dist:
            best_x = x
            best_y_val = y_val
            best_dist = dist
    
    if best_x is None: return
    
    __integrate_concavity_to_bounds(bin_descr, xl, best_x, sign)
    __integrate_concavity_to_bounds(bin_descr, best_x, xu, sign)


def __integrate_knowledge_to_bounds(S:dataset.Dataset, bin_descr:dict):
    ## image interc
    if 0 in S.knowledge.derivs.keys():
        for dp in S.knowledge.derivs[0]:
            bin_descr[dp.x] = {'mean': dp.y, 'lower': dp.y, 'upper': dp.y}
    
    ## positivity/negativity
    if 0 in S.knowledge.sign.keys():
        for (l,u,sign,th) in S.knowledge.sign[0]:
            __propagate_bound(bin_descr, l, u, th, sign)
    
    ## monotonicity
    if 1 in S.knowledge.sign.keys():
        for (l,u,sign,_) in S.knowledge.sign[1]:
            if sign == '+':
                for x in __get_xbin_range(bin_descr, l, u): __propagate_bound(bin_descr, x, u, __get_bound(x, bin_descr, 'lower'), '+')
                for x in __get_xbin_range(bin_descr, u, l): __propagate_bound(bin_descr, l, x, __get_bound(x, bin_descr, 'upper'), '-')
            else:
                for x in __get_xbin_range(bin_descr, u, l): __propagate_bound(bin_descr, l, x, __get_bound(x, bin_descr, 'lower'), '+')
                for x in __get_xbin_range(bin_descr, l, u): __propagate_bound(bin_descr, x, u, __get_bound(x, bin_descr, 'upper'), '-')
    
    ## concavity
    if 2 in S.knowledge.sign.keys():
        for (l,u,sign,_) in S.knowledge.sign[2]:
            __integrate_concavity_to_bounds(bin_descr, l, u, sign)
            

def compute_bounds(S:dataset.Dataset, npbins:int, exp_ratio:float=0.1) -> dict:
    X = np.linspace(S.xl, S.xu, npbins).tolist()
    exp_radius = (S.xu - S.xl) * exp_ratio * 0.5
    bin_coverage = 10 #len(S.data) * 0.05
    bin_descr = {}

    for x in X:
        l = x - exp_radius
        u = x + exp_radius
        covered_dps = []

        for dp in S.data:
            if dp.x >= l and dp.x < u:
                covered_dps.append(dp)
        
        if len(covered_dps) < bin_coverage:
            n_missing = bin_coverage - len(covered_dps)
            n2_missing = max( int(n_missing // 2), 1 )
            for _ in range(n2_missing):
                x_unif = random.uniform(l, u)
                covered_dps.append( dataset.DataPoint(x_unif, S.yl) )  #TODO: distribution based on kernel
                covered_dps.append( dataset.DataPoint(x_unif, S.yu) )
        
        Y = []
        W = []
        for dp in covered_dps:
            Y.append( dp.y )
            W.append( __y_kernel(x, l, u, dp) )
        
        Y_sample = Y#random.choices( Y, weights=W, k=500 )
        #print(Y_sample)
        #print(W)

        unique, counts = np.unique(Y_sample, return_counts=True)
        #print(f"x = {x}")
        #print(unique)
        #print(counts)
        #print()
        

        descr = {}
        descr['mean']  = np.mean(Y_sample)
        descr['std']   = np.std(Y_sample)
        descr['count'] = len(Y_sample)

        #print(descr['mean'])
        #print(descr['std'])
        #print(descr['count'])
        
    
        if descr['count'] == 0:
            descr['ci'] = None
            descr['lower'] = S.yl
            descr['upper'] = S.yu
        else:
            descr['ci'] = abs(1.96 * descr['std'] / math.sqrt(descr['count']))
            #descr['lower'] = np.min(Y_sample)
            #descr['upper'] = np.max(Y_sample)
            descr['lower'] = descr['mean'] - 2*descr['std']
            descr['upper'] = descr['mean'] + 2*descr['std']
        
        #print(descr['ci'])
        #print()

        bin_descr[x] = descr
    
    __integrate_knowledge_to_bounds(S, bin_descr)

    return bin_descr


def plot_bounds(S:dataset.Dataset, bin_bounds:dict):
    S.plot()

    X = list(bin_bounds.keys())
    X.sort()
    L = [bin_bounds[x]['lower'] for x in X]
    U = [bin_bounds[x]['upper'] for x in X]
    #M = [bin_bounds[x]['mean'] for x in X]

    plt.plot(X, L, linestyle='dashed', linewidth=2, color='green', label='CI Lower');
    plt.plot(X, U, linestyle='dashed', linewidth=2, color='red', label='CI Upper');
    #plt.plot(X, M, linestyle='dashed', linewidth=2, color='green', label='Mean');


def clean_data(S:dataset.Dataset, bin_bounds:dict):
    cleaned_data = []
    for dp in S.data:
        l = __get_bound(dp.x, bin_bounds, 'lower')
        u = __get_bound(dp.x, bin_bounds, 'upper')
        if dp.y >= l and dp.y <= u: cleaned_data.append(dp)
    S.data = cleaned_data
