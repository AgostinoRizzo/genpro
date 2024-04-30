import sys
import random
import numpy as np
from scipy.special import softmax
from qpsolvers import solve_ls 
sys.path.append('..')

import dataset
import backprop


class HistoryEntry:
    def __init__(self, msg:str, model_name:str,
                 pulled_S:dataset.Dataset, pulled_constrs:list, violated_constrs:list,
                 fit_model, simple_found:bool):
        self.msg = msg
        self.model_name = model_name
        self.pulled_S = pulled_S
        self.pulled_constrs = pulled_constrs
        self.violated_constrs = violated_constrs
        self.fit_model = fit_model
        self.simple_found = simple_found

class History:
    def __init__(self):
        self.entries = []
    
    def log_pull(self, unkn_stree:backprop.UnknownSyntaxTree,
                 pulled_S:dataset.Dataset, pulled_constrs:list, violated_constrs:list=[],
                 simple_found:bool=False):
        self.entries.append(
            HistoryEntry(f"Pull from {unkn_stree}", unkn_stree.label, pulled_S, pulled_constrs, violated_constrs, unkn_stree.model, simple_found))


def __get_constraints(DK:dataset.DataKnowledge, derivdeg:int=0, sample_size:int=20, mesh:bool=None) -> list[dataset.DataPoint, backprop.Relopt]:
    constrs = []

    if derivdeg in DK.derivs.keys():
        for dp in DK.derivs[derivdeg]:
            constrs.append( (dataset.DataPoint(dp.x,dp.y), backprop.Relopt('=')) )
    
    if derivdeg in DK.sign.keys():
        if mesh:
            X = np.linspace(DK.dataset.xl, DK.dataset.xu, sample_size).tolist()
            for x in X:
                for (l,u,sign,th) in DK.sign[derivdeg]:
                    if x >= l and x <= u:
                        constrs.append( (dataset.DataPoint(x,th), backprop.Relopt( '>' if sign == '+' else '<' )) )
                        break
                
        else:
            for (l,u,sign,th) in DK.sign[derivdeg]:
                X = np.linspace(l, u, sample_size).tolist()
                for x in X:
                    constrs.append( (dataset.DataPoint(x,th), backprop.Relopt( '>' if sign == '+' else '<' )) )
        
    return constrs


# input variables (coefficients) (in descending order of powers)
# return np.array of coefficients (in decreasing powers)
# pulled_constrs: list of pairs (dataset.DataPoint(dp.x, pulled_th), pulled_relopt)
def __qp_solve(X:np.array, Y:np.array, pulled_constrs:list, polydeg:int):
    R = []; s = []
    G = []; h = []
    A = []; b = []

    # define the objective function (R, s)
    for i in range(X.size):
        R_row = []
        for p in range(polydeg, -1, -1): R_row.append(X[i] ** p)
        s.append(Y[i])
        R.append(R_row)
    
    # define ineq constraints (G, h)
    for (dp, relopt) in pulled_constrs:
        if relopt.opt not in ['>', '<', '>=', '<=']: continue
        flip_sign = 1 if relopt.opt in ['<', '<='] else -1
        for i in range(X.size):
            G_row = []
            for p in range(polydeg, -1, -1):
                G_row.append( (dp.x ** p) * flip_sign )
            h.append(dp.y * flip_sign)
            G.append(G_row)

    # define eq constraints (A, b)
    for (dp, relopt) in pulled_constrs:
        if relopt.opt != '=': continue
        A_row = []
        for p in range(polydeg, -1, -1): A_row.append(dp.x ** p)
        b.append(dp.y)
        A.append(A_row)
    
    R = np.array(R); s = np.array(s)
    G = np.array(G); h = np.array(h)
    A = np.array(A); b = np.array(b)
    return solve_ls(R, s, G, h, A, b, solver='cvxopt', verbose=False)


def __fit_pulled_dataset(pulled_S:dataset.Dataset, pulled_constrs:list, unknown_stree:backprop.UnknownSyntaxTree, hist:History) -> bool:
    degree = 8
    dropout = 0.

    X = []
    Y = []
    #W = []
    for dp in pulled_S.data:
        #if dp.x >= 0 and dp.x <= 1 and dp.y >= 0 and dp.y <= 0.45: pass
        X.append(dp.x)
        Y.append(dp.y)
        #W.append(100000 - abs(dp.x))

    #P = np.poly1d( np.polyfit( np.array(X), np.array(Y), degree ) if len(pulled_S.data) > 0 else np.zeros(degree+1) )
    P = np.poly1d( __qp_solve( np.array(X), np.array(Y), pulled_constrs, degree ) if len(pulled_S.data) > 0 else np.zeros(degree+1) )

    nonzero_coeffs = degree + 1
    if dropout > 0.:
        c_std = (P.c - P.c.max()) / (P.c.max() - P.c.min())
        c_softmax = softmax(c_std)
        softmax_th = c_softmax.min() + ((c_softmax.max() - c_softmax.min()) * dropout)
        nonzero_coeffs = 0
        for i in range(P.c.size):
            if c_softmax[i] < softmax_th: P.c[i] = 0.
            else: nonzero_coeffs += 1
    
    unknown_stree.set_unknown_model(unknown_stree.label, P)
    simple_found = nonzero_coeffs <= (degree + 1) * 0.2
    hist.log_pull(unknown_stree, pulled_S, pulled_constrs, simple_found=simple_found)

    return simple_found


def jump_backprop(stree_d0:backprop.SyntaxTree, stree_d1:backprop.SyntaxTree, unknown_strees_labels:list, S:dataset.Dataset, maxiters:int=5, mesh:bool=None) -> bool:
    #n_unkn_strees = len(unknown_strees)
    #niters = min(n_unkn_strees, maxiters) if n_unkn_strees <= 1 else maxiters
    #last_unkn_stree_idx = -1
    hist = History()

    stree_d0.set_parent()
    stree_d1.set_parent()

    # fix all unknowns to x (simplest model)
    #for idx in range(n_unkn_strees):
    #    unknown_strees[idx].set_model( lambda x: x**6 + 2.83726e-11 )

    #for iter_idx in range(niters):
    for derivdeg in [0, 1]:
        stree = stree_d0
        if derivdeg == 1: stree = stree_d1

        for unkn_stree_label in unknown_strees_labels:
            for stree_derivdeg in [derivdeg]: # range(derivdeg+1):
                #unkn_stree_idx = random.choice([i for i in range(n_unkn_strees) if i != last_unkn_stree_idx])
                
                unknown_stree_fullname = unkn_stree_label + ("'" * stree_derivdeg)
                if stree.count_unknown_model(unknown_stree_fullname) != 1:
                    print(f"Cannot pull from {unknown_stree_fullname} for derivative {str(derivdeg)}: no unique occurence")
                    continue
                
                unknown_stree = stree.get_unknown_stree(unknown_stree_fullname)
                print(f"Pulling from {unknown_stree_fullname} for derivative {str(derivdeg)}")

                #
                # pull dataset from 'unknown_stree' 
                #
                pulled_S = dataset.Dataset()
                pulled_S.xl = S.xl
                pulled_S.xu = S.xu
                
                yl = 0; yu = 0

                if derivdeg == 0:
                    for dp in S.data:
                        stree.compute_output(dp.x)
                        pulled_y, _ = unknown_stree.pull_output(dp.y)
                        pulled_S.data.append( dataset.DataPoint(dp.x, pulled_y) )
                        yl = min(yl, pulled_y)
                        yu = max(yu, pulled_y)
                    
                    #pulled_S.yl = S.yl
                    #pulled_S.yu = S.yu
                    #pulled_S.minmax_scale_y()
                    pulled_S.yl = yl
                    pulled_S.yu = yu

                #
                # pull constraints from 'unknown_stree' 
                #
                pulled_constrs = []
                violated_constrs = []
                for (dp, relopt) in __get_constraints(S.knowledge, derivdeg, mesh=mesh):
                    stree.compute_output(dp.x)
                    try:
                        pulled_th, pulled_relopt = unknown_stree.pull_output(dp.y, relopt)
                        pulled_constrs.append( (dataset.DataPoint(dp.x, pulled_th), pulled_relopt) )
                    except backprop.PullViolation:
                        violated_constrs.append( (dataset.DataPoint(dp.x, dp.y), relopt) )

                #
                # fit pulled dataset and update the model of 'unknown_stree'
                #
                if len(violated_constrs) == 0:
                    __fit_pulled_dataset(pulled_S, pulled_constrs, unknown_stree, hist)  # when true, a 'simple' model was found
                else:
                    hist.log_pull(unknown_stree, pulled_S, pulled_constrs, violated_constrs, simple_found=False)

                #last_unkn_stree_idx = unkn_stree_idx
        
    return hist