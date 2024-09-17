import numpy as np
from qpsolvers import solve_ls
import random

from backprop import backprop, gp


def __add_sign_constrs(sign_constrs:dict, data, G, h):
    nvars = data.y.size

    for deriv, constrs in sign_constrs.items():
        derivdeg = len(deriv)

        if derivdeg == 0:
            for (l,u,sign,th) in constrs:
                
                for j, x in enumerate(data.X):
                    if (x >= l).all() and (x <= u).all():
                        G_row = np.zeros(nvars)
                        G_row[j] = -1.0 if sign == '+' else 1.0

                        G.append(G_row)
                        h.append(th)
        
        elif derivdeg == 1:
            for (l,u,sign,th) in constrs:
                
                for i, x_i in enumerate(data.X):
                    if (x_i < l).any() or (x_i > u).any(): continue

                    for j, x_j in enumerate(data.X):
                        if (x_j < l).any() or (x_j > u).any() or x_j <= x_i: continue

                        G_row = np.zeros(nvars)
                        if sign == '+':
                            G_row[i] =  1.0
                            G_row[j] = -1.0
                        else:
                            G_row[i] = -1.0
                            G_row[j] =  1.0
                        
                        G.append(G_row)
                        h.append(th)
        
        elif derivdeg == 2:
            for (l,u,sign,th) in constrs:
                
                for i, x_i in enumerate(data.X):
                    if (x_i < l).any() or (x_i > u).any(): continue

                    for j, x_j in enumerate(data.X):
                        if (x_j < l).any() or (x_j > u).any() or x_j <= x_i: continue

                        for k, x_k in enumerate(data.X):
                            if (x_k < l).any() or (x_k > u).any() or x_k <= x_j: continue

                            d_ij = abs(x_i - x_j)
                            d_jk = abs(x_j - x_k)

                            G_row = np.zeros(nvars)
                            if sign == '+':
                                G_row[i] = -1.0 / d_ij
                                G_row[j] =  (1.0 / d_ij) + (1.0 / d_jk)
                                G_row[k] = -1.0 / d_jk
                            else:
                                G_row[i] = 1.0 / d_ij
                                G_row[j] = (-1.0 / d_ij) + (-1.0 / d_jk)
                                G_row[k] = 1.0 / d_jk
                            
                            G.append(G_row)
                            h.append(th)


def __add_symm_constrs(symm_constrs:dict, data, A, b):
    nvars = data.y.size
    
    for deriv, (x_0, iseven) in symm_constrs.items():
        derivdeg = len(deriv)

        if derivdeg == 0:
            for i, x_i in enumerate(data.X):
                    if (x_i > x_0).any(): continue
                    
                    x_j_val = x_0 + (x_0 - x_i)

                    for j, x_j in enumerate(data.X):
                        if i == j or x_j != x_j_val: continue

                        A_row = np.zeros(nvars)
                        if iseven:
                            A_row[i] =  1.0
                            A_row[j] = -1.0
                        else:
                            A_row[i] = 1.0
                            A_row[j] = 1.0
                        
                        A.append(A_row)
                        b.append(0.0)


def __add_symm_datapts(data, symm_constrs:dict):
    X_symm = []
    y_symm = []

    for deriv, (x_0, iseven) in symm_constrs.items():
        derivdeg = len(deriv)

        if derivdeg == 0:
            for i, x in enumerate(data.X):
                
                x_symm = (x_0 + (x_0 - x)) if x < x_0 else (x_0 - (x - x_0))
                if x_symm in data.X: continue

                X_symm.append(x_symm)
                y_symm.append( data.y[i] if iseven else -data.y[i] )
    
    data.X = np.append(data.X, X_symm)
    data.y = np.append(data.y, y_symm)


def project(data, know):
    
    #
    # add symmetric data points
    #
    __add_symm_datapts(data, know.symm)

    nvars = data.y.size

    R = np.identity(nvars)
    s = data.y

    G = []
    h = []
    A = []
    b = []

    #
    # add constraints
    #
    __add_sign_constrs(know.sign, data, G, h)
    __add_symm_constrs(know.symm, data, A, b)

    G = np.array(G) if len(G) > 0 else None
    h = np.array(h) if len(h) > 0 else None
    A = np.array(A) if len(A) > 0 else None
    b = np.array(b) if len(b) > 0 else None

    y_proj = solve_ls(R, s, G, h, A, b, solver="osqp")
    data.y = y_proj


def project_semantic(sem, data, know):

    # sign constraints.
    for (l,u,sign,th) in know.sign[()]:
        
        if sign == '+':
            sem[ (data.X >= l).flatten() & (data.X <= u).flatten() & (sem <= th) ] = th + 1e-8
        else:
            sem[ (data.X >= l).flatten() & (data.X <= u).flatten() & (sem >= th) ] = th - 1e-8
    
    return sem


class Projector:
    def __init__(self, lib, know):
        self.lib = lib
        self.know = know
    
    def project(self, stree):
        y = stree(self.lib.data.X)
        proj_sem = project_semantic(y, self.lib.data, self.know)
        print(proj_sem)

        nodesCollector = backprop.SyntaxTreeNodeCollector()
        stree.accept(nodesCollector)

        backprop_nodes = []
        for n in nodesCollector.nodes:
            if backprop.SyntaxTree.is_invertible_path(n):
                backprop_nodes.append(n)

        if len(backprop_nodes) == 0:
            return stree
        
        cross_node = random.choice(backprop_nodes)

        stree.set_parent()

        pulled_y, _ = cross_node.pull_output(proj_sem)
        if (pulled_y == 0).all(): return stree

        sat_pulled_y = not np.isnan(pulled_y).any() and not np.isinf(pulled_y).any()
        if not sat_pulled_y: return stree

        new_sub_stree = self.lib.query(pulled_y)
        if new_sub_stree is None: return stree

        origin_stree = stree.clone()
        new_stree = gp.replace_subtree(stree, cross_node, new_sub_stree)
        if new_stree.get_max_depth() > 5:  # TODO: lookup based on max admissible depth.
            return origin_stree
        
        return new_stree