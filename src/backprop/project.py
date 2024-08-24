import numpy as np
from qpsolvers import solve_ls


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