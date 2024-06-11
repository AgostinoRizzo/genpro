import dataset
import numpy as np
import random


def __compute_fitness(S:dataset.Dataset, X:list, Y:list, h:float) -> float:
    
    M = []
    C = []
    for idx in range(len(X)-1):
        x1 = X[idx]
        x2 = X[idx+1]
        y1 = Y[idx]
        y2 = Y[idx+1]

        m = (y2-y1) / (x2-x1)
        c = y1 - (m*x1)
        M.append(m)
        C.append(c)

    fit = 0.

    for dp in S.data:
        idx = int( (dp.x - S.xl) // h )
        if idx >= len(M): idx = len(M)-1
        y = (M[idx] * dp.x) + C[idx]
        fit += (dp.y - y) ** 2
    
    return fit / len(S.data)


def __is_peak(Y:list, idx:int) -> bool:
    last_idx = len(Y) - 1
    if idx == 0 or idx == last_idx: return False
    return (Y[idx] < Y[idx-1] and Y[idx] < Y[idx+1]) or (Y[idx] > Y[idx-1] and Y[idx] > Y[idx+1])

def __compute_peaks(Y:list) -> int:
    npeaks = 0
    for idx in range(1, len(Y)-1):
        if __is_peak(Y, idx):
            npeaks += 1
    return npeaks

def __compute_peaks_fitness(Y:list, idx:int=None) -> float:
    fit = 0.
    if idx is not None:
        if idx == 0 or idx == len(Y) - 1 or not __is_peak(Y, idx): return 0
        return max( (Y[idx] - Y[idx-1])**2, (Y[idx] - Y[idx+1])**2 )
    for idx in range(1, len(Y)-1):
        #if __is_peak(Y, idx):
        fit += max( (Y[idx] - Y[idx-1])**2, (Y[idx] - Y[idx+1])**2 )
    return fit


def __smooth_mesh(X:list, Y:list, niters:int):

    for _ in range(niters):
        X_sample = X.copy()
        while len(X_sample) > 0:

            idx = random.randint(0, len(X_sample)-1)
            X_sample.pop(idx)
            if idx == 0 or idx == len(X)-1: continue

            right_dy = Y[idx+1] - Y[idx]
            left_dy  = Y[idx]   - Y[idx-1]

            #if right_dy * left_dy >= 0:
            Y[idx] = (Y[idx-1] + Y[idx+1]) / 2.


def __Kernel(x:float, x0:float, h:float) -> float:
        
        if x < x0-h or x > x0+h: return 0.
        
        u = (x - x0) / h
        u_abs = abs(u)

        # Tukey's tri-weight kernel function
        #if u_abs > 1: return 0.
        #return (1- (u_abs ** 3)) ** 3

        # Triweight kernel function
        if u_abs > 1: return 0.
        return (35/32) * (1 - (u ** 2)) ** 3

def __move_point(X:list, Y:list, idx0:int, dy:float, hkernel:float, smooth:bool):
    
    y0_old = Y[idx0]
    y0_new = Y[idx0] + ( dy * __Kernel(X[idx0], X[idx0], hkernel) )
    Y[idx0] = y0_new

    if smooth: return

    for step in [-1, 1]:
        idx = idx0 + step
        y_new = y0_new

        while idx >= 0 and idx < len(X):
            kernel_scale = __Kernel(X[idx], X[idx0], hkernel)
            if kernel_scale == 0.: break

            dy_curr = dy #max(0., y_new - Y[idx]) if dy > 0 else min(0., y_new - Y[idx])
            if dy_curr == 0.: break

            Y[idx] += dy_curr * kernel_scale
            y_new = Y[idx]

            idx += step

        #or (dy > 0 and Y[__idx] > Y[idx0]) or (dy < 0 and Y[__idx] < Y[idx0]): continue
        #dy_actual = dy
        #if __idx != idx0:
        #    
        #Y[__idx] += dy_actual * kernel_scale


def compute_mesh(S:dataset.Dataset, npoints:int=10, niters:int=10, ystep_ratio:float=0.01, kernel_ratio:float=0.1,
                 iter_callback=None):
    
    X = np.linspace(S.xl, S.xu, npoints).tolist()
    Y = [-1.5 for _ in range(len(X))]
    h = (S.xu - S.xl) / npoints
    ystep = (S.yu - S.yl) * ystep_ratio
    hkernel = (S.xu - S.xl) * kernel_ratio

    print(f"y step: {ystep}")
    print(f"h kernel: {hkernel}")

    smooth = False
    history = {'totiter': 0, 'fit': [], 'peak': [], 'peaks_fit': []}
    
    for iter_idx in range(niters):
        X_idx = [i for i in range(len(X))]
        n_updates = 0

        while len(X_idx) > 0:

            __idx = random.randint(0, len(X_idx)-1)
            idx = X_idx[ __idx ]
            X_idx.pop( __idx )
            
            y = Y[idx]
            dy_up   = + ystep
            dy_down = - ystep

            y_fit = __compute_peaks_fitness(Y, idx=idx) if smooth else __compute_fitness(S, X, Y, h)
            
            Y_cpy = Y.copy()
            __move_point(X, Y_cpy, idx, dy_up, hkernel, smooth)
            y_up_fit = __compute_peaks_fitness(Y_cpy, idx=idx) if smooth else __compute_fitness(S, X, Y_cpy, h)
            
            Y_cpy = Y.copy()
            __move_point(X, Y_cpy, idx, dy_down, hkernel, smooth)
            y_down_fit = __compute_peaks_fitness(Y_cpy, idx=idx) if smooth else __compute_fitness(S, X, Y_cpy, h)
            
            if idx == 2 and smooth:
                print(f"{y_fit}, {y_up_fit}, {y_down_fit}")

            if min(y_up_fit, y_down_fit) < y_fit:
                __move_point(X, Y, idx, dy_up if y_up_fit < y_down_fit else dy_down, hkernel, smooth)
                n_updates += 1
        
        history['totiter'] += 1
        history['fit'].append( __compute_fitness(S, X, Y, h) )
        history['peak'].append( __compute_peaks(Y) )
        history['peaks_fit'].append( __compute_peaks_fitness(Y) )

        if n_updates == 0:
            print(f"No more updates after {iter_idx + 1} iterations.")
            """print("Reshuffling...")

            for idx in range(len(Y)):
                Y[idx] += ystep * random.choice( [0, -1, 1] )
            """

            #if smooth: break
            #print("Smoothing...")
            #smooth = True

            break
        
        if iter_callback is not None:
            iter_callback(X, Y)
    
    #__smooth_mesh(X, Y, 10)

    return X, Y, history