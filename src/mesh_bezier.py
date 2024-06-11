import dataset
import numpy as np
import random
import bezier_spline


def __create_bezier_curves(S:dataset.Dataset, X:list, Y:list) -> bezier_spline.BezierCurveConnector:
    
    curve_conn = bezier_spline.BezierCurveConnector(S.xl, S.xu)
    npoints = len(X)
    nsplines = int( (npoints / 2) - 1 )
    for _ in range(nsplines):
        curve = bezier_spline.BezierCurve()
        curve_conn.connect(curve)
    
    chromo = []
    for idx in range(npoints):
        if idx > 0 and idx < npoints-1: chromo.append(X[idx])
        chromo.append(Y[idx])
    
    curve_conn.set_chromo(np.array(chromo))
    
    
    return curve_conn

def __compute_fitness(S:dataset.Dataset, X:list, Y:list, h:float) -> float:
    return -__create_bezier_curves(S, X, Y).fitness(S)

#def __compute_fitness(S:dataset.Dataset, X:list, Y:list, h:float, spline_idx:int) -> float:
#    return -__create_bezier_curves(S, X, Y).curves[spline_idx].fitness(S)


def compute_mesh(S:dataset.Dataset, nsplines:int=1, niters:int=10, ystep_ratio:float=0.01, kernel_ratio:float=0.1,
                 iter_callback=None):
    
    npoints = (4*nsplines) - (2*(nsplines-1))
    X = np.linspace(S.xl, S.xu, npoints).tolist()
    Y = [-2. for _ in range(npoints)]
    h = (S.xu - S.xl) / npoints
    ystep = (S.yu - S.yl) * ystep_ratio
    hkernel = (S.xu - S.xl) * kernel_ratio

    print(f"y step: {ystep}")
    print(f"h kernel: {hkernel}")

    history = {'totiter': 0, 'fit': []}
    
    for iter_idx in range(niters):
        X_idx = [i for i in range(npoints)]
        n_updates = 0

        while len(X_idx) > 0:

            __idx = random.randint(0, len(X_idx)-1)
            idx = X_idx[ __idx ]
            X_idx.pop( __idx )
            
            dy_up   = + ystep
            dy_down = - ystep

            y_fit = __compute_fitness(S, X, Y, h)
            
            Y_cpy = Y.copy()
            Y_cpy[idx] += dy_up
            y_up_fit = __compute_fitness(S, X, Y_cpy, h)
            
            Y_cpy = Y.copy()
            Y_cpy[idx] += dy_down
            y_down_fit =  __compute_fitness(S, X, Y_cpy, h)

            if min(y_up_fit, y_down_fit) < y_fit:
                Y[idx] += dy_up if y_up_fit < y_down_fit else dy_down
                n_updates += 1
        
        history['totiter'] += 1
        history['fit'].append( __compute_fitness(S, X, Y, h) )

        if n_updates == 0:
            print(f"No more updates after {iter_idx + 1} iterations.")
            break
        
        if iter_callback is not None:
            iter_callback(X, Y)

    return __create_bezier_curves(S, X, Y), history


def compute_mesh_single(S:dataset.Dataset, nsplines:int=1, niters:int=10, ystep_ratio:float=0.01, kernel_ratio:float=0.1,
                 iter_callback=None):
    
    npoints = (4*nsplines) - (2*(nsplines-1))
    X = np.linspace(S.xl, S.xu, npoints).tolist()
    Y = [-2. for _ in range(npoints)]
    h = (S.xu - S.xl) / npoints
    ystep = (S.yu - S.yl) * ystep_ratio
    hkernel = (S.xu - S.xl) * kernel_ratio

    print(f"y step: {ystep}")
    print(f"h kernel: {hkernel}")

    history = {'totiter': 0, 'fit': []}
    
    for spline_idx in range(nsplines):
        if spline_idx != 1: continue
        print(f"Fitting spline {spline_idx}")

        for iter_idx in range(niters):
            X_idx = [i for i in range(spline_idx*2, (spline_idx*2) + 4)]
            n_updates = 0

            while len(X_idx) > 0:

                __idx = random.randint(0, len(X_idx)-1)
                idx = X_idx[ __idx ]
                X_idx.pop( __idx )
                
                dy_up   = + ystep
                dy_down = - ystep

                y_fit = __compute_fitness(S, X, Y, h, spline_idx=spline_idx)
                
                Y_cpy = Y.copy()
                Y_cpy[idx] += dy_up
                y_up_fit = __compute_fitness(S, X, Y_cpy, h, spline_idx=spline_idx)
                
                Y_cpy = Y.copy()
                Y_cpy[idx] += dy_down
                y_down_fit =  __compute_fitness(S, X, Y_cpy, h, spline_idx=spline_idx)

                if min(y_up_fit, y_down_fit) < y_fit:
                    Y[idx] += dy_up if y_up_fit < y_down_fit else dy_down
                    n_updates += 1
            
            history['totiter'] += 1
            history['fit'].append( __compute_fitness(S, X, Y, h, spline_idx=spline_idx) )

            if n_updates == 0:
                print(f"No more updates after {iter_idx + 1} iterations.")
                break
            
            if iter_callback is not None:
                iter_callback(X, Y)

    return __create_bezier_curves(S, X, Y), history


import pygad
def compute_mesh_ga(S:dataset.Dataset, nsplines:int=1, niters:int=10, ystep_ratio:float=0.01, kernel_ratio:float=0.1,
                 iter_callback=None):
    
    npoints = (4*nsplines) - (2*(nsplines-1))
    X = np.linspace(S.xl, S.xu, npoints).tolist()
    Y = [-2. for _ in range(npoints)]
   
    curve_conn = bezier_spline.BezierCurveConnector(S.xl, S.xu)
    for _ in range(nsplines):
        curve = bezier_spline.BezierCurve()
        curve_conn.connect(curve)
    
    
    def fitness_func(ga_instance, solution, solution_idx):
        chromo = []
        for idx in range(npoints):
            if idx > 0 and idx < npoints-1: chromo.append(X[idx])
            chromo.append(solution[idx])
        curve_conn.set_chromo(np.array(chromo))
        return curve_conn.fitness(S)
    
    fitness_function = fitness_func

    num_generations = 50
    num_parents_mating = 4

    sol_per_pop = 8
    num_genes = npoints

    init_range_low = S.yl
    init_range_high = S.yu

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 10

    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)
    
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    chromo = []
    for idx in range(npoints):
        if idx > 0 and idx < npoints-1: chromo.append(X[idx])
        chromo.append(solution[idx])
    curve_conn.set_chromo(np.array(chromo))

    return curve_conn, ga_instance