import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import pygad
import sympy

import dataset


_S = None
_tspline = None
_function_inputs = None
_silent = None

def _fitfunc(ga_instance, solution, solution_idx):
    _tspline.set_chromo(solution)
    return _tspline.fitness(_S)

def _crossover_func(parents, offspring_size, ga_instance):  # single point crossover
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        random_split_point = np.random.choice(range(offspring_size[1]))

        parent1[random_split_point:] = parent2[random_split_point:]

        _tspline.set_chromo(np.array(parent1))
        offspring.append(parent1)
        
        idx += 1

    return np.array(offspring)

def _mutation_func(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        random_gene_idx = np.random.choice(range(offspring.shape[1]))

        delta = np.random.uniform(-1., 1.)  # TODO: fix it!
        offspring[chromosome_idx][random_gene_idx] += delta
        
    return offspring

def _create_initial_population(size:int) -> np.array:
    pop = []
    chromo_len = _tspline.get_chromo_length()
    for _ in range(size):
        
        chromo = []
        for _ in range(chromo_len):
            chromo.append(np.random.uniform(-10, 10))  # TODO: fix it!))
        
        pop.append(chromo)
    
    return np.array(pop)


def _callback_generation(ga_instance):
    global _silent
    if _silent: return
    clear_output(wait=True)
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    #print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


def _init_ga():
    fitness_function = _fitfunc

    num_generations = 500
    num_parents_mating = 4

    sol_per_pop = 100
    num_genes = _function_inputs

    initial_population = _create_initial_population(sol_per_pop)

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = _crossover_func

    mutation_type = _mutation_func
    mutation_percent_genes = 1

    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       initial_population=initial_population,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=_callback_generation)
    
    return ga_instance

def _ga_output(ga_instance):
    global _silent
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    if not _silent:
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=math.sqrt(sys.maxsize-solution_fitness)))

        prediction = np.sum(np.array(_function_inputs)*solution)
        print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

        ga_instance.plot_fitness();

    return solution


class TaylorSpline:
    def __init__(self, x0:float, degree:int=1, xl:float=None, xu:float=None) -> None:
        self.x0 = x0
        self.deriv = [0 for _ in range(degree+1)]
        self.xl = xl
        self.xu = xu
        self.fixed_deriv = {}
        self.inters_derivs = {}
    
    def set_deriv(self, deriv:list):
        self.deriv = deriv
    
    def get_degree(self) -> int:
        return len(self.deriv)

    def y(self, x:float) -> float:
        degree = len(self.deriv)
        y = 0.
        for n in range(degree):
            y += (self.deriv[n] / math.factorial(n)) * ((x - self.x0) ** n)
        
        return y
    
    def y_prime(self, x:float) -> float:
        degree = len(self.deriv)
        y_prime = 0.
        for n in range(1, degree):
            y_prime += (self.deriv[n] / math.factorial(n)) * n * ((x - self.x0) ** (n-1))
        
        return y_prime
    
    def fix_deriv(self, degree:int, value:float):
        self.fixed_deriv[degree] = value
        self.deriv[degree] = value
    
    def intersect(self, derivs:dict, side:str='all'):
        xl = self.xl if side == 'left'  or side == 'all' else self.x0  # TODO: check/ignore intersection on expansion point (or to the left).
        xu = self.xu if side == 'right' or side == 'all' else self.x0  # TODO: check/ignore intersection on expansion point (or to the left).
        for xy_order, xy_lst in derivs.items():
            for xy in xy_lst:
                if xy.x >= xl and xy.x <= xu:
                    if xy_order not in self.inters_derivs.keys(): self.inters_derivs[xy_order] = []
                    self.inters_derivs[xy_order].append(xy)

    def fitness(self, S:dataset.Dataset) -> float:
        sse = 0.
        with_limits = self.xl is not None and self.xu is not None
        for dp in S.data:
            if not with_limits or (with_limits and dp.x >= self.xl and dp.x <= self.xu):
                sse += (self.y(dp.x) - dp.y) ** 2
        return -sse

    def get_chromo_length(self) -> int:
        return len(self.deriv) - len(self.fixed_deriv.keys())

    def set_chromo(self, chromo:np.array):
        chromo = chromo.tolist()
        chromo_idx = 0
        for didx in range(len(self.deriv)):
            if didx not in self.fixed_deriv.keys():
                self.deriv[didx] = chromo[chromo_idx]
                chromo_idx += 1

    def fit(self, S:dataset.Dataset, silent:bool=False, method:str='linsys'):
        if method == 'ga':
            global _S
            global _tspline
            global _function_inputs
            global _silent

            _S = S
            _tspline = self
            _function_inputs = self.get_chromo_length()
            _silent = silent
            ga_instance = _init_ga()
            ga_instance.run()
            solution = _ga_output(ga_instance)
            self.set_chromo(solution)
        
        elif method == 'linsys':
            tspline_solver = TaylorSplineSolver(self, S)
            sol = tspline_solver.solve(silent=silent)

            for d in sol.keys():
                order = int(str(d)[1])
                sol_d = sol[d]
                if type(sol_d) not in [float, int] and len(sol_d.free_symbols) > 0:
                    for d in sol_d.free_symbols: sol_d = sol_d.subs({d:2.}) 
                self.deriv[order] = float(sol_d)
            
            for didx in range(len(self.deriv)-1, -1, -1):
                if type(self.deriv[didx]) != float and type(self.deriv[didx]) != int:  # TODO: check sympy type
                    sol_d = self.deriv[didx].subs(sol)
                    if type(sol_d) not in [float, int]:
                        for d in sol_d.free_symbols: sol_d = sol_d.subs({d:2.}) 
                    self.deriv[didx] = float(sol_d)
                    sol[sympy.symbols(f"d{didx}")] = self.deriv[didx]
        
        else:
            raise RuntimeError('Invalid fitting method.')

    def plot(self, show:bool=True):
        xl = -5 if self.xl is None else self.xl  # TODO: fix it (default)
        xu = -5 if self.xu is None else self.xu  # TODO: fix it (default)
        x = np.linspace(xl, xu, 100)
        plt.plot(x, self.y(x))
        plt.ylim(-2, 2) 
        if show: plt.show()


class TaylorSplineSolver:
    def __init__(self, tspline:TaylorSpline, S:dataset.Dataset) -> None:
        self.tspline = tspline
        self.S = S
    
    def solve(self, silent:bool=False):
        degree = self.tspline.get_degree()
        unknowns = sympy.symbols(f"d0:{degree}")
        x = sympy.symbols('x')
        
        #
        # build fitness function from dataset.
        #
        fit_expr = 0.
        for dp in self.S.data:
            if dp.x < self.tspline.xl or dp.x > self.tspline.xu: continue
            
            point_fit_expr = 0.
            for n in range(degree):
                point_fit_expr += (unknowns[n] / math.factorial(n)) * ((dp.x - self.tspline.x0) ** n)
            point_fit_expr = (point_fit_expr - dp.y) ** 2

            fit_expr += point_fit_expr

        if not silent: print('Degree: ' + str(degree) + '; Unknowns: ' + str(unknowns))
       

        #
        # fix derivatives (on x0) for intersection derivatives.
        #
        inters_eqs = []
        inters_derivs = []
        for xy_order, xy_lst in self.tspline.inters_derivs.items():
            for xy in xy_lst:
                
                f_x = 0.
                for n in range(degree):
                    f_x += (unknowns[n] / math.factorial(n)) * ((x - self.tspline.x0) ** n)
                for _ in range(xy_order):
                    f_x = sympy.diff(f_x, x)
                f_x = f_x.subs({x: xy.x})
                
                fixed = False

                for n in range(degree):  # TODO: try/check derivative order.
                    if n in self.tspline.fixed_deriv.keys() or n in inters_derivs: continue

                    for n_fixed in self.tspline.fixed_deriv.keys():
                        f_x = f_x.subs({unknowns[n_fixed]: self.tspline.fixed_deriv[n_fixed]})
                    eq = sympy.solve(f_x - xy.y, unknowns[n])[0] - unknowns[n]

                    inters_eqs.append(eq)
                    inters_derivs.append(n)

                    fixed = True
                    break

                if not fixed:
                    total_points = 0
                    for _, xy_lst in self.tspline.inters_derivs.items(): total_points += len(xy_lst)
                    raise RuntimeError(f"Cannot intersect {total_points} points with spline of order {degree-1}.")
        
        res = sympy.solve(inters_eqs, *unknowns)  # TODO: manage no solution
        for d in inters_derivs:
            if unknowns[d] in res.keys():  # when independent only (TODO: check it)
                self.tspline.fix_deriv(d, res[unknowns[d]])
                

        #
        # substitute fixed derivatives.
        #
        fixed_deriv_map = {}

        for didx in self.tspline.fixed_deriv.keys():
            fixed_deriv_map[unknowns[didx]] = self.tspline.fixed_deriv[didx]
        
        unknowns_origin = unknowns
        unknowns = []

        for n in range(degree):
            if n not in self.tspline.fixed_deriv.keys():
                unknowns.append(unknowns_origin[n])
        
        fit_expr = fit_expr.subs(fixed_deriv_map)

        #
        # generate equations.
        #
        if not silent: print("Generating equations...")
        eqs = []
        for d in unknowns:
            eqs.append(sympy.diff(fit_expr, d))

        if len(eqs) == 0:  # all derivatives are fixed (len(self.fixed_deriv) == len(deriv)).
            if not silent: print('All derivatives already fixed (unique/trivial solution).')
            res = {}
            for d in self.tspline.fixed_deriv.keys():
                res[unknowns_origin[d]] = self.tspline.fixed_deriv[d]
            return res
        
        if not silent: print('Solving...')
        
        res = sympy.solve(eqs, *unknowns)  # TODO: manage no solution
        if not silent: print("Result: " + str(res))
        return res
            
            

class TaylorSplineConnector:
    
    def fit(self, S:dataset.Dataset, spline_degree:int) -> list:
        exp_radius = (S.xu - S.xl) * 0.2  # TODO: fix it or as hyper-parameter
        print(f"ExpRadius = {exp_radius}")

        x0 = 0. #(S.xu + S.xl) / 2. # TODO: fix it or as hyper-parameter
        tsplines = []

        #
        # fit root spline
        #
        tspline_root = TaylorSpline(x0, spline_degree, x0-exp_radius, x0+exp_radius)
        print(f"Fitting root on x0 = {x0} to [{x0-exp_radius}, {x0+exp_radius}]")
        tspline_root.intersect(S.knowledge.derivs, side='all')
        tspline_root.fit(S, silent=True)
        tspline_root.xl = x0 - exp_radius * 0.8
        tspline_root.xu = x0 + exp_radius * 0.8

        #
        # expand to the right
        #
        x0_root = x0
        x0 += exp_radius * 0.8  # TODO: fix it or hyper-parameter
        join_y = tspline_root.y(x0)
        join_deriv = tspline_root.y_prime(x0)
        tsplines.append(tspline_root)

        
        while x0 < S.xu:
            tspline = TaylorSpline(x0, spline_degree, x0-exp_radius, x0+exp_radius)
            if join_y is not None:
                tspline.fix_deriv(0, join_y)
                tspline.fix_deriv(1, join_deriv)
            
            print(f"Fitting (to right) on x0 = {x0} to [{x0-exp_radius}, {x0+exp_radius}]")
            tspline.intersect(S.knowledge.derivs, side='right')
            tspline.fit(S, silent=True)
            tspline.xl = x0
            tspline.xu = x0 + exp_radius * 0.8

            x0 += exp_radius * 0.8  # TODO: fix it or hyper-parameter
            join_y = tspline.y(x0)
            join_deriv = tspline.y_prime(x0)
            tsplines.append(tspline)
        
        #
        # expand to the left
        #
        x0 = x0_root - exp_radius * 0.8  # TODO: fix it or hyper-parameter
        join_y = tspline_root.y(x0)
        join_deriv = tspline_root.y_prime(x0)

        while x0 > S.xl:  # expand to the left
            tspline = TaylorSpline(x0, spline_degree, x0-exp_radius, x0+exp_radius)
            if join_y is not None:
                tspline.fix_deriv(0, join_y)
                tspline.fix_deriv(1, join_deriv)
            
            print(f"Fitting (to left) on x0 = {x0} to [{x0-exp_radius}, {x0+exp_radius}]")
            tspline.intersect(S.knowledge.derivs, side='left')
            tspline.fit(S, silent=True)
            tspline.xl = x0 - exp_radius * 0.8
            tspline.xu = x0

            x0 -= exp_radius * 0.8  # TODO: fix it or hyper-parameter
            join_y = tspline.y(x0)
            join_deriv = tspline.y_prime(x0)
            tsplines.append(tspline)
        
        return tsplines

