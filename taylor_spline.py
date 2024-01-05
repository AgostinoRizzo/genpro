import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import pygad
import sympy
from qpsolvers import solve_qp

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
        self.deriv = [0. for _ in range(degree+1)]
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
            #sol = tspline_solver.solve(silent=silent)
            sol = tspline_solver.solve_qp(silent=silent)

            for d in sol.keys():
                order = int(str(d)[1])
                sol_d = sol[d]
                #if type(sol_d) not in [float, int] and len(sol_d.free_symbols) > 0:
                #    for d in sol_d.free_symbols: sol_d = sol_d.subs({d:2.}) 
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
    
    def compute_length(self) -> float:
        dx = (self.xu - self.xl) * 0.1
        x = self.xl + dx
        l = 0.
        prev_point = dataset.DataPoint(self.xl, self.y(self.xl))
        while x < self.xu:
            curr_point = dataset.DataPoint(x, self.y(x))
            l += prev_point.distance(curr_point)
            prev_point = curr_point
            x += dx
        return l

    def plot(self, show:bool=True):
        xl = -1 if self.xl is None else self.xl  # TODO: fix it (default)
        xu =  1 if self.xu is None else self.xu  # TODO: fix it (default)
        x = np.linspace(xl, xu, 100)
        plt.plot(x, self.y(x))
        plt.ylim(-2, 2) 
        if show: plt.show()


class TaylorSplineSolver:
    def __init__(self, tspline:TaylorSpline, S:dataset.Dataset) -> None:
        self.tspline = tspline
        self.S = S
      
    def solve(self, silent:bool=False) -> dict:
        degree = self.tspline.get_degree()
        unknowns = sympy.symbols(f"d0:{degree}")
        x = sympy.symbols('x')
        
        #
        # build fitness function from dataset.
        #
        fit_expr = self.__build_fitexpr(unknowns, silent)
        if type(fit_expr) == float: return {}  # TODO: generalize type check (needed when no data points).
       
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
        #if not silent: print('Simplifying fit expression...')
        #fit_expr = sympy.simplify(fit_expr)
        #print(str(fit_expr))
        
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
        if not silent: print("Result: " + str(res) + "\n")
        return res
    
    def solve_qp(self, silent:bool=False) -> dict:
        
        degree = self.tspline.get_degree()
        unknowns = sympy.symbols(f"d0:{degree}")
        zero_unknowns = {}
        for n in range(degree): zero_unknowns[unknowns[n]] = 0.

        x = sympy.symbols('x')
        if not silent: print('Degree: ' + str(degree) + '; Unknowns: ' + str(unknowns))
        
        P = []
        q = []
        A = []
        b = []
        lb = [-np.inf for _ in range(degree)]
        ub = [+np.inf for _ in range(degree)]

        #
        # build fitness matrix from dataset.
        #
        P, q = self.__build_fitexpr_matrix(unknowns, silent)
        # TODO: generalize size (0) check (needed when no data points).
       
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
        # add fixed derivatives contraints.
        #        
        for didx in self.tspline.fixed_deriv.keys():
            fix_deriv_eq = sympy.simplify(unknowns[didx] - self.tspline.fixed_deriv[didx])
            
            A_row = []
            ones = 0
            zeros = 0
            one_n = -1
            for n in range(degree):
                coeff = fix_deriv_eq.coeff(unknowns[n])
                A_row.append(coeff)
                if coeff == 1.: ones += 1; one_n = n
                elif coeff == 0.: zeros += 1
            
            b_term = -fix_deriv_eq.subs(zero_unknowns)
            if ones == 1 and zeros == degree-1: lb[one_n] = b_term; ub[one_n] = b_term
            else:
                A.append(A_row)
                b.append(b_term)
        
        #
        # solve quadprog.
        #
        if not silent: print("Solving quadprog...")
        
        if len(A) == 0: A = None; b = None
        else: A = np.array(A, dtype=np.double); b = np.array(b, dtype=np.double)
        lb = np.array(lb, dtype=np.double)
        ub = np.array(ub, dtype=np.double)
        G = None
        h = None
        
        sol = solve_qp(P, q, A=A, b=b, lb=lb, ub=ub, solver="cvxopt", verbose=False)
        sol_map = {}
        for n in range(degree):
            sol_map[unknowns[n]] = sol[n]
        
        if not silent: print(f"QP solution: {sol_map}")
        return sol_map
    
    def __build_fitexpr(self, unknowns:list, silent:bool=True):
        degree = self.tspline.get_degree()
        if not silent: print("Building fitness function from dataset...")
        fit_expr = 0.
        for dp in self.S.data:
            if dp.x < self.tspline.xl or dp.x > self.tspline.xu: continue
            
            point_fit_expr = 0.
            for n in range(degree):
                point_fit_expr += (unknowns[n] / math.factorial(n)) * ((dp.x - self.tspline.x0) ** n)
            
            point_fit_expr = (point_fit_expr - dp.y) ** 2

            fit_expr += point_fit_expr

        if not silent: print('Degree: ' + str(degree) + '; Unknowns: ' + str(unknowns))
        return fit_expr     

    def __build_fitexpr_matrix(self, unknowns:list, silent:bool=True) -> (list, list):
        degree = self.tspline.get_degree()
        if not silent: print("Building fitness matrix from dataset...")
        
        M = []
        b = []

        for dp in self.S.data:
            if dp.x < self.tspline.xl or dp.x > self.tspline.xu: continue
            
            M_row = []
            for n in range(degree):
                M_row.append( (1. / math.factorial(n)) * ((dp.x - self.tspline.x0) ** n) )
            M.append(M_row)

            b.append(dp.y)

        M = np.array(M, dtype=np.double)
        Q = np.dot(M.T, M)
        q = -np.dot(M.T, np.array(b, dtype=np.double))
        
        return Q, q        
            

class TaylorSplineConnector:
    
    def fit(self, S:dataset.Dataset, spline_degree:int, silent:bool=True) -> list:
        exp_radius = (S.xu - S.xl) * 0.2 #0.2 #0.05  # TODO: fix it or as hyper-parameter
        exp_span = 0.4
        print(f"ExpRadius = {exp_radius}")

        x0 = (S.xu + S.xl) / 2. # TODO: fix it or as hyper-parameter
        tsplines = []

        #
        # fit root spline
        #
        tspline_root, exp_radius_actual_root = TaylorSplineConnector.__fit_tspline(spline_degree+1, x0, exp_radius, exp_span, S, side='all', silent=silent)
        tspline_root.xl = x0 - exp_radius_actual_root * exp_span  # TODO: remove it (redundant, alreadt in __fit_tspline)
        tspline_root.xu = x0 + exp_radius_actual_root * exp_span  # TODO: remove it (redundant, alreadt in __fit_tspline)

        #
        # expand to the right
        #
        x0_root = x0
        x0 += exp_radius_actual_root * exp_span  # TODO: fix it or hyper-parameter
        join_y = tspline_root.y(x0)
        join_deriv = tspline_root.y_prime(x0)
        tsplines.append(tspline_root)


        while x0 < S.xu:
            tspline, exp_radius_actual = TaylorSplineConnector.__fit_tspline(spline_degree, x0, exp_radius, exp_span, 
                                                                             S, side='right', join_y=join_y, join_deriv=join_deriv, 
                                                                             silent=silent)
            tspline.xl = x0
            tspline.xu = x0 + exp_radius_actual * exp_span

            x0 += exp_radius_actual * exp_span  # TODO: fix it or hyper-parameter
            join_y = tspline.y(x0)
            join_deriv = tspline.y_prime(x0)
            tsplines.append(tspline)
        
        #
        # expand to the left
        #
        x0 = x0_root - exp_radius_actual_root * exp_span  # TODO: fix it or hyper-parameter
        join_y = tspline_root.y(x0)
        join_deriv = tspline_root.y_prime(x0)

        while x0 > S.xl:  # expand to the left
            tspline, exp_radius_actual = TaylorSplineConnector.__fit_tspline(spline_degree, x0, exp_radius, exp_span, 
                                                                             S, side='left', join_y=join_y, join_deriv=join_deriv,
                                                                             silent=silent)
            tspline.xl = x0 - exp_radius_actual * exp_span
            tspline.xu = x0

            x0 -= exp_radius_actual * exp_span  # TODO: fix it or hyper-parameter
            join_y = tspline.y(x0)
            join_deriv = tspline.y_prime(x0)
            tsplines.append(tspline)
        
        return tsplines
    
    def __fit_tspline(spline_degree:int,
                      x0:float, exp_radius:float, 
                      exp_span:float,
                      S:dataset.Dataset,
                      side:str='all',
                      join_y:float=None,
                      join_deriv:float=None,
                      silent:bool=False) -> (TaylorSpline, float):
        exp_length = exp_radius * 2.
        exp_radius_actual = exp_radius
        tspline_length = 0
        tspline = None
        length_reached = False
        increase_length = None

        while not length_reached:
            tspline = None
            if   side == 'all':   tspline = TaylorSpline(x0, spline_degree, x0-exp_radius_actual, x0+exp_radius_actual)
            elif side == 'right': tspline = TaylorSpline(x0, spline_degree, x0, x0+exp_radius_actual)
            elif side == 'left':  tspline = TaylorSpline(x0, spline_degree, x0-exp_radius_actual, x0)
            else: raise RuntimeError('Invalid side.')
            
            if join_y is not None and join_deriv is not None:
                tspline.fix_deriv(0, join_y)
                tspline.fix_deriv(1, join_deriv)
            
            print(f"Fitting on x0 = {x0} to [{x0-exp_radius_actual}, {x0+exp_radius_actual}]")
            tspline.intersect(S.knowledge.derivs, side='all')
            tspline.fit(S, silent=silent)
            
            tspline_length = tspline.compute_length()

            if tspline_length == 0.:
                exp_radius_actual = exp_radius * 0.2
                increase_length = True
                continue
            
            length_reached = True
            if tspline_length < exp_length and (increase_length is None or increase_length):
                exp_radius_actual += exp_radius * 0.2
                length_reached = False
                increase_length = True
            elif tspline_length > exp_length and (increase_length is None or not increase_length):
                exp_radius_actual -= exp_radius * 0.2
                length_reached = False
                increase_length = False

            if tspline.xl < S.xl or tspline.xu > S.xu:
                length_reached = True
            
            length_reached = True


        tspline.xl = x0 - exp_radius_actual * exp_span
        tspline.xu = x0 + exp_radius_actual * exp_span

        return tspline, exp_radius_actual
