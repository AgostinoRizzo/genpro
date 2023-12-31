import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
import pygad

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

    num_generations = 300
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
        self.inters_point:dataset.DataPoint() = None
    
    def set_deriv(self, deriv:list):
        self.deriv = deriv

    def y(self, x:float) -> float:
        degree = len(self.deriv)
        y = 0.

        if self.inters_point is None:
            for n in range(degree):
                y += (self.deriv[n] / math.factorial(n)) * ((x - self.x0) ** n)
        else:
            y += self.inters_point.y
            for n in range(1, degree):
                y -= (self.deriv[n] / math.factorial(n)) * ((self.inters_point.x - self.x0) ** n)
            for n in range(1, degree):
                y += (self.deriv[n] / math.factorial(n)) * ((x - self.x0) ** n)
        return y
    
    def set_inters_point(self, x:float, y:float):
        self.inters_point = dataset.DataPoint(x, y)
    
    def fitness(self, S:dataset.Dataset) -> float:
        sse = 0.
        with_limits = self.xl is not None and self.xu is not None
        for dp in S.data:
            if not with_limits or (with_limits and dp.x >= self.xl and dp.x <= self.xu):
                sse += (self.y(dp.x) - dp.y) ** 2
        return -sse

    def get_chromo_length(self) -> int:
        return len(self.deriv)

    def set_chromo(self, chromo:np.array):
        self.deriv = chromo.tolist()

    def fit(self, S:dataset.Dataset, silent:bool=False):
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

    def plot(self, show:bool=True):
        xl = -5 if self.xl is None else self.xl  # TODO: fix it (default)
        xu = -5 if self.xu is None else self.xu  # TODO: fix it (default)
        x = np.linspace(xl, xu, 100)
        plt.plot(x, self.y(x), color='red')
        if show: plt.show()


class TaylorSplineConnector:
    def __init__(self, steps:float) -> None:
        self.steps = steps
    
    def fit(self, S:dataset.Dataset) -> list:
        stepsize = (S.xu - S.xl) / self.steps
        stepsize_2 = stepsize / 2.
        padding = 0.8
        print(f"Stepsize = {stepsize}; stepsize_2 = {stepsize_2}; padding = {padding}")

        x0 = S.xl + stepsize_2
        join_x = None
        join_y = None
        tsplines = []

        while x0 <= S.xu:
            tspline = TaylorSpline(x0, 3, x0-stepsize_2, x0+stepsize_2)
            if join_x is not None:
                tspline.set_inters_point(join_x, join_y)
            
            print(f"Fitting on x0 = {x0} from {x0-stepsize_2} to {x0+stepsize_2}")
            tspline.fit(S, silent=True)

            join_x = x0+stepsize_2 - padding
            join_y = tspline.y(join_x)
            x0 += stepsize - padding
            tsplines.append(tspline)
        
        return tsplines

