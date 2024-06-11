import sys
import random
import numpy as np
from math import sqrt
import pygad
from IPython.display import clear_output

import dataset
import bezier_spline


__S = None
__spline = None
__function_inputs = None


def __fitfunc(ga_instance, solution, solution_idx):
    __spline.set_chromo(solution)
    return __spline.fitness(__S)


def __crossover_func(parents, offspring_size, ga_instance):  # single point crossover
    #print(f"Parents: {parents}")
    #print(f"Parents shape: {parents.shape}")
    #print(f"Offspring_size: {offspring_size}")

    offspring = []
    while len(offspring) != offspring_size[0]:
        
        added = False
        for _ in range(1):

            parent1_idx = random.randint(0, parents.shape[0]-1)
            parent2_idx = parent1_idx
            while parent1_idx == parent2_idx:
                parent2_idx = random.randint(0, parents.shape[0]-1)
            
            parent1 = parents[parent1_idx, :].copy()
            parent2 = parents[parent2_idx, :].copy()

            chromo_size = offspring_size[1]
            x_pleft = __spline.getx(1)
            x_pright = parent2[1]
            safe_coross_idx = []

            for cross_idx in range(1, chromo_size):
                if x_pleft <= x_pright: safe_coross_idx.append(cross_idx)
                if cross_idx < chromo_size - 1 and cross_idx % 2 != 0:
                    x_pleft = parent1[cross_idx]
                    x_pright = parent2[cross_idx+2]
            
            if len(safe_coross_idx) == 0:
                continue

            random_cross_point = np.random.choice(safe_coross_idx)

            #print(f"Crossover between {parent1} and {parent2} at {random_cross_point}")
            parent1[random_cross_point:] = parent2[random_cross_point:]
            #print(f"\t{parent1}")

            __spline.set_chromo(np.array(parent1))
            offspring.append(parent1)
            added = True
            break
        
        if not added:
            raise RuntimeError('Crossover error. Invalid limit exceeded.')

    return np.array(offspring)


def __mutation_func(offspring, ga_instance):
    coord_map = __spline.get_chromo_coordmap()
    chromo_size = offspring.shape[1]

    for chromosome_idx in range(offspring.shape[0]):
        random_gene_idx = np.random.choice(range(offspring.shape[1]))

        to_mutate = offspring[chromosome_idx].copy()
        delta = None

        if coord_map[random_gene_idx] == 'x':
            x_left = None
            x_right = None
            if random_gene_idx < 3: x_left = __spline.getx(1)
            else: x_left = to_mutate[random_gene_idx-2]

            if random_gene_idx >= chromo_size - 3: x_right = __spline.getx(__spline.nnodes)
            else: x_right = to_mutate[random_gene_idx+2]
            
            delta = (x_right - x_left) / 2
            delta = np.random.uniform(-delta, delta)

        else:
            delta = (__S.yu-__S.yl)*0.5*0.5
            delta = np.random.uniform(-delta, delta)

        #print(f"Before mutation {to_mutate} at {random_gene_idx}")
        #__spline.set_chromo(to_mutate)
        #if not __spline.isvalid(__S):
        #    print(f"Not valid {to_mutate} at {random_gene_idx}")
        #    print(f"\tOrigin: {offspring[chromosome_idx]}")
        to_mutate[random_gene_idx] += delta
        #print(f"\tAfter mutation {to_mutate}")
        offspring[chromosome_idx][random_gene_idx] += delta
        
    return offspring


def __create_initial_population(size:int) -> np.array:
    pop = []
    chromo_len  = __spline.get_chromo_length()
    chromo_cmap = __spline.get_chromo_coordmap()
    for _ in range(size):
        added = False
        for _ in range(100000):
            chromo = []
            for ci in range(chromo_len):
                l = None; u = None
                if chromo_cmap[ci] == 'x': l = __S.xl; u = __S.xu
                else: l = __S.yl; u = __S.yu

                chromo.append(random.uniform(l, u))
            
            __spline.set_chromo(np.array(chromo))
            if __spline.isvalid(__S):
                pop.append(chromo)
                added = True
                break
        
        if not added:
            raise RuntimeError('Random chrome error. Invalid limit exceeded.')
    
    return np.array(pop)


def __callback_generation(ga_instance):
    clear_output(wait=True)
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    #print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


def __init_ga(popsize:int=100, ngens:int=100):
    fitness_function = __fitfunc

    num_generations = ngens
    num_parents_mating = 4

    sol_per_pop = popsize
    num_genes = __function_inputs

    initial_population = __create_initial_population(sol_per_pop)
    #init_range_low = -20
    #init_range_high = 20

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = __crossover_func #"single_point"

    mutation_type = __mutation_func #"random"
    mutation_percent_genes = 1

    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       initial_population=initial_population,
                       #init_range_low=init_range_low,
                       #init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=None)#__callback_generation)
    
    return ga_instance

def __ga_output(ga_instance):
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=sqrt(sys.maxsize-solution_fitness)))

    prediction = np.sum(np.array(__function_inputs)*solution)
    print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

    ga_instance.plot_fitness();

    return solution

def fit_spline(spline, S:dataset.Dataset):
    global __spline
    global __S
    global __function_inputs

    __S = S
    __spline = spline
    __function_inputs = __spline.get_chromo_length()
    ga_instance = __init_ga()
    ga_instance.run()
    solution = __ga_output(ga_instance)
    spline.set_chromo(solution)

def fit_spline_local(S:dataset.Dataset):
    global __spline
    global __S
    global __function_inputs

    __S = S

    n_curves = 2
    xfix = S.xl
    spline_connector = bezier_spline.BezierCurveConnector(S.xl, S.xu)

    for ic in range(n_curves):
        if xfix >= S.xu: raise RuntimeError('X upper limit reached before conclusion.')
        c: bezier_spline.BezierCurve = bezier_spline.BezierCurve()
        
        c.fixnode(1, 'x', xfix)
        if ic == n_curves-1: c.fixnode(4, 'x', S.xu)
        else: c.fixnode(4, 'x', 0.)
        
        __spline = c
        __function_inputs = c.get_chromo_length()
        ga_instance = __init_ga()
        print(f"Fitting {ic+1}/{n_curves} curve...")
        ga_instance.run()
        solution = __ga_output(ga_instance)
        c.set_chromo(solution)

        xfix = c.getx(4)
        spline_connector.connect(c)
    
    return spline_connector


def fit_spline_global(S:dataset.Dataset, nnodes:int=4, popsize:int=100, ngens:int=100):
    global __spline
    global __S
    global __function_inputs

    __S = S

    c = bezier_spline.AnchoredBezierCurve(S.xl, S.xu, nnodes)
    print(f"Chromo length: {c.get_chromo_length()}")
    print(f"Initial chromo: {c.get_chromo()}")
    
    __spline = c
    __function_inputs = c.get_chromo_length()
    ga_instance = __init_ga(popsize=popsize, ngens=ngens)
    print(f"Fitting curve...")
    ga_instance.run()
    solution = __ga_output(ga_instance)
    print(f"Solution: {solution}")
    c.set_chromo(solution)

    return c