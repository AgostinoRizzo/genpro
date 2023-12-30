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
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        
        added = False
        for _ in range(10):
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

            random_split_point = np.random.choice(range(offspring_size[1]))

            parent1[random_split_point:] = parent2[random_split_point:]

            __spline.set_chromo(np.array(parent1))
            if __spline.isvalid(__S):
                offspring.append(parent1)
                added = True
                break
        
        if not added:
            raise RuntimeError('Crossover error. Invalid limit exceeded.')
        idx += 1

    return np.array(offspring)

def __mutation_func(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        random_gene_idx = np.random.choice(range(offspring.shape[1]))

        added = False
        for _ in range(100):
            delta = min((__S.xu-__S.xl)*0.1*0.5, (__S.yu-__S.yl)*0.1*0.5)
            delta = np.random.uniform(-delta, delta)  # TODO: choose it
            mutated = offspring[chromosome_idx].copy()
            mutated[random_gene_idx] += delta
            __spline.set_chromo(mutated)
            
            if __spline.isvalid(__S):
                offspring[chromosome_idx, random_gene_idx] += delta
                added = True
                break
        
        if not added:
            raise RuntimeError('Mutation error. Invalid limit exceeded.')
        
    return offspring

def __create_initial_population(size:int) -> np.array:
    pop = []
    chromo_len = __spline.get_chromo_length()
    for _ in range(size):
        added = False
        for _ in range(100000):
            chromo = []
            for _ in range(chromo_len):
                chromo.append(random.uniform(__S.xl, __S.xu))  # TODO: adjust limits
            
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


def __init_ga():
    fitness_function = __fitfunc

    num_generations = 100
    num_parents_mating = 4

    sol_per_pop = 100 #10
    num_genes = __function_inputs

    initial_population = __create_initial_population(sol_per_pop)

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
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       on_generation=__callback_generation)
    
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

    n_curves = 1
    xfix = S.xl
    spline_connector = bezier_spline.BezierCurveConnector(S.xl, S.xu)

    for ic in range(n_curves):
        if xfix >= S.xu: raise RuntimeError('X upper limit reached before conclusion.')
        c: bezier_spline.BezierCurve = bezier_spline.BezierCurve()
        
        c.fixnode(1, 'x', xfix)
        #if ic == n_curves-1: c.fixnode(4, 'x', S.xu)
        
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