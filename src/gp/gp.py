import random
import logging
import numpy as np
from scipy.spatial.distance import pdist as scipy_pdist
from scipy.spatial.distance import squareform as scipy_squareform

import dataset
from symbols.syntax_tree import SyntaxTree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.unaop import UnaryOperatorSyntaxTree
from symbols.misc  import UnknownSyntaxTree
from backprop import lpbackprop, jump_backprop
from backprop import bpropagator
from backprop import project
from gp import utils, creator, evaluation, evaluator, selector, crossover, mutator, corrector

from symbols import syntax_tree
import profiling


class Diversifier:
    def diversify(self, population, eval_map):
        pass

class SemanticCrowdingDiversifier(Diversifier):
    def __init__(self, data):
        self.data = data
        self.mean_dist = None
    
    def track(self, population):
        welldef_sems = []
        for p in population:
            sem_p = p(self.data.X)
            if np.isnan(sem_p).any() or np.isinf(sem_p).any(): continue
            welldef_sems.append(sem_p)

        dists = scipy_pdist( np.array(welldef_sems) )
        self.mean_dist = np.quantile( dists[np.isfinite(dists)], 0.01 )

    def diversify(self, population, eval_map):
        #self.track(population)
        #return
        welldef_fronts = {}
        undef_fronts = {}
        T_dist = {}

        for p in population:
            sem_p = p(self.data.X)
            front_p = eval_map[id(p)].fea_ratio

            if np.isnan(sem_p).any() or np.isinf(sem_p).any():
                if front_p not in undef_fronts: undef_fronts[front_p] = []
                undef_fronts[front_p].append(p)
            else:
                if front_p not in welldef_fronts: welldef_fronts[front_p] = []
                welldef_fronts[front_p].append(p)
                T_dist[id(p)] = scipy_squareform( scipy_pdist( np.array([p(self.data.X), self.data.y]) ) )[0].max()
        
        for front, ps in welldef_fronts.items():
            sem = np.array([p(self.data.X) for p in ps])
            #dist = np.min( scipy_squareform(scipy_pdist(sem)), axis=1 )
            dist_sort = np.sort( scipy_squareform(scipy_pdist(sem)), axis=1 )
            dist = dist_sort[:,0] if dist_sort.shape[1] == 1 else dist_sort[:,1]

            for i, p in enumerate(ps):
                eval_map[id(p)].crowdist = dist[i] / T_dist[id(p)]

        for front, ps in undef_fronts.items():
            for i, p in enumerate(ps):
                eval_map[id(p)].crowdist = 0


class GP:
    def __init__(self,
                 popsize:int,
                 ngen:int,
                 max_depth:int,
                 max_length:int,
                 S_train:dataset.NumpyDataset,
                 S_test:dataset.NumpyDataset,
                 creator:creator.SolutionCreator,
                 evaluator:evaluator.Evaluator,
                 selector:selector.Selector,
                 crossover:crossover.Crossover,
                 mutator:mutator.Mutator,
                 corrector:corrector.Corrector,
                 mutrate:float,
                 elitism:int=0,
                 knowledge=None):
        
        self.population = creator.create_population(popsize, max_depth, max_length)
        self.eval_map = {}
        self.popsize = popsize
        self.ngen = ngen
        self.S_train = S_train
        self.S_test = S_test
        self.evaluator = evaluator
        self.selector = selector
        self.crossover = crossover
        self.mutator = mutator
        self.corrector = corrector
        self.mutrate = mutrate
        self.elitism = elitism
        self.knowledge = knowledge
        self.stats = evaluator.create_stats()
        self.genidx = 0
        
        from backprop.pareto_front import DataLengthFrontTracker, MultiHeadFrontTracker
        #self.fea_front_tracker = DataLengthFrontTracker()
        self.fea_front_tracker = MultiHeadFrontTracker()
    
    def evolve(self, newgen_callback=None) -> tuple[list[SyntaxTree], dict]:
        """
        returns best syntax tree and its evaluation.
        """
        
        self._evaluate_all()
        self.population = utils.sort_population(self.population, self.eval_map)
        self.stats.update(self.population, self.eval_map)
        
        for self.genidx in range(1, self.ngen):

            if newgen_callback is not None:
                newgen_callback(self.genidx, f"\nGeneration {self.genidx} {self.stats.best_eval}")
            
            children = self._create_children()
            self._replace(children)
            self.stats.update(self.population, self.eval_map)
            
        """
        print()
        for p in self.population:
            print(p, self.eval_map[id(p)].fea_ratio, self.eval_map[id(p)].data_r2, self.eval_map[id(p)].know_mse)
        """
        
        return self.stats.best, self.stats.best_eval
        
    def _evaluate_all(self):
        self.eval_map.clear()
        for stree in self.population:
            self.eval_map[id(stree)] = self.evaluator.evaluate(stree)
    
    def _create_children(self) -> list[SyntaxTree]:
        
        children = []
        
        while len(children) < self.popsize:
            for _ in range(self.popsize - len(children)):

                parents = self.selector.select(self.population, self.eval_map, 2)

                child = self.crossover.cross(parents[0], parents[1])  # 100% crossover rate (child must be a new object!)
                
                if random.random() < self.mutrate:
                    child = self.mutator.mutate(child)
                
                if child.validate(): #TODO: and child not in children:
                    
                    child = child.simplify()
                    #before_child_eval = self.evaluator.evaluate(child)
                    #before_child = child.clone()
                    if self.corrector is not None:
                        profiling.enable()
                        child, _, _, _ = self.corrector.correct(child)
                        profiling.disable()

                    child_eval = self.evaluator.evaluate(child)
                    #if not child_eval.better_than(before_child_eval):
                    #    child = before_child
                    #    child_eval = self.evaluator.evaluate(child)

                    children.append(child)
                    self.eval_map[id(child)] = child_eval

                    if type(child_eval) is evaluation.LayeredEvaluation and child_eval.fea_ratio == 1.0:
                        self.fea_front_tracker.track(child, child_eval)
        
        return children
    
    def _replace(self, children:list[SyntaxTree]):
        if self.elitism > 0:
            children = utils.sort_population(children, self.eval_map)
            for i in range(self.elitism):
                children[-1-i] = self.population[i]
        self.population = children  # generational replacement.

        self.population = utils.sort_population(self.population, self.eval_map)
        #self.population = (self.fea_front_tracker.get_population() + self.population)[:self.popsize]
        
        # update evaluation map based on new population.
        new_eval_map = {}
        for stree in self.population:
            new_eval_map[id(stree)] = self.eval_map[id(stree)]
        
        self.eval_map = new_eval_map
