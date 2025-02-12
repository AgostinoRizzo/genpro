import random
import logging
import numpy as np
from scipy.spatial.distance import pdist as scipy_pdist
from scipy.spatial.distance import squareform as scipy_squareform
from dataclasses import dataclass

import dataset
from symbols.syntax_tree import SyntaxTree
from symbols.const import ConstantSyntaxTree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.unaop import UnaryOperatorSyntaxTree
from symbols.misc  import UnknownSyntaxTree
from backprop import lpbackprop, jump_backprop
from backprop import bpropagator
from backprop import project
from backprop.bperrors import BackpropError
from backprop.library import LibraryError
from backprop.pareto_front import DataLengthFrontTracker, MultiHeadFrontTracker, FrontDuplicateError
from gp import utils, creator, evaluation, evaluator, selector, crossover, mutator, corrector
from gp.stats import CorrectorGPStats, PropertiesGPStats

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


@dataclass
class GPSettings:
    popsize:int
    ngen:int
    max_depth:int
    max_length:int
    S_train:dataset.NumpyDataset
    S_test:dataset.NumpyDataset
    creator:creator.SolutionCreator
    evaluator:evaluator.Evaluator
    selector:selector.Selector
    crossover:crossover.Crossover
    mutator:mutator.Mutator
    corrector:corrector.Corrector
    mutrate:float
    elitism:int=0
    knowledge:dataset.DataKnowledge=None
    track_fea_front:bool=True


class GP:
    def __init__(self, args:GPSettings):
        self.population = args.creator.create_population(args.popsize, args.max_depth, args.max_length, min_length=3)
        self.eval_map   = {}
        self.popsize    = args.popsize
        self.ngen       = args.ngen
        self.S_train    = args.S_train
        self.S_test     = args.S_test
        self.evaluator  = args.evaluator
        self.selector   = args.selector
        self.crossover  = args.crossover
        self.mutator    = args.mutator
        self.corrector  = args.corrector
        self.mutrate    = args.mutrate
        self.elitism    = args.elitism
        self.knowledge  = args.knowledge
        self.stats      = args.evaluator.create_stats()
        self.genidx     = 0
        self.fea_front_tracker = None
        self.visualizer = None

        self.stats = PropertiesGPStats(self.stats)
        if self.corrector is not None:
            self.corrector.evaluator = self.evaluator
            self.stats = CorrectorGPStats(self.stats)
        
        if args.track_fea_front:
            self.fea_front_tracker = MultiHeadFrontTracker(self.popsize, max_fronts=1, min_fea_ratio=0.9)
        
    def evolve(self, newgen_callback=None) -> tuple[list[SyntaxTree], dict]:
        """
        returns best syntax tree and its evaluation.
        """
        
        self._evaluate_all()
        self._on_initial_generation()
        self.stats.update(self)

        for self.genidx in range(1, self.ngen):

            if newgen_callback is not None:
                newgen_callback(self.genidx, f"\nGeneration {self.genidx} {self.eval_map[id(self.population[0])]}")
            
            children = self._create_children()
            self._replace(children)
            self.stats.update(self)
            
        """
        print()
        for p in self.population:
            print(p, self.eval_map[id(p)].fea_ratio, self.eval_map[id(p)].data_r2, self.eval_map[id(p)].know_mse)
        """
        
        """print()
        for p in self.population:
            f  = "{:.2f}".format(self.eval_map[id(p)].fea_ratio)
            r2 = "{:.2f}".format(self.eval_map[id(p)].data_eval.value)
            print(f, r2, p)"""
        
        return self.population[0], self.eval_map[id(self.population[0])]
    
    def _on_initial_generation(self):
        self.population = utils.sort_population(self.population, self.eval_map)
        if self.visualizer is not None:
            for p in self.population:
                self.visualizer.track(p, 'Initial population')
        
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
                    
                    if self.visualizer is not None:
                        self.visualizer.track(child, 'After Crossover&Mutation')

                    child_eval = None
                    if self.corrector is not None:
                        
                        try:
                            #profiling.enable()
                            corrected_child, new_node, C_pulled, _ = self.corrector.correct(child)
                            #profiling.disable()
                            corrected_child = corrected_child.simplify()
                            if type(corrected_child) is not ConstantSyntaxTree:
                                child = corrected_child
                                self.stats.on_correction(C_pulled)

                        except BackpropError as backprop_e:
                            self.stats.on_backprop_error(backprop_e)
                        except LibraryError as lib_e:
                            self.stats.on_library_error(lib_e)
                        except corrector.AlreadyCorrectedError as corr_e:
                            child_eval = corr_e.stree_eval

                    if self.visualizer is not None:
                        self.visualizer.track(child, 'After Correction')

                    if child_eval is None:
                        child_eval = self.evaluator.evaluate(child)

                    children.append(child)
                    self.eval_map[id(child)] = child_eval

                    if self.fea_front_tracker is not None:
                        try: self.fea_front_tracker.track(child, (child_eval.data_eval.value, child.cache.nnodes), child_eval)
                        except FrontDuplicateError: pass
        
        return children
    
    def _replace(self, children:list[SyntaxTree]):
        if self.elitism > 0:
            children = utils.sort_population(children, self.eval_map)
            for i in range(self.elitism):
                children[-1-i] = self.population[i]
        
        self.population = children  # generational replacement.
        self.population = utils.sort_population(self.population, self.eval_map)
        
        # update evaluation map based on new population.
        self._update_evaluation()
    
    def _update_evaluation(self):
        new_eval_map = {}
        for stree in self.population:
            new_eval_map[id(stree)] = self.eval_map[id(stree)]
        self.eval_map = new_eval_map


"""class MOGP(GP):
    def __init__(self, args:GPSettings):
        super().__init__(args)
        self.elitism = 0
        self.front_tracker = DataLengthFrontTracker(self.popsize)
    
    def _on_initial_generation(self):
        self.__update_population_from_fronts(self.population)
    
    def _replace(self, children:list[SyntaxTree]):
        self.__update_population_from_fronts(children)

        # update evaluation map based on new population.
        self._update_evaluation()

    def __update_population_from_fronts(self, children:list[SyntaxTree]):
        duplicates = []
        
        for c in children:
            c_eval = self.eval_map[id(c)]
            try: self.front_tracker.track(c, (c_eval.data_eval.value, c.cache.nnodes), c_eval)
            except FrontDuplicateError:
                duplicates.append(c)
        
        self.population = self.front_tracker.get_population(self.popsize)
        for stree in self.population:
            self.eval_map[id(stree)] = self.front_tracker.eval_map[id(stree)]
        
        remaining = self.popsize - len(self.population)
        if remaining > 0:
            self.population += utils.sort_population(duplicates, self.eval_map)[:remaining]
        
        assert len(self.population) == self.popsize"""


class MOGP(GP):
    def __init__(self, args:GPSettings):
        args.track_fea_front = False
        super().__init__(args)
        self.elitism = 0
        self.fea_fronts_size = 0

        self.fea_front_tracker = MultiHeadFrontTracker(self.popsize)
        assert type(self.evaluator) is evaluator.LayeredEvaluator
    
    def _on_initial_generation(self):
        self.__update_population_from_fronts(self.population)
    
    def _replace(self, children:list[SyntaxTree]):
        self.__update_population_from_fronts(children)

        # update evaluation map based on new population.
        self._update_evaluation()

    def __update_population_from_fronts(self, children:list[SyntaxTree]):
        duplicates = []
        
        for c in children:
            c_eval = self.eval_map[id(c)]
            try: self.fea_front_tracker.track(c, (c_eval.data_eval.value, c.cache.nnodes), c_eval)
            except FrontDuplicateError:
                duplicates.append(c)
        
        populations = self.fea_front_tracker.get_populations()
        self.population = []
        for h, p in populations:
            for stree in p:
                self.eval_map[id(stree)] = self.fea_front_tracker.heads[h].eval_map[id(stree)]
            self.population += p
        
        remaining = self.popsize - len(self.population)
        if remaining > 0:
            self.population += utils.sort_population(duplicates, self.eval_map)[:remaining]
        
        assert len(self.population) == self.popsize