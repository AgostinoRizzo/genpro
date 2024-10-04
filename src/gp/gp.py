from functools import cmp_to_key
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
from gp import creator, evaluator, selector, crossover, mutator, corrector

from symbols import syntax_tree
import profiling


def sort_population(population:list[SyntaxTree], eval_map:dict) -> list[SyntaxTree]:
    def strees_cmp(stree1, stree2) -> int:
        nonlocal eval_map
        stree1_eval = eval_map[id(stree1)]
        stree2_eval = eval_map[id(stree2)]
        if stree1_eval.better_than(stree2_eval): return -1
        if stree2_eval.better_than(stree1_eval): return  1
        return 0
    return sorted(population, key=cmp_to_key(strees_cmp))


def replace_subtree(stree:SyntaxTree,
                    sub_stree:SyntaxTree,
                    new_sub_stree:SyntaxTree) -> SyntaxTree:
    stree.set_parent()

    if sub_stree.parent is None:
        return new_sub_stree
    if type(sub_stree.parent) is BinaryOperatorSyntaxTree:
        if   id(sub_stree) == id(sub_stree.parent.left) : sub_stree.parent.left  = new_sub_stree
        elif id(sub_stree) == id(sub_stree.parent.right): sub_stree.parent.right = new_sub_stree
    elif type(sub_stree.parent) is UnaryOperatorSyntaxTree:
        if id(sub_stree) == id(sub_stree.parent.inner): sub_stree.parent.inner = new_sub_stree
    
    return stree


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


def generate_trunks(max_depth:int, nvars:int, knowledge):
    from backprop import lpbackprop
    from backprop import xgp
    solutionCreator = xgp.RandomTemplateSolutionCreator(nvars=nvars)
    all_trunks = []
    satunsat_trunks = {'sat': [], 'unsat': []}

    for _ in range(100):
        trunk = solutionCreator.create_population(1, max_depth=max_depth)[0]
        if type(trunk) is UnknownSyntaxTree or \
           trunk in all_trunks or \
           check_unsat_trunk(satunsat_trunks, trunk): continue
        all_trunks.append(trunk)
        #print(f"Checking trunk: {trunk}")
        sat, _ = lpbackprop.lpbackprop(knowledge, trunk, None)
        if sat:
            satunsat_trunks['sat'].append(trunk)
            #print(f"SAT  : {trunk}")
        else:
            satunsat_trunks['unsat'].append(trunk)
            #print(f"UNSAT: {trunk}")
    return satunsat_trunks

def check_unsat_trunk(trunks:map, stree) -> bool:
    for unsat_trunk in trunks['unsat']:
        """if type(stree) is BinaryOperatorSyntaxTree and stree.operator == '*' and \
            type(stree.left) is backprop.VariableSyntaxTree and type(stree.right) is backprop.ConstantSyntaxTree and \
                type(unsat_trunk) is BinaryOperatorSyntaxTree and \
                type(unsat_trunk.left) is UnknownSyntaxTree and type(unsat_trunk.right) is UnknownSyntaxTree:
                    print()"""
        if stree.match(unsat_trunk):
            return True
    return False


class GPStats:
    def __init__(self):
        self.best = None
        self.best_eval = None
        self.qualities = {'currBest': [], 'currAvg': [], 'currWorst': [], 'best': []}
    
    def update(self, population, eval_map):
        currBest  = 0.0
        currAvg   = 0.0
        currWorst = 1.0

        for stree in population:
            stree_eval = eval_map[id(stree)]
            stree_eval_value = stree_eval.get_value()
            
            currBest   = max(currBest, stree_eval_value)
            currAvg   += stree_eval_value
            currWorst  = min(currWorst, stree_eval_value)

            if self.best is None or stree_eval.better_than(self.best_eval):
                self.best = stree
                self.best_eval = stree_eval

        currAvg /= len(population)

        self.qualities['currBest' ].append(currBest)    
        self.qualities['currAvg'  ].append(currAvg)
        self.qualities['currWorst'].append(currWorst)
        #self.qualities['best'     ].append(self.bests_eval_map[id(self.bests[0])].get_value())
        

class FUGPStats(GPStats):
    def __init__(self,):
        super().__init__()
        self.buckets = {}
        self.fea_ratio = {'currBest': [], 'currAvg': [], 'currWorst': [], 'best': []}
    
    def update(self, population, eval_map):
        super().update(population, eval_map)

        fea_ratio_best  = 0.0
        fea_ratio_avg   = 0.0
        fea_ratio_worst = 1.0

        nconsts = eval_map[id(population[0])].know_n
        for stree in population:
            fea_ratio = eval_map[id(stree)].fea_ratio

            fea_ratio_best   = max(fea_ratio_best, fea_ratio)
            fea_ratio_avg   += fea_ratio
            fea_ratio_worst  = min(fea_ratio_worst, fea_ratio)
        
        fea_ratio_avg /= len(population)

        self.fea_ratio['currBest' ].append(fea_ratio_best)    
        self.fea_ratio['currAvg'  ].append(fea_ratio_avg)
        self.fea_ratio['currWorst'].append(fea_ratio_worst)
        #self.fea_ratio['best'     ].append(self.bests_eval_map[id(self.bests[0])].fea_ratio)


class GP:
    def __init__(self,
                 popsize:int,
                 ngen:int,
                 max_depth:int,
                 S_train:dataset.NumpyDataset,
                 S_test:dataset.NumpyDataset,
                 creator:creator.SolutionCreator,
                 evaluator:evaluator.Evaluator,
                 selector:selector.Selector,
                 crossover:crossover.Crossover,
                 mutator:mutator.Mutator,
                 mutrate:float,
                 elitism:int=0,
                 knowledge=None,
                 rseed=None):
        
        self.population = creator.create_population(popsize, max_depth)
        self.eval_map = {}
        self.popsize = popsize
        self.ngen = ngen
        self.S_train = S_train
        self.S_test = S_test
        self.evaluator = evaluator
        self.selector = selector
        self.crossover = crossover
        self.mutator = mutator
        self.mutrate = mutrate
        self.elitism = elitism
        self.knowledge = knowledge
        if rseed is not None:
            random.seed(rseed)
        self.stats = evaluator.create_stats()
        self.genidx = 0
        
        from backprop.pareto_front import DataLengthFrontTracker, MultiHeadFrontTracker
        #self.fea_front_tracker = DataLengthFrontTracker()
        self.fea_front_tracker = MultiHeadFrontTracker()

        self.corrector = corrector.Corrector(S_train, knowledge, max_depth)
    
    def evolve(self, newgen_callback=None) -> tuple[list[SyntaxTree], dict]:
        """
        returns best syntax tree and its evaluation.
        """
        
        self._evaluate_all()
        self.population = sort_population(self.population, self.eval_map)
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

                profiling.enable()
                child = self.crossover.cross(parents[0], parents[1])  # 100% crossover rate (child must be a new object!)
                profiling.disable()
                
                if random.random() < self.mutrate:
                    child = self.mutator.mutate(child)
                
                if child.validate(): #TODO: and child not in children:
                    
                    child = child.simplify()
                    child, _, _, _ = self.corrector.correct(child)

                    child_eval = self.evaluator.evaluate(child)

                    children.append(child)
                    self.eval_map[id(child)] = child_eval

                    if child_eval.fea_ratio == 1.0:
                        self.fea_front_tracker.track(child, child_eval)
        
        return children
    
    def _replace(self, children:list[SyntaxTree]):
        if self.elitism > 0:
            children = sort_population(children, self.eval_map)
            for i in range(self.elitism):
                children[-1-i] = self.population[i]
        self.population = children  # generational replacement.

        self.population = sort_population(self.population, self.eval_map)
        #self.population = (self.fea_front_tracker.get_population() + self.population)[:self.popsize]
        
        # update evaluation map based on new population.
        new_eval_map = {}
        for stree in self.population:
            new_eval_map[id(stree)] = self.eval_map[id(stree)]

            assert self.eval_map[id(stree)].fea_ratio == self.evaluator.evaluate(stree).fea_ratio
        self.eval_map = new_eval_map
