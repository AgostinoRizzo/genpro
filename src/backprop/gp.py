from functools import cmp_to_key
import random
import logging
import numpy as np
from scipy.spatial.distance import pdist as scipy_pdist
from scipy.spatial.distance import squareform as scipy_squareform

import dataset
from backprop import backprop, lpbackprop, jump_backprop
from backprop import bpropagator


class Evaluation:
    def __init__(self, minimize:bool=True):
        self.minimize = minimize
    def better_than(self, other) -> bool: return False

class RealEvaluation(Evaluation):
    def __init__(self, value, minimize:bool=True):
        super().__init__(minimize)
        self.value = value
    def better_than(self, other) -> bool:
        if np.isnan(other.value):
            if np.isnan(self.value): return False
            return True
        if self.minimize: return self.value < other.value
        return self.value > other.value
    def get_value(self):
        return self.value
    def __str__(self) -> str:
        return f"{self.value}"

class FUEvaluation(Evaluation):
    def __init__(self, know_mse, know_nv, know_n, know_ls, know_sat, data_r2, nnodes):
        self.know_mse  = know_mse
        self.know_nv   = know_nv
        self.know_n    = know_n
        self.know_ls   = know_ls
        self.know_sat  = know_sat
        self.data_r2   = data_r2
        self.nnodes    = nnodes
        self.crowdist  = 0

        self.fea_ratio = 1. - (know_nv / know_n)
        self.fea_bucket = int( self.fea_ratio * 10. )
        self.feasible  = know_nv == 0
    
        self.data_nnodes = data_r2 / nnodes

        self.genidx = 0

    """def better_than(self, other) -> bool:
        if self.feasible and other.feasible:
            return self.data_r2 > other.data_r2
        
        if not self.feasible and not other.feasible:
            if self.know_nv == other.know_nv: return self.know_mse < other.know_mse
            return self.know_nv < other.know_nv
        
        if self.feasible and not other.feasible: return True
        return False"""
    
    def better_than(self, other) -> bool:
        #if self.know_ls and not other.know_ls: return True
        #if not self.know_ls and other.know_ls: return False

        #if self.know_sat < other.know_sat: return True
        #if self.know_sat > other.know_sat: return False

        #if self.know_sat and not other.know_sat: return True
        #if not self.know_sat and other.know_sat: return False

        #if self.fea_bucket > other.fea_bucket: return True
        #if self.fea_bucket < other.fea_bucket: return False

        #if self.feasible and not other.feasible: return True
        #if not self.feasible and other.feasible: return False

        if self.data_r2 == 0.0: return False

        if self.fea_ratio > other.fea_ratio: return True
        if self.fea_ratio < other.fea_ratio: return False

        #if self.fea_ratio > other.fea_ratio and self.data_r2 > 0: return True
        #if self.fea_ratio < other.fea_ratio and other.data_r2 > 0: return False

        #if self.nnodes < other.nnodes: return True
        #if self.nnodes > other.nnodes: return False

        #if self.fea_ratio < 1.0:
        #    return self.know_mse < other.know_mse

        return self.data_r2 > other.data_r2

        #return self.data_nnodes > other.data_nnodes
        #if self.crowdist > other.crowdist: return True
        #if self.crowdist < other.crowdist: return False
    
    def get_value(self):
        return self.data_r2
    
    def __str__(self) -> str:
        return f"know_mse:  {self.know_mse }\n" + \
               f"know_nv:   {self.know_nv  }\n" + \
               f"know_n:    {self.know_n   }\n" + \
               f"know_ls:   {self.know_ls  }\n" + \
               f"fea_ratio: {self.fea_ratio}\n" + \
               f"data_r2:   {self.data_r2  }"


def sort_population(population:list[backprop.SyntaxTree], eval_map:dict) -> list[backprop.SyntaxTree]:
    def strees_cmp(stree1, stree2) -> int:
        nonlocal eval_map
        stree1_eval = eval_map[id(stree1)]
        stree2_eval = eval_map[id(stree2)]
        if stree1_eval.better_than(stree2_eval): return -1
        if stree2_eval.better_than(stree1_eval): return  1
        return 0
    return sorted(population, key=cmp_to_key(strees_cmp))


def replace_subtree(stree:backprop.SyntaxTree,
                    sub_stree:backprop.SyntaxTree,
                    new_sub_stree:backprop.SyntaxTree) -> backprop.SyntaxTree:
    stree.set_parent()

    if sub_stree.parent is None:
        return new_sub_stree
    if type(sub_stree.parent) is backprop.BinaryOperatorSyntaxTree:
        if   id(sub_stree) == id(sub_stree.parent.left) : sub_stree.parent.left  = new_sub_stree
        elif id(sub_stree) == id(sub_stree.parent.right): sub_stree.parent.right = new_sub_stree
    elif type(sub_stree.parent) is backprop.UnaryOperatorSyntaxTree:
        if id(sub_stree) == id(sub_stree.parent.inner): sub_stree.parent.inner = new_sub_stree
    
    return stree


class Evaluator:
    def evaluate(self, stree:backprop.SyntaxTree): return None
    def create_stats(self, nbests:int=1): return GPStats(nbests)

class R2Evaluator(Evaluator):
    def __init__(self, dataset, minimize:bool=True):
        self.dataset = dataset
        self.minimize = False
    
    def evaluate(self, stree:backprop.SyntaxTree):
        return RealEvaluation(max(0., self.dataset.evaluate(stree)['r2']), self.minimize)

# different from srgp.KnowledgeEvaluator
class KnowledgeEvaluator(Evaluator):
    def __init__(self, knowledge):
        self.K = knowledge
    
    def _compute_stree_derivs(self, stree, derivs):
        return backprop.SyntaxTree.diff_all(stree, derivs, include_zeroth=True)
    
    def evaluate(self, stree:backprop.SyntaxTree):
        K_derivs = self.K.get_derivs()
        stree_derivs = self._compute_stree_derivs(stree, K_derivs)
        K_eval = self.K.evaluate(stree_derivs)
        K_eval = (K_eval['mse0'] + K_eval['mse1'] + K_eval['mse2']) / 3  # TODO: separate?! or a weighted mean?
        if np.isnan(K_eval): K_eval = 1e12
        return RealEvaluation(K_eval, minimize=True)

class NumericalKnowledgeEvaluator(KnowledgeEvaluator):
    def __init__(self, knowledge):
        super().__init__(knowledge)
    
    def _compute_stree_derivs(self, stree, derivs):
        return backprop.Derivative.create_all(stree, derivs, self.K.nvars, self.K.numlims)
    
    def evaluate(self, stree:backprop.SyntaxTree):
        K_derivs = self.K.get_derivs()
        stree_derivs = self._compute_stree_derivs(stree, K_derivs)
        K_eval = self.K.evaluate(stree_derivs, eval_deriv=True)
        K_eval = (K_eval['mse0'] + K_eval['mse1'] + K_eval['mse2']) / 3  # TODO: separate?! or a weighted mean?
        if np.isnan(K_eval): K_eval = 1e12
        return RealEvaluation(K_eval, minimize=True)

class FUEvaluator(Evaluator):
    def __init__(self, dataset, knowledge):
        self.data = dataset
        self.know = knowledge
    
    def _compute_stree_derivs(self, stree, derivs):
        return backprop.SyntaxTree.diff_all(stree, derivs, include_zeroth=True)

    def evaluate(self, stree:backprop.SyntaxTree, eval_deriv=False):
        know_derivs = self.know.get_derivs()
        stree_derivs = self._compute_stree_derivs(stree, know_derivs)
        
        know_eval = self.know.evaluate(stree_derivs, eval_deriv)
        know_mse  = (know_eval['mse0'] + know_eval['mse1'] + know_eval['mse2']) / 3  # TODO: separate?! or a weighted mean?
        know_nv   =  know_eval['nv0' ] + know_eval['nv1' ] + know_eval['nv2' ]
        know_n    =  know_eval['n0'  ] + know_eval['n1'  ] + know_eval['n2'  ]
        know_ls   =  know_eval['ls0' ] and know_eval['ls1' ] and know_eval['ls2' ]
        know_sat  = stree.sat
        if np.isnan(know_mse): know_mse = 1e12

        data_r2 = max(0., self.data.evaluate(stree)['r2']) # TODO: put this into data.evaluate(...).

        """optCollector = backprop.SyntaxTreeOperatorCollector()
        stree.accept(optCollector)
        ispoly = True
        for o in ['/', 'log', 'exp', 'sqrt']:
            if o in optCollector.opts:
                ispoly = False
                break
        if ispoly:
            know_nv = 1e10
            know_mse = 1e10"""

        return FUEvaluation(know_mse, know_nv, know_n, know_ls, know_sat, data_r2, stree.get_nnodes())
    
    def create_stats(self, nbests:int=1): return FUGPStats(nbests)

class NumericalFUEvaluator(FUEvaluator):
    def __init__(self, dataset, knowledge):
        super().__init__(dataset, knowledge)
    
    def _compute_stree_derivs(self, stree, derivs):
        return backprop.Derivative.create_all(stree, derivs, self.know.nvars, self.know.numlims)



class Selector:
    def select(self, population:list[backprop.SyntaxTree], eval_map:dict, nparents:int=2) -> list[backprop.SyntaxTree]:
        return None

class TournamentSelector(Selector):
    def __init__(self, group_size:int):
        self.group_size = group_size
    
    def select(self, population:list[backprop.SyntaxTree], eval_map:dict, nparents:int=2) -> list[backprop.SyntaxTree]:
        parents = []
        for _ in range(nparents):
            group = random.choices(population, k=self.group_size)
            sorted_group = sort_population(group, eval_map)
            parents.append(sorted_group[0])
        return parents


class Crossover:
    def cross(self, parent1:backprop.SyntaxTree, parent2:backprop.SyntaxTree) -> backprop.SyntaxTree:
        return None

class SubTreeCrossover:
    def __init__(self, max_depth:int):
        self.max_depth = max_depth
    
    def cross(self, parent1:backprop.SyntaxTree, parent2:backprop.SyntaxTree) -> backprop.SyntaxTree:
        nnodes1 = parent1.get_nnodes()
        
        child = parent1.clone()
        child.set_parent()
        nodeSelector = backprop.SyntaxTreeNodeSelector(random.randrange(nnodes1))
        child.accept(nodeSelector)
        cross_point1 = nodeSelector.node
        child.set_parent()
        if cross_point1 is None:
            parent1.get_nnodes()
        cross_point1_depth = cross_point1.get_depth()
        max_nesting_depth = self.max_depth - cross_point1_depth
        
        nodesCollector = backprop.SyntaxTreeNodeCollector()
        parent2.accept(nodesCollector)
        allowedNodes = []
        for node in nodesCollector.nodes:
            if node.get_max_depth() <= max_nesting_depth: allowedNodes.append(node)
        
        if len(allowedNodes) == 0: return child
        cross_point2 = random.choice(allowedNodes)
        
        #print(f"From {child}")
        #print(f"\tReplacing {cross_point1}")
        #print(f"\tWith {cross_point2}")
        got = replace_subtree(child, cross_point1, cross_point2.clone())
        got.clear_cache()
        #print(f"Got {got}")
        #if got.get_max_depth() > self.max_depth:
        #    raise RuntimeError(f"Max depth: {got.get_max_depth()}, {max_nesting_depth}, {cross_point1.get_depth()}, {cross_point2.get_depth()}")
        return got

class SubTreeImprovementCrossover:
    def __init__(self, max_depth:int, S_train):
        self.max_depth = max_depth
        self.evaluator = R2Evaluator(S_train)
    
    def cross(self, parent1:backprop.SyntaxTree, parent2:backprop.SyntaxTree) -> backprop.SyntaxTree:
        nnodes1 = parent1.get_nnodes()
        nnodes2 = parent2.get_nnodes()
        
        child = parent1.clone()
        nodeSelector = backprop.SyntaxTreeNodeSelector(random.randrange(nnodes1))
        child.accept(nodeSelector)
        cross_point1 = nodeSelector.node
        child.set_parent()
        if cross_point1 is None:
            parent1.get_nnodes()
        cross_point1_depth = cross_point1.get_depth()
        max_nesting_depth = (self.max_depth - cross_point1_depth) - 1
        if max_nesting_depth < 0: return child
        
        nodesCollector = backprop.SyntaxTreeNodeCollector()
        parent2.accept(nodesCollector)
        allowedNodes = []
        for node in nodesCollector.nodes:
            if node.get_max_depth() <= max_nesting_depth: allowedNodes.append(node)
        
        if len(allowedNodes) == 0: return child
        cross_point2 = random.choice(allowedNodes)

        got = replace_subtree(child, cross_point1, cross_point2.clone())
        got.clear_cache()

        p1_eval = self.evaluator.evaluate(parent1)
        got_eval = self.evaluator.evaluate(got)
        if got_eval.better_than(p1_eval):
            print(f"{cross_point2.simplify()}")

        return got

class ExhaustiveSubTreeCrossover:
    def __init__(self, max_depth:int, S_train, knowledge):
        self.max_depth = max_depth
        self.data_evaluator = R2Evaluator(S_train)
        self.know_evaluator = KnowledgeEvaluator(knowledge)
    
    def cross(self, parent1:backprop.SyntaxTree, parent2:backprop.SyntaxTree) -> backprop.SyntaxTree:
        nnodes1 = parent1.get_nnodes()
        nnodes2 = parent2.get_nnodes()
        
        child = parent1.clone()
        nodeSelector = backprop.SyntaxTreeNodeSelector(random.randrange(nnodes1))
        child.accept(nodeSelector)
        cross_point1 = nodeSelector.node
        child.set_parent()
        if cross_point1 is None:
            return child
        cross_point1_depth = cross_point1.get_depth()
        max_nesting_depth = (self.max_depth - cross_point1_depth) - 1
        if max_nesting_depth < 0: return child
        
        nodesCollector = backprop.SyntaxTreeNodeCollector()
        parent2.accept(nodesCollector)
        allowedNodes = []
        for node in nodesCollector.nodes:
            if node.get_max_depth() <= max_nesting_depth: allowedNodes.append(node)
        
        if len(allowedNodes) == 0: return child
        best_offspring = None
        best_know_eval = None
        best_data_eval = None
        for n in allowedNodes:
            n_clone = n.clone()
            offspring = replace_subtree(child, cross_point1, n_clone)
            offspring.clear_cache()

            know_eval = self.know_evaluator.evaluate(offspring)
            data_eval = self.data_evaluator.evaluate(offspring)

            child = replace_subtree(child, n_clone, cross_point1)

            #if offspring.is_linear(): continue
            try:
                if offspring.diff().simplify().is_const(): continue
            except RuntimeError:
                continue

            if best_offspring is None or know_eval.better_than(best_know_eval) or data_eval.better_than(best_data_eval):
                best_offspring = offspring
                best_know_eval = know_eval
                best_data_eval = data_eval

        if best_offspring is None:
            return child
        return best_offspring

class KnowledgePropagationCrossover(Crossover):
    def __init__(self, max_depth:int, S_train, knowledge):
        self.max_depth = max_depth
        self.data_evaluator = R2Evaluator(S_train)
        self.know_evaluator = KnowledgeEvaluator(knowledge)
    
    def cross(self, parent1:backprop.SyntaxTree, parent2:backprop.SyntaxTree) -> backprop.SyntaxTree:
        nnodes1 = parent1.get_nnodes()
        nnodes2 = parent2.get_nnodes()
        
        # select subtree from parents.
        cross_points = [parent1, parent2]
        cross_points = [None, None]
        for idx, parent in enumerate([parent1, parent2]):
            nodesCollector = backprop.SyntaxTreeNodeCollector()
            parent.accept(nodesCollector)
            allowedNodes = []
            for node in nodesCollector.nodes:
                if node.get_max_depth() < self.max_depth: allowedNodes.append(node)
            if len(allowedNodes) == 0: return parent1
            cross_points[idx] = random.choice(allowedNodes).clone()
        
        # build best trunk.
        best_child = None
        best_know_eval = None
        best_data_eval = None
        for opt in backprop.BinaryOperatorSyntaxTree.OPERATORS:
            if opt == '^': continue
            
            trunk = backprop.BinaryOperatorSyntaxTree(opt, *cross_points)
            know_eval = self.know_evaluator.evaluate(trunk)
            data_eval = self.data_evaluator.evaluate(trunk)

            if best_child is None or know_eval.better_than(best_know_eval) or data_eval.better_than(best_data_eval):
                best_child = trunk
                best_know_eval = know_eval
                best_data_eval = data_eval
        
        if best_child is None: return parent1
        return best_child
        

class Mutator:
    def mutate(self, stree:backprop.SyntaxTree) -> backprop.SyntaxTree:
        pass

class MultiMutator(Mutator):
    def __init__(self, *mutators):
        self.mutators = list(mutators)
    
    def mutate(self, stree:backprop.SyntaxTree) -> backprop.SyntaxTree:
        return random.choice(self.mutators).mutate(stree)

class SubtreeReplacerMutator(Mutator):
    def __init__(self, max_depth, solutionCreator):
        self.max_depth = max_depth
        self.solutionCreator = solutionCreator
    
    def mutate(self, stree:backprop.SyntaxTree) -> backprop.SyntaxTree:
        nodesCounter = backprop.SyntaxTreeNodeCounter()
        stree.accept(nodesCounter)
        nnodes = nodesCounter.nnodes

        nodeSelector = backprop.SyntaxTreeNodeSelector(random.randrange(nnodes))
        stree.accept(nodeSelector)
        sub_stree = nodeSelector.node

        # replace subtree with random branch.

        stree.set_parent()
        sub_stree_depth = sub_stree.get_depth()
        
        new_sub_stree_depth = self.max_depth - sub_stree_depth
        if new_sub_stree_depth >= 0:
            new_sub_stree = self.solutionCreator.create_population(1, new_sub_stree_depth)[0]
            stree = replace_subtree(stree, sub_stree, new_sub_stree)
        
        return stree.simplify()
    
class FunctionSymbolMutator(Mutator):
    def mutate(self, stree:backprop.SyntaxTree) -> backprop.SyntaxTree:
        nodesCounter = backprop.SyntaxTreeNodeCounter()
        stree.accept(nodesCounter)
        nnodes = nodesCounter.nnodes

        nodeSelector = backprop.SyntaxTreeNodeSelector(random.randrange(nnodes))
        stree.accept(nodeSelector)
        sub_stree = nodeSelector.node
  
        # change a single function symbol.

        if   type(sub_stree) is backprop.BinaryOperatorSyntaxTree:
            sub_stree.operator = random.choice(
                [opt for opt in backprop.BinaryOperatorSyntaxTree.OPERATORS if opt != sub_stree.operator])
        elif type(sub_stree) is backprop.UnaryOperatorSyntaxTree:
            #sub_stree.operator = random.choice(
            #    [opt for opt in backprop.UnaryOperatorSyntaxTree.OPERATORS if opt != sub_stree.operator])
            pass

        return stree

class NumericParameterMutator(Mutator):
    def __init__(self, all:bool=True):
        self.all = all
    
    def mutate(self, stree:backprop.SyntaxTree) -> backprop.SyntaxTree:
        constsCollector = backprop.ConstantSyntaxTreeCollector()
        stree.accept(constsCollector)

        if len(constsCollector.constants) == 0: return stree
        constsToMutate = constsCollector.constants if self.all else [random.choice(constsCollector.constants)]
        for c in constsToMutate:
            c.val += random.gauss(mu=0.0, sigma=1.0)

        return stree


class SolutionCreator:
    def create_population(self, popsize:int, max_depth:int, noconsts:bool=False) -> list[backprop.SyntaxTree]:
        pass

class RandomSolutionCreator(SolutionCreator):
    def __init__(self, nvars:int, randstate:int=None, trunks=None):
        assert nvars > 0
        self.stree_generator = backprop.SyntaxTreeGenerator(randstate, nvars)
        self.trunks = trunks
    
    def create_population(self, popsize:int, max_depth:int, noconsts:bool=False) -> list[backprop.SyntaxTree]:
        assert popsize > 0 and max_depth >= 0
        population = self.stree_generator.create_random(max_depth, popsize, check_duplicates=True, noconsts=noconsts)
        if self.trunks is not None:
            population = [p for p in population if not check_unsat_trunk(self.trunks, p)]
        return population


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
        if type(trunk) is backprop.UnknownSyntaxTree or \
           trunk in all_trunks or \
           check_unsat_trunk(satunsat_trunks, trunk): continue
        all_trunks.append(trunk)
        print(f"Checking trunk: {trunk}")
        sat, _ = lpbackprop.lpbackprop(knowledge, trunk, None)
        if sat:
            satunsat_trunks['sat'].append(trunk)
            print(f"SAT  : {trunk}")
        else:
            satunsat_trunks['unsat'].append(trunk)
            print(f"UNSAT: {trunk}")
    return satunsat_trunks

def check_unsat_trunk(trunks:map, stree) -> bool:
    for unsat_trunk in trunks['unsat']:
        if type(stree) is backprop.BinaryOperatorSyntaxTree and stree.operator == '*' and \
            type(stree.left) is backprop.VariableSyntaxTree and type(stree.right) is backprop.ConstantSyntaxTree and \
                type(unsat_trunk) is backprop.BinaryOperatorSyntaxTree and \
                type(unsat_trunk.left) is backprop.UnknownSyntaxTree and type(unsat_trunk.right) is backprop.UnknownSyntaxTree:
                    print()
        if stree.match(unsat_trunk): return True
    return False


class GPStats:
    def __init__(self, nbests:int=1):
        self.nbests = nbests
        self.bests = []
        self.bests_eval_map = {}
        self.qualities = {'currBest': [], 'currAvg': [], 'currWorst': [], 'best': []}
    
    def update(self, population, eval_map):
        self.__update_bests(population, eval_map)
        self.__update_qualities(population, eval_map)

    def __update_bests(self, population, eval_map):
        """
        It is assumed 'population' is already sorted.
        """
        merged_eval_map = {}
        merged_eval_map.update(eval_map)
        merged_eval_map.update(self.bests_eval_map)
        merged = sort_population(population[:self.nbests] + self.bests, merged_eval_map)
        
        merged_unique = []
        for stree in merged:
            if stree not in merged_unique: merged_unique.append(stree)
        
        self.bests = merged_unique[:min(self.nbests, len(merged_unique))]
        self.bests_eval_map.clear()
        for b in self.bests:
            self.bests_eval_map[id(b)] = merged_eval_map[id(b)]

    def __update_qualities(self, population, eval_map):
        currAvg = 0.0
        for stree in population:
            currAvg += eval_map[id(stree)].get_value()
        currAvg /= len(population)

        self.qualities['currBest' ].append(eval_map[id(population[0])].get_value())    
        self.qualities['currAvg'  ].append(currAvg)
        self.qualities['currWorst'].append(eval_map[id(population[-1])].get_value())
        self.qualities['best'     ].append(self.bests_eval_map[id(self.bests[0])].get_value())

class FUGPStats(GPStats):
    def __init__(self, nbests:int=1):
        super().__init__(nbests)
        self.pland = []
        self.buckets = {}
        self.fea_ratio = {'currBest': [], 'currAvg': [], 'currWorst': [], 'best': []}
    
    def update(self, population, eval_map):
        super().update(population, eval_map)

        pland_ratio = 0.0
        fea_ratio_avg = 0.0
        nconsts = eval_map[id(population[0])].know_n
        for stree in population:
            stree_eval = eval_map[id(stree)]
            if stree_eval.know_ls: pland_ratio += 1
            fea_ratio_avg += stree_eval.fea_ratio
        pland_ratio /= len(population)
        fea_ratio_avg /= len(population)

        self.pland.append(pland_ratio)
        self.fea_ratio['currBest' ].append(eval_map[id(population[0])].fea_ratio)    
        self.fea_ratio['currAvg'  ].append(fea_ratio_avg)
        self.fea_ratio['currWorst'].append(eval_map[id(population[-1])].fea_ratio)
        self.fea_ratio['best'     ].append(self.bests_eval_map[id(self.bests[0])].fea_ratio)


class GP:
    def __init__(self,
                 popsize:int,
                 ngen:int,
                 max_depth:int,
                 S_train:dataset.NumpyDataset,
                 S_test:dataset.NumpyDataset,
                 creator:SolutionCreator,
                 evaluator:Evaluator,
                 selector:Selector,
                 crossover:Crossover,
                 mutator:Mutator,
                 mutrate:float,
                 diversifier:Diversifier=None,
                 elitism:int=0,
                 backprop_intv:int=-1,  # < 0 disabled.
                 knowledge=None,
                 trunks=None,
                 nbests:int=1,
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
        self.diversifier = diversifier
        self.elitism = elitism
        self.backprop_intv = backprop_intv
        self.last_backprop = -1  # last backprop generation idx.
        self.knowledge = knowledge
        self.trunks = trunks
        if rseed is not None:
            random.seed(rseed)
        self.stats = evaluator.create_stats(nbests)
        self.genidx = 0

        # TODO: remove it!
        self.popsize = len(self.population)

        from backprop import gp_backprop
        self.backpropagator = gp_backprop.Backpropagator(S_train, knowledge)
        self.backprop_rate = 1.0

        self.backpropagator = bpropagator.BackPropagator(self.S_train, self.knowledge, self.evaluator)
        
        #for i, p in enumerate(self.population):
        #    self.population[i] = backprop.SemanticSyntaxTree(p(self.S_train.X))

        from backprop.selector import EclipseSelector
        from backprop.crossover import EclipseCrossover
        self.sem_selector = EclipseSelector(S_train)
        self.sem_crossover = EclipseCrossover(self.sem_selector, S_train)
        self.best_sem_solution = None
        self.best_sem_eval = None

        from backprop.pareto_front import DataLengthFrontTracker, MultiHeadFrontTracker
        #self.fea_front_tracker = DataLengthFrontTracker()
        self.fea_front_tracker = MultiHeadFrontTracker()
    
    # returns nbests best strees found + evaluation map.
    def evolve(self) -> tuple[list[backprop.SyntaxTree], dict]:
        print(f"Generation 0")
        #self._backprop(genidx=0)
        logging.info('Start from generation 0')
        self._evaluate_all()
        logging.info('After self.evaluate_all')
        #if self.diversifier is not None:
        #    self.diversifier.diversify(self.population, self.eval_map)
        self.population = sort_population(self.population, self.eval_map)
        logging.info('After pop sort')
        self.stats.update(self.population, self.eval_map)
        logging.info('After stats update')

        for self.genidx in range(1, self.ngen):
            print(f"Generation {self.genidx} {self.eval_map[id(self.population[0])]}")
            #if self.genidx == 60:
            #    self._evaluate_all()
            #    self._replace(self.population)
            #    self.population = sort_population(self.population, self.eval_map)
            children = self._create_children()
            self._replace(children)
            #self._backprop(genidx)
            #if self.diversifier is not None:
            #    self.diversifier.diversify(self.population, self.eval_map)
            
            #self.population = sort_population(self.population, self.eval_map)
            self.stats.update(self.population, self.eval_map)
            #logging.info(f"--- Generation {genidx} [Current best: {self.eval_map[id(self.population[0])]}," + \
            #             f"Global best: {self.bests_eval_map[id(self.bests[0])]}] ---")
        
        print()
        for p in self.population:
            print(p, self.eval_map[id(p)].fea_ratio, self.eval_map[id(p)].data_r2, self.eval_map[id(p)].know_mse)
        return self.stats.bests, self.stats.bests_eval_map
        #return [self.backpropagator.refmod], {id(self.backpropagator.refmod): self.backpropagator.refmod_eval}
    
    def _evaluate_all(self):
        self.eval_map.clear()
        for stree in self.population:
            #self.eval_map[id(stree)] = self.backpropagator.backprop(stree) if self.genidx < 60 else self.evaluator.evaluate(stree)
            self.eval_map[id(stree)] = self.evaluator.evaluate(stree)
    
    def _create_children(self) -> list[backprop.SyntaxTree]:
        
        """best_eval = self.eval_map[id(self.population[0])]
        if self.backpropagator.refmod_eval is None or best_eval.better_than(self.backpropagator.refmod_eval):
            self.backpropagator.update_refmod(self.population[0], best_eval)"""
        
        children = []

        # generate semantic children.
        if self.genidx > 10 and False:
            self.sem_selector.update(self.population)
            sem_child = self.sem_crossover.cross( *self.sem_selector.select(self.population, self.eval_map) )
            sem_child_eval = self.evaluator.evaluate(sem_child)
            #children.append(sem_child)
            #self.eval_map[id(sem_child)] = sem_child_eval
            if self.best_sem_solution is None or sem_child_eval.better_than(self.best_sem_eval):
                self.best_sem_solution = sem_child
                self.best_sem_eval = sem_child_eval
            print(f"Created sem child {sem_child_eval.get_value()} vs {self.eval_map[id(self.population[0])]}")

        while len(children) < self.popsize:
            for _ in range(self.popsize - len(children)):
                parents = self.selector.select(self.population, self.eval_map, 2)
                child = self.crossover.cross(parents[0], parents[1])  # 100% crossover rate (child must be a new object!)
                if random.random() < self.mutrate:
                    child = self.mutator.mutate(child)
                if child.validate() and (self.trunks is None or not check_unsat_trunk(self.trunks, child)): #TODO: and child not in children:
                    
                    """if random.random() < self.backprop_rate:
                        #print(f"Replacing {child}")
                        if self.backpropagator.backprop(child.clone()) is None:
                            #print(f"Unsat child: {child}")
                            continue
                        #print(f"With {child}")"""
                    
                    """if random.random() < 1.0:
                        if type(child) is backprop.ConstantSyntaxTree:
                            continue
                            print(f"Constant: {child}")
                        sat, stree_cost = lpbackprop.lpbackprop(self.knowledge, child, None)
                        #val = child.validate()
                        if not sat:
                            print(f"Unsat child: {child}")
                            continue
                        else:
                            print(f"SAT child: {child}")"""

                    child = child.simplify()

                    if self.diversifier is not None and random.random() < 0.8:
                        child = self.diversifier.diversify(child)

                    child_eval = self.evaluator.evaluate(child)
                    
                    #if self.backpropagator.refmod_eval.data_r2 >= 1.0: # and self.genidx % 5 == 0:
                    
                    #self.backpropagator.propagate(child, child_eval, True)
                    
                    #new_child_eval = self.backpropagator.propagate(child, child_eval)
                    #if new_child_eval is not None: child_eval = new_child_eval
                    
                    #child_eval.data_r2 = max(child_eval.data_r2, child.best_match_r2)
                    
                    children.append(child)
                    #self.eval_map[id(child)] = self.backpropagator.backprop(child) if self.genidx < 60 else self.evaluator.evaluate(child)
                    self.eval_map[id(child)] = child_eval

                    if child_eval.fea_ratio == 1.0:
                        self.fea_front_tracker.track(child, child_eval)

                    #print(f"From parents {parents[0]} and {parents[1]} got {child}")
                    #self.eval_map[id(child)] = self.evaluator.evaluate(child)

        if self.best_sem_solution is not None and self.genidx == self.ngen - 1:
            children.append(self.best_sem_solution)
            self.eval_map[id(self.best_sem_solution)] = self.best_sem_eval
        
        return children
    
    def _replace(self, children:list[backprop.SyntaxTree]):
        if self.elitism > 0:
            children = sort_population(children, self.eval_map)
            for i in range(self.elitism):
                children[-1-i] = self.population[i]
        self.population = children  # generational replacement.

        self.population = sort_population(self.population, self.eval_map)

        self.population = (self.fea_front_tracker.get_population() + self.population)[:self.popsize]
        #if self.genidx % 2 == 0:
        #self.population = (self.fea_front_tracker.front_tracker_a.get_population() + self.population)[:self.popsize]
        #else:
        #    self.population = (self.fea_front_tracker.front_tracker_b.get_population() + self.population)[:self.popsize]
        
        # update evaluation map based on new population.
        new_eval_map = {}
        for stree in self.population:
            new_eval_map[id(stree)] = self.eval_map[id(stree)]
        self.eval_map = new_eval_map
    
    """
    def _backprop(self, genidx):
        if self.backprop_intv < 0 or (genidx - self.last_backprop) < self.backprop_intv:
            return
        for stree_idx, stree in enumerate(self.population):
            
            nodesCounter = backprop.SyntaxTreeNodeCounter()
            stree.accept(nodesCounter)
            nnodes = nodesCounter.nnodes
            
            stree_backprop = stree.clone()
            nodeSelector = backprop.SyntaxTreeNodeSelector(random.randrange(nnodes))
            stree_backprop.accept(nodeSelector)
            
            stree_backprop = replace_subtree(stree_backprop, nodeSelector.node, backprop.UnknownSyntaxTree())
            print(f"Backprop: {stree_backprop}")
            
            try:
                all_derivs = self.knowledge.get_derivs()
                stree_backprop_map = backprop.SyntaxTree.diff_all(stree_backprop, all_derivs, include_zeroth=True)
            except RuntimeError():
                continue

            best_unkn_models = {}
            best_eval = None

            def onsynth_callback(synth_unkn_models:dict):
                nonlocal best_unkn_models
                nonlocal best_eval
                
                for unkn in synth_unkn_models.keys():
                    unkn_model, coeffs_mask, constrs = synth_unkn_models[unkn]
                
                #try:
                hist, __best_unkn_models, __best_eval = jump_backprop.jump_backprop(stree_backprop_map, synth_unkn_models, self.S_train, self.S_test, max_rounds=1)

                if best_eval is None or __best_eval.better_than(best_eval):
                    best_unkn_models = __best_unkn_models
                    best_eval = __best_eval
                #except RuntimeError:
                #    pass

            lpbackprop.lpbackprop(self.knowledge, stree_backprop, onsynth_callback)

            if best_eval is not None:
                for unkn_label in best_unkn_models.keys():
                    stree_backprop.set_unknown_model(unkn_label, best_unkn_models[unkn_label])
                print(f"SAT: {stree_backprop}")
                self.population[stree_idx] = stree_backprop
                if id(stree) in self.eval_map: del self.eval_map[id(stree)]
                self.eval_map[id(stree_backprop)] = self.evaluator.evaluate(stree_backprop)
        
        self.last_backprop = genidx
    """
        