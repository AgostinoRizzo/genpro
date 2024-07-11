from functools import cmp_to_key
import random
import logging
import numpy as np

import dataset
from backprop import backprop, lpbackprop, jump_backprop


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
    def __str__(self) -> str:
        return f"{self.value}"


def random_population(popsize:int, max_depth:int, nvars:int=1, check_duplicates:bool=True, randstate:int=None) -> list[backprop.SyntaxTree]:
    assert popsize >= 1
    return backprop.SyntaxTreeGenerator(randstate, nvars).create_random(max_depth, popsize, check_duplicates)

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
    def evaluate(self, stree:backprop.SyntaxTree):
        return None


class R2Evaluator:
    def __init__(self, dataset, minimize:bool=True):
        self.dataset = dataset
        self.minimize = False
    
    def evaluate(self, stree:backprop.SyntaxTree):
        return RealEvaluation(max(0., self.dataset.evaluate(stree)['r2']), self.minimize)


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
        nnodes2 = parent2.get_nnodes()
        
        child = parent1.clone()
        nodeSelector = backprop.SyntaxTreeNodeSelector(random.randrange(nnodes1))
        child.accept(nodeSelector)
        cross_point1 = nodeSelector.node
        child.set_parent()
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
        
        #print(f"From {child}")
        #print(f"\tReplacing {cross_point1}")
        #print(f"\tWith {cross_point2}")
        got = replace_subtree(child, cross_point1, cross_point2.clone())
        got.clear_cache()
        #print(f"Got {got}")
        #if got.get_max_depth() > self.max_depth:
        #    raise RuntimeError(f"Max depth: {got.get_max_depth()}, {max_nesting_depth}, {cross_point1.get_depth()}, {cross_point2.get_depth()}")
        return got
        

class Mutator:
    def mutate(self, stree:backprop.SyntaxTree) -> backprop.SyntaxTree:
        pass

class MultiMutator(Mutator):
    def __init__(self, *mutators):
        self.mutators = list(mutators)
    
    def mutate(self, stree:backprop.SyntaxTree) -> backprop.SyntaxTree:
        return random.choice(self.mutators).mutate(stree)

class SubtreeReplacerMutator(Mutator):
    def __init__(self, max_depth:int):
        self.max_depth = max_depth
    
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
            new_sub_stree = backprop.SyntaxTreeGenerator().create_random(new_sub_stree_depth, 1)[0]
            stree = replace_subtree(stree, sub_stree, new_sub_stree)
        
        return stree
    
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
            sub_stree.operator = random.choice(
                [opt for opt in backprop.UnaryOperatorSyntaxTree.OPERATORS if opt != sub_stree.operator])

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

class GP:
    def __init__(self,
                 population:list[backprop.SyntaxTree],
                 ngen:int,
                 S_train:dataset.NumpyDataset,
                 S_test:dataset.NumpyDataset,
                 evaluator:Evaluator,
                 selector:Selector,
                 crossover:Crossover,
                 mutator:Mutator,
                 mutrate:float,
                 elitism:int=0,
                 backprop_intv:int=-1,  # < 0 disabled.
                 knowledge=None,
                 nbests:int=1,
                 rseed=None):
        self.population = population
        self.eval_map = {}
        self.popsize = len(population)
        self.ngen = ngen
        self.S_train = S_train
        self.S_test = S_test
        self.evaluator = evaluator
        self.selector = selector
        self.crossover = crossover
        self.mutator = mutator
        self.mutrate = mutrate
        self.elitism = elitism
        self.backprop_intv = backprop_intv
        self.last_backprop = -1  # last backprop generation idx.
        self.knowledge = knowledge
        self.bests = []
        self.bests_eval_map = {}
        self.nbests = nbests
        if rseed is not None:
            random.seed(rseed)
        self.qualities = {'currBest': [], 'currAvg': [], 'currWorst': [], 'best': []}
    
    # returns nbests best strees found + evaluation map.
    def evolve(self) -> tuple[list[backprop.SyntaxTree], dict]:
        print(f"Generation 0")
        #self.__backprop(genidx=0)
        self.__evaluate_all()
        self.population = sort_population(self.population, self.eval_map)
        self.__update_bests()
        self.__update_qualities()

        for genidx in range(1, self.ngen):
            print(f"Generation {genidx}")
            children = self.__create_children()
            self.__replace(children)
            #self.__backprop(genidx)
            self.population = sort_population(self.population, self.eval_map)
            self.__update_bests()
            self.__update_qualities()
            #logging.info(f"--- Generation {genidx} [Current best: {self.eval_map[id(self.population[0])]}," + \
            #             f"Global best: {self.bests_eval_map[id(self.bests[0])]}] ---")
        
        return self.bests, self.bests_eval_map

    def __update_bests(self):
        """
        It is assumed self.population is already sorted.
        """
        merged_eval_map = {}
        merged_eval_map.update(self.eval_map)
        merged_eval_map.update(self.bests_eval_map)
        merged = sort_population(self.population[:self.nbests] + self.bests, merged_eval_map)
        
        merged_unique = []
        for stree in merged:
            if stree not in merged_unique: merged_unique.append(stree)
        
        self.bests = merged_unique[:min(self.nbests, len(merged_unique))]
        self.bests_eval_map.clear()
        for b in self.bests:
            self.bests_eval_map[id(b)] = merged_eval_map[id(b)]
    
    def __evaluate_all(self):
        self.eval_map.clear()
        for stree in self.population:
            self.eval_map[id(stree)] = self.evaluator.evaluate(stree)
    
    def __create_children(self) -> list[backprop.SyntaxTree]:
        children = []
        while len(children) < self.popsize:
            for _ in range(self.popsize - len(children)):
                parents = self.selector.select(self.population, self.eval_map, 2)
                child = self.crossover.cross(parents[0], parents[1])  # 100% crossover rate (child must be a new object!)
                if random.random() < self.mutrate:
                    child = self.mutator.mutate(child)
                if child.validate(): #TODO: and child not in children:
                    children.append(child)
                    #print(f"From parents {parents[0]} and {parents[1]} got {child}")
                    self.eval_map[id(child)] = self.evaluator.evaluate(child)
        return children
    
    def __replace(self, children:list[backprop.SyntaxTree]):
        if self.elitism > 0:
            children = sort_population(children, self.eval_map)
            for i in range(self.elitism):
                children[-1-i] = self.population[i]
        self.population = children  # generational replacement.
        
        # update evaluation map based on new population.
        new_eval_map = {}
        for stree in self.population:
            new_eval_map[id(stree)] = self.eval_map[id(stree)]
        self.eval_map = new_eval_map
    
    """
    def __backprop(self, genidx):
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

    def __update_qualities(self):
        currAvg = 0.0
        for stree in self.population:
            currAvg += self.eval_map[id(stree)].value
        currAvg /= self.popsize

        self.qualities['currBest' ].append(self.eval_map[id(self.population[0])].value)    
        self.qualities['currAvg'  ].append(currAvg)
        self.qualities['currWorst'].append(self.eval_map[id(self.population[-1])].value)
        self.qualities['best'     ].append(self.bests_eval_map[id(self.bests[0])].value)
        