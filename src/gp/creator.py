import random
from symbols.syntax_tree import SyntaxTree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.unaop import UnaryOperatorSyntaxTree
from symbols.var import VariableSyntaxTree
from symbols.const import ConstantSyntaxTree
from symbols.generator import SyntaxTreeGenerator
from symbols.grammar import get_nesting_operators, get_una_nesting_operators
from gp import utils


class ExtensionPoint():
    def __init__(self):
        self.parent = None
        self.isunary = False
        self.isleft = False
    
    def set_unary_parent(self, parent):
        self.parent = parent
        self.isunary = True
        self.isleft = False
    
    def set_binary_parent(self, parent, isleft:bool):
        self.parent = parent
        self.isunary = False
        self.isleft = isleft
    
    def extend(self, child):
        if self.isunary:
            self.parent.inner = child
        elif self.isleft:
            self.parent.left = child
        else:
            self.parent.right = child
        child.parent = self.parent


def createRandomTerminal(cl:float, cu:float, nvars:int, create_consts:bool=True, const_prob:float=0.5):
    if create_consts and const_prob > 0.0 and random.random() < const_prob:
        return ConstantSyntaxTree(val=random.uniform(cl, cu))
    return VariableSyntaxTree(idx=random.randrange(nvars))

def createRandomNonTerminal(max_nchildren:int, parent_opt:str=None):
    operator = None
    if max_nchildren == 1:
        operator = random.choice(get_una_nesting_operators(parent_opt))
    else:
        operator = random.choice(get_nesting_operators(parent_opt))
    
    if operator in UnaryOperatorSyntaxTree.OPERATORS:
        p = ExtensionPoint()
        t = UnaryOperatorSyntaxTree(operator, p)
        p.set_unary_parent(t)
        return t, [p]
    
    p1, p2 = ExtensionPoint(), ExtensionPoint()
    nt = BinaryOperatorSyntaxTree(operator, p1, p2)
    p1.set_binary_parent(nt, isleft=True)
    p2.set_binary_parent(nt, isleft=False)
    return nt, [p1, p2]
    

def ptc2(target_len:int, max_depth:int, cl:float, cu:float, nvars:int, create_consts:bool, parent_opt:str=None, const_prob:float=0.5):
    if target_len <= 0 or max_depth < 0:
        raise ValueError('Invalid target_len or max_depth.')
    
    if target_len == 1 or max_depth == 0:
        return createRandomTerminal(cl, cu, nvars, create_consts, const_prob=const_prob)
    
    target_len -= 1
    root, Q = createRandomNonTerminal(target_len, parent_opt)
    
    while len(Q) > 0 and len(Q) < target_len:
        i = random.randrange(len(Q))
        extpoint = Q[i]
        Q.pop(i)

        target_len -= 1

        if extpoint.parent.get_depth() == max_depth - 1:
            extpoint.extend(createRandomTerminal(cl, cu, nvars, const_prob=const_prob))
            continue
        
        child, __Q = createRandomNonTerminal(target_len - len(Q), extpoint.parent.operator)
        extpoint.extend(child)

        Q += __Q

    for extpoint in Q:
        extpoint.extend(createRandomTerminal(cl, cu, nvars, const_prob=const_prob))
    
    return root


class SolutionCreator:
    def create_population(self, popsize:int, max_depth:int, max_length:int, create_consts:bool=True, parent_opt:str=None) -> list[SyntaxTree]:
        pass


class RandomSolutionCreator(SolutionCreator):
    def __init__(self, nvars:int, trunks=None, y_iqr:float=1.0):
        assert nvars > 0
        self.stree_generator = SyntaxTreeGenerator(nvars, y_iqr)
        self.trunks = trunks
    
    def create_population(self, popsize:int, max_depth:int, max_length:int, create_consts:bool=True, parent_opt:str=None) -> list[SyntaxTree]:
        assert popsize > 0 and max_depth >= 0
        population = self.stree_generator.create_random(max_depth, popsize, check_duplicates=True)
        if self.trunks is not None:
            population = [p for p in population if not utils.check_unsat_trunk(self.trunks, p)]
            left = popsize - len(population)
            if left > 0: population += self.create_population(left, max_depth, max_length)
        return population


class PTC2RandomSolutionCreator(SolutionCreator):
    def __init__(self, nvars:int, cl:float=-1.0, cu:float=1.0, simplify:bool=True, const_prob:float=0.5):
        assert nvars > 0
        self.nvars = nvars
        self.cl, self.cu = cl, cu
        self.simplify = simplify
        self.const_prob = const_prob
    
    def create_population(self, popsize:int, max_depth:int, max_length:int, create_consts:bool=True, parent_opt:str=None) -> list[SyntaxTree]:
        assert popsize > 0 and max_depth >= 0 and max_length > 0
        population = []
        population_set = set()
        while len(population) < popsize:
            target_len = random.randint(1, max_length)
            stree = ptc2(target_len, max_depth, self.cl, self.cu, self.nvars, create_consts, parent_opt, const_prob=self.const_prob)
            stree = stree.simplify() if self.simplify else stree
            if not create_consts and type(stree) is ConstantSyntaxTree:
                continue
            #stree_hash = stree.get_hash()
            #if stree_hash not in population_set:
            population.append(stree)
            #    population_set.add(stree_hash)
        return population
