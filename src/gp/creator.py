import random
from symbols.syntax_tree import SyntaxTree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.unaop import UnaryOperatorSyntaxTree
from symbols.var import VariableSyntaxTree
from symbols.const import ConstantSyntaxTree
from symbols.generator import SyntaxTreeGenerator
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


def createRandomTerminal(cl:float, cu:float, nvars:int):
    if random.random() < 0.5:
        return ConstantSyntaxTree(val=random.uniform(cl, cu))
    return VariableSyntaxTree(idx=random.randrange(nvars))

def createRandomNonTerminal(max_nchildren:int):
    operator = None
    if max_nchildren == 1:
        operator = random.choice(SyntaxTreeGenerator.UNA_OPERATORS)
    else:
        operator = random.choice(SyntaxTreeGenerator.OPERATORS)
    
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
    

def ptc2(target_len:int, max_depth:int, cl:float, cu:float, nvars:int):
    if target_len <= 0 or max_depth < 0:
        raise ValueError('Invalid target_len or max_depth.')
    
    if target_len == 1 or max_depth == 0:
        return createRandomTerminal(cl, cu, nvars)
    
    target_len -= 1
    root, Q = createRandomNonTerminal(target_len)
    
    while len(Q) > 0 and len(Q) < target_len:
        i = random.randrange(len(Q))
        extpoint = Q[i]
        Q.pop(i)

        target_len -= 1

        if extpoint.parent.get_depth() == max_depth - 1:
            extpoint.extend(createRandomTerminal(cl, cu, nvars))
            continue
        
        child, __Q = createRandomNonTerminal(target_len - len(Q))  # TODO: check validity.
        extpoint.extend(child)

        Q += __Q

    for extpoint in Q:
        extpoint.extend(createRandomTerminal(cl, cu, nvars))
    
    return root


class SolutionCreator:
    def create_population(self, popsize:int, max_depth:int, noconsts:bool=False) -> list[SyntaxTree]:
        pass


class RandomSolutionCreator(SolutionCreator):
    def __init__(self, nvars:int, trunks=None, y_iqr:float=1.0):
        assert nvars > 0
        self.stree_generator = SyntaxTreeGenerator(nvars, y_iqr)
        self.trunks = trunks
    
    def create_population(self, popsize:int, max_depth:int, noconsts:bool=False) -> list[SyntaxTree]:
        assert popsize > 0 and max_depth >= 0
        population = self.stree_generator.create_random(max_depth, popsize, check_duplicates=True, noconsts=noconsts)
        if self.trunks is not None:
            population = [p for p in population if not utils.check_unsat_trunk(self.trunks, p)]
            left = popsize - len(population)
            if left > 0: population += self.create_population(left, max_depth, noconsts)
        return population


class PTC2RandomSolutionCreator(SolutionCreator):
    def __init__(self, nvars:int, y_iqr:float=1.0):
        assert nvars > 0
        self.nvars = nvars
        self.cl, self.cu = -y_iqr, y_iqr
        self.max_length = 20  # TODO: generalize it.
    
    def create_population(self, popsize:int, max_depth:int) -> list[SyntaxTree]:
        assert popsize > 0 and max_depth >= 0
        population = []
        for _ in range(popsize):
            target_len = random.randint(1, self.max_length)
            stree = ptc2(target_len, max_depth, self.cl, self.cu, self.nvars)
            population.append(stree.simplify())
        return population
