from symbols.syntax_tree import SyntaxTree
from symbols.generator import SyntaxTreeGenerator
from gp import utils

class SolutionCreator:
    def create_population(self, popsize:int, max_depth:int, noconsts:bool=False) -> list[SyntaxTree]:
        pass

class RandomSolutionCreator(SolutionCreator):
    def __init__(self, nvars:int, trunks=None):
        assert nvars > 0
        self.stree_generator = SyntaxTreeGenerator(nvars)
        self.trunks = trunks
    
    def create_population(self, popsize:int, max_depth:int, noconsts:bool=False) -> list[SyntaxTree]:
        assert popsize > 0 and max_depth >= 0
        population = self.stree_generator.create_random(max_depth, popsize, check_duplicates=True, noconsts=noconsts)
        if self.trunks is not None:
            population = [p for p in population if not utils.check_unsat_trunk(self.trunks, p)]
            left = popsize - len(population)
            if left > 0: population += self.create_population(left, max_depth, noconsts)
        return population