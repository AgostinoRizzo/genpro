from symbols.syntax_tree import SyntaxTree


class EvolutionTracker:
    def __init__(self):
        self.crossovers = []
        self.evaluations = []

    def track_crossover(self, parent1:SyntaxTree, parent2:SyntaxTree, child:SyntaxTree, eval_map:dict):
        self.crossovers.append((parent1, parent2, child))
        self.evaluations.append((eval_map[id(parent1)], eval_map[id(parent2)], eval_map[id(child)]))