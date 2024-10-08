import random
from symbols.syntax_tree import SyntaxTree
from gp import utils

class Selector:
    def select(self, population:list[SyntaxTree], eval_map:dict, nparents:int=2) -> list[SyntaxTree]:
        return None


class TournamentSelector(Selector):
    def __init__(self, group_size:int):
        self.group_size = group_size
    
    def select(self, population:list[SyntaxTree], eval_map:dict, nparents:int=2) -> list[SyntaxTree]:
        parents = []
        for _ in range(nparents):
            group = random.choices(population, k=self.group_size)
            sorted_group = utils.sort_population(group, eval_map)
            parents.append(sorted_group[0])
        return parents


class EclipseSelector(Selector):
    def __init__(self, data):
        self.data = data
        self.sT = data.y
        self.cross_coeffs = {}
        self.cross_map = {}
        self.next_selidx = -1
        
    def update(self, population:list[SyntaxTree]):
        self.cross_coeffs = {}
        self.cross_map = {}

        sem_map = {}
        for p in population:
            sem_map[id(p)] = p(self.data.X)
        
        popsize = len(population)

        for i in range(popsize):
            p1 = population[i]
            s1 = sem_map[id(p1)]

            if np.isnan(s1).any() or np.isinf(s1).any() or (s1 == 0).all():
                continue

            for j in range(i+1, popsize):
                p2 = population[j]
                s2 = sem_map[id(p2)]
                
                if np.isnan(s2).any() or np.isinf(s2).any() or (s2 == 0).all():
                    continue
                
                r1T = self.sT - s1
                r12 = s2 - s1

                #cross_val = np.arccos( np.dot(r1T, r12) / (np.linalg.norm(r1T) * np.linalg.norm(r12)) )

                A = np.column_stack((s1, s2))
                b = self.sT
                
                result = np.linalg.lstsq(A, b)
                residuals = result[1]
                if residuals.size == 0: continue

                self.cross_coeffs[(id(p1), id(p2))] = result[0]
                cross_val = residuals[0]
                self.cross_map[cross_val] = (p1, p2)
        
        self.sorted_cross_val = sorted(self.cross_map.keys(), reverse=True)
        self.next_selidx = len(self.sorted_cross_val) - 1

    def select(self, population:list[SyntaxTree], eval_map:dict, nparents:int=2) -> list[SyntaxTree]:
        assert nparents == 2 and self.next_selidx >= 0

        selidx = self.next_selidx
        self.next_selidx -= 1
        return self.cross_map[ self.sorted_cross_val[selidx] ]
