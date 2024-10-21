import matplotlib.pyplot as plt
import numpy as np


"""
this class acts according to the Chain of Responsibility pattern.
"""
class GPStats:
    def __init__(self, next=None):
        self.next = next
    
    def update(self, gp):
        if self.next is not None:
            self.next.update(gp)
    
    def plot(self):
        if self.next is not None:
            self.next.plot()


class QualitiesGPStats(GPStats):
    def __init__(self, min_qual:float, max_qual:float, qual_name:str, next=None):
        super().__init__(next)
        self.min_qual = min_qual
        self.max_qual = max_qual
        self.qual_name = qual_name
        self.best = None
        self.best_eval = None
        self.qualities = {'currBest': [], 'currAvg': [], 'currWorst': [], 'best': []}
    
    def update(self, gp):
        super().update(gp)

        population = gp.population
        eval_map = gp.eval_map
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

    def plot(self):
        super().plot()

        for quality, qseries in self.qualities.items():
            plt.plot(qseries, label=quality)

        plt.legend()
        plt.ylim((self.min_qual-0.01, self.max_qual+0.01))
        plt.xlabel('Generation')
        plt.ylabel(self.qual_name)
        plt.title('Qualities')
        plt.show()
        

class FeasibilityGPStats(GPStats):
    def __init__(self, next=None):
        super().__init__(next)
        self.buckets = {}
        self.fea_ratio = {'currBest': [], 'currAvg': [], 'currWorst': [], 'best': []}
    
    def update(self, gp):
        super().update(gp)

        population = gp.population
        eval_map = gp.eval_map

        fea_ratio_best  = 0.0
        fea_ratio_avg   = 0.0
        fea_ratio_worst = 1.0
        
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

    def plot(self):
        super().plot()

        for quality, qseries in self.fea_ratio.items():
            plt.plot(qseries, label=quality)

        plt.legend()
        plt.ylim((-0.01, 1.01))
        plt.xlabel('Generation')
        plt.ylabel('Ratio')
        plt.title('Feasibility')
        plt.show()


class CorrectorGPStats(GPStats):
    def __init__(self, next=None):
        super().__init__(next)
        self.correction_rate = []
        self.ncorrections = 0
        self.successful_corrections = 0
    
    def update(self, gp):
        super().update(gp)

        rate = np.nan
        if gp.genidx > 0:
            rate = 0.0 if self.ncorrections == 0.0 else (self.successful_corrections / self.ncorrections)
        self.correction_rate.append(rate)
        self.ncorrections = 0
        self.successful_corrections = 0
    
    def on_correction(self, success_status:bool=True):
        if success_status:
            self.successful_corrections += 1
        self.ncorrections += 1
    
    def plot(self):
        super().plot()

        plt.plot(self.correction_rate)
        
        plt.ylim((-0.01, 1.01))
        plt.xlabel('Generation')
        plt.ylabel('Rate')
        plt.title('Correction Rate')
        plt.show()
