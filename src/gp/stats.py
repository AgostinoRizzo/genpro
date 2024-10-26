import matplotlib.pyplot as plt
import numpy as np

from backprop.bperrors import BackpropError
from backprop.library import LibraryError


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
        self.softening_rate = []
        self.error_rate = {}
        
        self.ntrials = 0
        self.ncorrections = 0
        self.nsoftenings = 0
        self.nerrors = {}

        for e in BackpropError.__subclasses__() + LibraryError.__subclasses__():
            self.error_rate[e.__name__] = []
            self.nerrors[e.__name__] = 0
        
    def update(self, gp):
        super().update(gp)

        rate = np.nan
        if gp.genidx > 0:
            rate = 0.0 if self.ntrials == 0.0 else (self.ncorrections / self.ntrials)
        self.correction_rate.append(rate)

        rate = np.nan
        if gp.genidx > 0:
            rate = 0.0 if self.ntrials == 0.0 else (self.nsoftenings / self.ntrials)
        self.softening_rate.append(rate)

        for e_name, n in self.nerrors.items():
            rate = np.nan
            if gp.genidx > 0:
                rate = 0.0 if n == 0.0 else (n / self.ntrials)
            self.error_rate[e_name].append(rate)
        
        self.ntrials = 0
        self.ncorrections = 0
        self.nsoftenings = 0
        for e_name in self.nerrors:
            self.nerrors[e_name] = 0
    
    def on_correction(self, C_pulled):
        self.ntrials += 1
        self.ncorrections += 1
        if C_pulled.are_none(): self.nsoftenings += 1
    
    def on_backprop_error(self, e:BackpropError):
        self.__on_error(e)
    
    def on_library_error(self, e:LibraryError):
        self.__on_error(e)
    
    def __on_error(self, e):
        self.ntrials += 1
        self.nerrors[e.__class__.__name__] += 1
    
    def plot(self):
        super().plot()

        plt.plot(self.correction_rate, label='correction-rate')
        plt.plot(self.softening_rate, label='softening-rate')
        for e_name, rate in self.error_rate.items():
            plt.plot(rate, label=e_name)
        
        plt.legend()
        plt.ylim((-0.01, 1.01))
        plt.xlabel('Generation')
        plt.ylabel('Rate')
        plt.title('Correction Rate')
        plt.show()
