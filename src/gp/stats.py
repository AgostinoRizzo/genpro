import matplotlib.pyplot as plt
import numpy as np

from backprop.bperrors import BackpropError
from backprop.library import LibraryError


def series_float_to_string(series, ndecimals:int=4, sep:str=':'):
    return sep.join(format(x, f".{ndecimals}f") for x in series)

def series_int_to_string(series, sep:str=':'):
    return sep.join(str(x) for x in series)


"""
this class acts according to the Chain of Responsibility pattern.
"""
class GPStats:
    def __init__(self, next=None):
        self.next = next
    
    def update(self, gp):
        if self.next is not None:
            self.next.update(gp)
    
    def get_qualities_stats(self):
        if self.next is not None:
            return self.next.get_qualities_stats()
    
    def get_feasibility_stats(self):
        if self.next is not None:
            return self.next.get_feasibility_stats()
    
    def get_properties_stats(self):
        if self.next is not None:
            return self.next.get_properties_stats()
    
    def plot(self):
        if self.next is not None:
            self.next.plot()


class QualitiesGPStats(GPStats):
    def __init__(self, best_qual:float, worst_qual:float, qual_name:str, next=None):
        super().__init__(next)
        assert best_qual != worst_qual
        self.best_qual = best_qual
        self.worst_qual = worst_qual
        self.minimize = self.best_qual < self.worst_qual
        self.qual_name = qual_name
        self.best = None
        self.best_eval = None
        self.qualities = {'currBest': [], 'currAvg': [], 'currWorst': [], 'best': [], 'currTop': []}
    
    def update(self, gp):
        super().update(gp)

        population = gp.population
        eval_map = gp.eval_map
        currBest  = self.worst_qual
        currWorst = self.best_qual
        currAvg   = 0.0
        count = 0

        for stree in population:
            stree_eval = eval_map[id(stree)].get_quality()
            stree_eval_value = stree_eval.get_value()

            if  (self.minimize     and stree_eval_value > self.worst_qual) or \
                (not self.minimize and stree_eval_value < self.worst_qual):
                stree_eval_value = self.worst_qual
            
            if np.isfinite(stree_eval_value):
                if self.minimize:
                    currBest   = min(currBest, stree_eval_value)
                    currWorst  = max(currWorst, stree_eval_value)
                else:
                    currBest   = max(currBest, stree_eval_value)
                    currWorst  = min(currWorst, stree_eval_value)
                currAvg += stree_eval_value
                count += 1

            if self.best is None or stree_eval.better_than(self.best_eval):
                self.best = stree
                self.best_eval = stree_eval

        if count > 0: currAvg /= count
        else: currAvg = np.nan

        currTop = eval_map[id(population[0])].get_quality().get_value()  # best according to the fitness function (population[0]).

        self.qualities['currBest' ].append(currBest)    
        self.qualities['currAvg'  ].append(currAvg)
        self.qualities['currWorst'].append(currWorst)
        self.qualities['currTop'  ].append(currTop)  
        #self.qualities['best'     ].append(self.bests_eval_map[id(self.bests[0])].get_value())

    def get_qualities_stats(self):
        return self
    
    def plot(self):
        super().plot()

        for quality, qseries in self.qualities.items():
            plt.plot(qseries, label=quality)

        plt.legend()
        plt.xlabel('Generation')
        plt.ylabel(self.qual_name)
        plt.title('Qualities')
        plt.show()
        

class FeasibilityGPStats(GPStats):
    def __init__(self, next=None):
        super().__init__(next)
        self.buckets = {}
        self.fea_ratio = {'currBest': [], 'currAvg': [], 'currWorst': [], 'best': [], 'prop': [], 'currTop': []}
    
    def update(self, gp):
        super().update(gp)

        population = gp.population
        eval_map = gp.eval_map

        fea_ratio_best  = 0.0
        fea_ratio_avg   = 0.0
        fea_ratio_worst = 1.0
        fea_prop        = 0.0
        
        for stree in population:
            fea_ratio = eval_map[id(stree)].fea_ratio

            fea_ratio_best   = max(fea_ratio_best, fea_ratio)
            fea_ratio_avg   += fea_ratio
            fea_ratio_worst  = min(fea_ratio_worst, fea_ratio)
            if fea_ratio == 1.0: fea_prop += 1
        
        fea_ratio_avg /= len(population)
        fea_prop /= len(population)

        fea_ratio_top = eval_map[id(population[0])].fea_ratio  # best according to the fitness function (population[0]).

        self.fea_ratio['currBest' ].append(fea_ratio_best)    
        self.fea_ratio['currAvg'  ].append(fea_ratio_avg)
        self.fea_ratio['currWorst'].append(fea_ratio_worst)
        self.fea_ratio['prop'     ].append(fea_prop)
        self.fea_ratio['currTop'  ].append(fea_ratio_top)
        #self.fea_ratio['best'     ].append(self.bests_eval_map[id(self.bests[0])].fea_ratio)

    def get_feasibility_stats(self):
        return self
    
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


class PropertiesGPStats(GPStats):
    def __init__(self, next=None):
        super().__init__(next)
        self.lengths = {'currTop': []}
    
    def update(self, gp):
        super().update(gp)
        currTop = gp.population[0].get_nnodes()  # best according to the fitness function (population[0]).
        self.lengths['currTop'].append(currTop)
    
    def get_properties_stats(self):
        return self
    
    def plot(self):
        super().plot()

        for lengths, lseries in self.lengths.items():
            plt.plot(lseries, label=lengths)

        plt.legend()
        plt.xlabel('Generation')
        plt.ylabel('#nodes')
        plt.title('Length')
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
