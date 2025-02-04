from functools import cmp_to_key
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from symbols.visitor import SyntaxTreeIneqOperatorCollector
from gp.evaluation import LayeredEvaluation


"""class SymbolicFrequencies:
    def __init__(self):
        self.freq = {}
    
    def add(self, stree):
        symbset = SymbolicFrequencies.get_symbset(stree)
        if symbset not in self.freq:
            self.freq[symbset] = 1
        else:
            self.freq[symbset] += 1

    def remove(self, stree):
        symbset = SymbolicFrequencies.get_symbset(stree)
        if symbset not in self.freq: return
        if self.freq[symbset] <= 1:
            del self.freq[symbset]
        else:
            self.freq[symbset] -= 1
    
    def compute_similarity(self, stree) -> int:
        symbset = SymbolicFrequencies.get_symbset(stree)
        if symbset not in self.freq: return 0
        return self.freq[symbset]
    
    @staticmethod
    def get_symbset(stree):
        optCollector = SyntaxTreeIneqOperatorCollector()
        stree.accept(optCollector)
        return frozenset(optCollector.opts)"""


class SymbolicFrequencies:
    def __init__(self):
        self.freq = {}
    
    def add(self, stree):
        symbset = SymbolicFrequencies.get_symbset(stree)
        for s in symbset:
            if s not in self.freq:
                self.freq[s] = 1
            else:
                self.freq[s] += 1

    def remove(self, stree):
        symbset = SymbolicFrequencies.get_symbset(stree)
        for s in symbset:
            if s not in self.freq: return
            if self.freq[s] <= 1:
                del self.freq[s]
            else:
                self.freq[s] -= 1
    
    def compute_similarity(self, stree) -> int:
        symbset = SymbolicFrequencies.get_symbset(stree)
        simil = 0
        
        for s in symbset:
            if s in self.freq:
                simil += self.freq[s]
        
        for s, f in self.freq.items():
            if s not in symbset:
                simil -= f
        
        return simil
    
    @staticmethod
    def get_symbset(stree):
        optCollector = SyntaxTreeIneqOperatorCollector()
        stree.accept(optCollector)
        return frozenset(optCollector.opts)


class FrontDuplicateError(RuntimeError):
    pass


class DataLengthFrontTracker:

    def __init__(self, popsize:int, max_fronts=np.inf):
        assert popsize > 0
        self.popsize = popsize
        self.max_fronts = max_fronts
        self.front = {}
        self.eval_map = {}
        self.symbfreq = SymbolicFrequencies()

    def track(self, stree, datalength_eval:tuple, evaluation, frontidx:int=0, tracked_counter:int=0):
        if frontidx >= self.max_fronts or self.popsize == 0:
            return
        if frontidx not in self.front:
            self.front[frontidx] = []
        
        stree_data, stree_length = datalength_eval
        stree_data = 1-stree_data # adjustment for now

        to_remove = []
        is_dominated = False

        curr_front = self.front[frontidx]
        tracked_counter += len(curr_front)

        for idx, (_, data, length) in enumerate(curr_front):
            if (stree_data >= data and stree_length < length) or (stree_data > data and stree_length <= length):
                to_remove.append(idx)
            
            if  data == stree_data and length == stree_length:
                raise FrontDuplicateError()
            elif data >= stree_data and length <= stree_length:
                is_dominated = True
        
        if len(to_remove) > 0 or not is_dominated:
            for idx in sorted(to_remove, reverse=True):
                curr_sol = curr_front[idx]
                self.symbfreq.remove(curr_sol[0])
                curr_front.pop(idx)
                curr_sol_eval = self.eval_map[id(curr_sol[0])]
                del self.eval_map[id(curr_sol[0])]
                if tracked_counter < self.popsize:
                    self.track(curr_sol[0], (curr_sol[1], curr_sol[2]), curr_sol_eval, frontidx + 1, tracked_counter)
            
            self.front[frontidx].append((stree, stree_data, stree_length))
            self.eval_map[id(stree)] = evaluation
            self.symbfreq.add(stree)

        elif is_dominated and tracked_counter < self.popsize:
            self.track(stree, datalength_eval, evaluation, frontidx + 1, tracked_counter)

    def get_front(self, frontidx:int=0):
        _, symbdist = self.compute_symbdist(frontidx)
        crowdist = self.compute_crowdist(frontidx)

        def fcmp(a, b):
            nonlocal symbdist, crowdist
            #symbdist_diff = symbdist[id(a[0])] - symbdist[id(b[0])]
            #if symbdist_diff == 0:
            return crowdist[id(b[0])] - crowdist[id(a[0])]
            #return symbdist_diff
        
        return sorted(self.front[frontidx], key=cmp_to_key(fcmp))

    def get_population(self, max_size:int):
        population = []
        for frontidx in range(len(self.front)):
            population += [stree for stree, _, _ in self.get_front(frontidx)]
            if len(population) >= max_size:
                return population[:max_size]
        return population
    
    """
    def compute_symbdist(self) -> dict:
        symbset  = {}
        symbdist = {}

        for stree, _, _ in self.front:
            optCollector = SyntaxTreeIneqOperatorCollector()
            stree.accept(optCollector)
            symbset[id(stree)] = optCollector.opts
            symbdist[id(stree)] = 0
        
        if len(self.front) == 1:
            symbdist[id(self.front[0][0])] = 1.0
            return symbset, symbdist 
        
        for i, (stree_a, _, _) in enumerate(self.front):
            for stree_b, _, _ in self.front[i+1:]:

                symbset_a = symbset[id(stree_a)]
                symbset_b = symbset[id(stree_b)]
                dist = len( symbset_a.intersection(symbset_b) ) / len( symbset_a.union(symbset_b) )
                
                symbdist[id(stree_a)] += dist
                symbdist[id(stree_b)] += dist

        for stree_id in symbdist.keys():
            symbdist[stree_id] /= len(self.front) - 1
        
        return symbset, symbdist
    """

    def compute_symbdist(self, frontidx:int=0) -> dict:
        symbset  = {}
        symbdist = {}
        optset_count = {}

        for stree, _, _ in self.front[frontidx]:
            optCollector = SyntaxTreeIneqOperatorCollector()
            stree.accept(optCollector)

            opts = frozenset(optCollector.opts)
            symbset[id(stree)] = opts
            if opts not in optset_count:
                optset_count[opts] = 0
            optset_count[opts] += 1
        
        for stree_id, opts in symbset.items():
            symbdist[stree_id] = optset_count[opts]
        
        return symbset, symbdist
    
    def compute_crowdist(self, frontidx:int=0) -> dict:
        crowdist = {}

        if len(self.front[frontidx]) == 0:
            return crowdist

        def fcmp(a, b):
            return a[1] - b[1]
        sorted_front = sorted(self.front[frontidx], key=cmp_to_key(fcmp))

        # data stats.
        all_data = np.array([data for _, data, length in sorted_front])
        data_min = all_data.min()
        data_max = all_data.max()
        data_range = data_max - data_min

        # length stats.
        all_lengths = np.array([length for _, data, length in sorted_front])
        length_min = all_lengths.min()
        length_max = all_lengths.max()
        length_range = length_max - length_min

        crowdist[id(sorted_front[0 ][0])] = np.infty
        crowdist[id(sorted_front[-1][0])] = np.infty
        for i, (stree, data, length) in enumerate(sorted_front[1:len(sorted_front)-1]):
            i += 1

            data_next = sorted_front[i+1][1]
            data_prev = sorted_front[i-1][1]
            data_next = (data_next - data_min) / data_range  # normalize data (R2 score) objective.
            data_prev = (data_prev - data_min) / data_range

            length_next = sorted_front[i+1][2]
            length_prev = sorted_front[i-1][2]
            length_next = (length_next - length_min) / length_range  # normalize length objective.
            length_prev = (length_prev - length_min) / length_range

            crowdist[id(stree)] = data_next - data_prev + length_next - length_prev
        
        return crowdist

    
    def compute_extend_of_convergence(self, data_lu:tuple[float,float], length_lu:tuple[float,float], frontidx:int=0) -> float:
        data_min, data_max     = data_lu
        length_min, length_max = length_lu
        data_range             = data_max - data_min
        length_range           = length_max - length_min

        def fcmp(a, b):
            return a[1] - b[1]
        front = sorted(self.front[frontidx], key=cmp_to_key(fcmp))
        if len(front) == 0: return 1.0

        data_front   = [((data-data_min)/data_range)       for _, data, _   in front]  # get data   front normalized.
        length_front = [((length-length_min)/length_range) for _, _, length in front]  # get length front normalized.
        
        extent = 0.0
        for i in range(len(front)):
            if i == 0:
                extent += data_front[i] * length_front[i]
            else:
                extent += (data_front[i] - data_front[i-1]) * length_front[i]
        extent += 1.0 - data_front[-1]

        return 1.0 - extent
    
    def is_empty(self) -> bool:
        for f in self.front.values():
            if len(f) > 0: return False
        return True
    
    def plot(self, data_lu:tuple[float,float], length_lu:tuple[float,float], frontids:list):
        
        def fcmp(a, b):
            return a[1] - b[1]
        
        plt.figure(figsize=(5,5))
        
        for frontidx in range(len(self.front)) if len(frontids) == 0 else frontids:
            
            front = sorted(self.front[frontidx], key=cmp_to_key(fcmp))
            if len(front) == 0: continue

            # plot front lines.
            if len(front) > 1:
                x_line = [front[0][1]]
                y_line = [front[0][2]]

                for _, data, length in front[1:]:
                    x_line += [x_line[-1], data]
                    y_line += [length, length]
                
                plt.plot(x_line, y_line, linestyle='dashed', c='k')
            
            plt.plot([data_lu[0],   front[0][1] ], [front[0][2],  front[0][2] ], linestyle='dashed', c='k')
            plt.plot([front[-1][1], front[-1][1]], [front[-1][2], length_lu[1]], linestyle='dashed', c='k')
            
            # plot front dots.
            symbset, symbdist = self.compute_symbdist(frontidx)

            x = [data for _, data, _ in front]
            y = [length for _, _, length in front]
            c = [symbdist[id(stree)] for stree, _, _ in front]

            plt.scatter(x, y, c='k') #, c=c, cmap='coolwarm')
        
        xl, xu = data_lu
        yl, yu = length_lu
        xy_margin = 0.05
        x_margin = (xu-xl) * xy_margin
        y_margin = (yu-yl) * xy_margin
        xl -= x_margin
        xu += x_margin
        yl -= y_margin
        yu += y_margin

        #plt.colorbar()
        plt.xlabel('R²-score')
        plt.ylabel('Length')
        plt.xlim((xl, xu))
        plt.ylim((yl, yu))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        #plt.gca().yaxis.tick_right()
        #plt.gca().yaxis.set_label_position("right")
        plt.title('Pareto Front(s)')
        plt.show()


class MultiHeadFrontTracker:
    def __init__(self, popsize:int, max_fronts=np.inf, min_fea_ratio:float=0.0):
        self.popsize = popsize
        self.max_fronts = max_fronts
        self.min_fea_ratio = min_fea_ratio
        self.heads = {}
    
    def track(self, stree, datalength_eval:tuple, evaluation):
        fea_ratio = evaluation.fea_ratio if type(evaluation) is LayeredEvaluation else 1.0
        #if fea_ratio < self.min_fea_ratio: return
        if fea_ratio < 1.0: fea_ratio = 0.0
        if fea_ratio not in self.heads:
            self.heads[fea_ratio] = DataLengthFrontTracker(self.popsize, self.max_fronts)
        self.heads[fea_ratio].track(stree, datalength_eval, evaluation)
    
    def get_front(self, fea_ratio:float=1.0, frontidx:int=0):
        if fea_ratio not in self.heads:
            return []
        return self.heads[fea_ratio].get_front(frontidx)
    
    def get_populations(self) -> list[list]:
        populations = []
        current_popsize = 0
        remaining_popsize = self.popsize

        for h, f in sorted(self.heads.items(), reverse=True):
            
            f.popsize = remaining_popsize
            current_population = f.get_population(remaining_popsize)
            current_popsize += len(current_population)
            remaining_popsize = self.popsize - current_popsize

            if remaining_popsize <= 0:
                for __h, __f in self.heads.items():
                    if __h < h: f.popsize = 0
                populations.append( (h, current_population[:len(current_population)+remaining_popsize]) )
                return populations
            
            populations.append((h, current_population))
        
        return populations

    def get_head(self, headidx:int=0):
        fea_ratio = sorted(self.heads.keys(), reverse=True)[headidx]
        return self.heads[fea_ratio], fea_ratio
