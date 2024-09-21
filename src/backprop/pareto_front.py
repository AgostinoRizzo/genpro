from functools import cmp_to_key
import numpy as np
import random
import matplotlib.pyplot as plt

from symbols.visitor import SyntaxTreeIneqOperatorCollector


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


class DataLengthFrontTracker:

    def __init__(self, nfronts:int=8):
        self.front = {f: [] for f in range(nfronts)}
        self.nfronts = nfronts
        self.symbfreq = SymbolicFrequencies()

    def track(self, stree, evaluation, frontidx:int=0):
        if frontidx >= self.nfronts: return

        stree_data = evaluation.data_r2
        stree_length = evaluation.nnodes

        if stree_data < 0.01: return

        to_remove = []
        is_dominated = False

        for idx, (_, data, length) in enumerate(self.front[frontidx]):
            if stree_data >= data and stree_length <= length:
                to_remove.append(idx)
            
            if data >= stree_data and length <= stree_length:
                is_dominated = True
        
        if len(to_remove) > 0 or not is_dominated:
            for idx in sorted(to_remove, reverse=True):
                self.symbfreq.remove(self.front[frontidx][idx][0])
                self.front[frontidx].pop(idx)  # TODO: add removed into next front.
            
            self.front[frontidx].append((stree, stree_data, stree_length))
            self.symbfreq.add(stree)

        elif is_dominated:
            self.track(stree, evaluation, frontidx + 1)

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
    
    def get_population(self):
        population = []
        for frontidx in range(self.nfronts):
            population += [stree for stree, _, _ in self.get_front(frontidx)]
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

    def is_empty(self) -> bool:
        for f in self.front.values():
            if len(f) > 0: return False
        return True
    
    def plot(self, frontidx:int=-1):
        
        def fcmp(a, b):
            return a[1] - b[1]
        
        for frontidx in range(self.nfronts) if frontidx < 0 else range(frontidx, frontidx + 1):
            
            front = sorted(self.front[frontidx], key=cmp_to_key(fcmp))

            # plot front lines.
            if len(front) > 1:
                x_line = [front[0][1]]
                y_line = [front[0][2]]

                for _, data, length in front[1:]:
                    x_line += [x_line[-1], data]
                    y_line += [length, length]
                
                plt.plot(x_line, y_line, linestyle='dashed')
            
            # plot front dots.
            symbset, symbdist = self.compute_symbdist(frontidx)

            x = [data for _, data, _ in front]
            y = [length for _, _, length in front]
            c = [symbdist[id(stree)] for stree, _, _ in front]

            plt.scatter(x, y, c=c, cmap='coolwarm')
        
        plt.colorbar()
        plt.xlabel('R2')
        plt.ylabel('#Nodes')
        plt.title('Data-Length Front')
        plt.show()


class MultiHeadFrontTracker:
    def __init__(self):
        self.front_tracker_a = DataLengthFrontTracker()
        self.front_tracker_b = DataLengthFrontTracker()
    
    def track(self, stree, evaluation, frontidx:int=0):
        simil_a = self.front_tracker_a.symbfreq.compute_similarity(stree)
        simil_b = self.front_tracker_b.symbfreq.compute_similarity(stree)
        
        if simil_a == 0 and simil_b == 0:
            if not self.front_tracker_a.is_empty() and self.front_tracker_b.is_empty():
                self.front_tracker_b.track(stree, evaluation, frontidx)
            elif random.choice([0, 1]) == 0:
                self.front_tracker_a.track(stree, evaluation, frontidx)
            else:
                self.front_tracker_b.track(stree, evaluation, frontidx)
            return

        if simil_a >= simil_b or True:
        #if 'exp' not in SymbolicFrequencies.get_symbset(stree):
            self.front_tracker_a.track(stree, evaluation, frontidx)
        else:
            self.front_tracker_b.track(stree, evaluation, frontidx)
    
    def get_front(self, frontidx:int=0):
        crowdist = self.front_tracker_a.compute_crowdist(frontidx)
        crowdist.update(self.front_tracker_b.compute_crowdist(frontidx))

        def fcmp(a, b):
            nonlocal crowdist
            return crowdist[id(b[0])] - crowdist[id(a[0])]
        
        return sorted(self.front_tracker_a.front[frontidx] + self.front_tracker_b.front[frontidx], key=cmp_to_key(fcmp))
    
    def get_population(self):
        population = []
        for frontidx in range(self.front_tracker_a.nfronts):
            population += [stree for stree, _, _ in self.get_front(frontidx)]
        return population