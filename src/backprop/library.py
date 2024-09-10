import numpy as np
from scipy.spatial import KDTree
import nmslib
from multiprocessing import Process, Lock, Condition
from backprop import backprop, gp


class SyntaxTreeCloneProvider:
    def __init__(self, stree_index:list):
        self.stree_index = stree_index
    
    def get_stree(self, idx:int):
        return self.stree_index[idx]


class MultiprocSyntaxTreeCloneProvider:
    def __init__(self, stree_index:list):
        self.stree_index = stree_index
        self.stree_clone = [None] * len(stree_index)
        
        self.to_clone_buff = []
        self.lock = Lock()
        self.to_clone_cond = Condition(lock=self.lock)
        self.cloned_cond = Condition(lock=self.lock)
        self.pr = Process(target=self.__run, daemon=True)
        self.pr.start()
    
    def get_stree(self, idx:int):
        with self.lock:
            if self.stree_clone[idx] is None:
                self.to_clone_buff.append(idx)
                self.to_clone_cond.notify()
            while self.stree_clone[idx] is None:
                self.cloned_cond.wait()
            
            stree = self.stree_clone[idx]
            self.stree_clone[idx] = None
            
            return stree
    
    def __run(self):
        with self.lock:
            #for i in range(len(self.stree_clone)):
            #    self.stree_clone[i] = self.stree_index[i].clone()

            while True:
                while len(self.to_clone_buff) == 0:
                    self.to_clone_cond.wait()
                
                for idx in self.to_clone_buff:
                    self.stree_clone[idx] = self.stree_index[idx].clone()
                
                self.to_clone_buff.clear()
                self.cloned_cond.notify()


class KnnIndex:
    def query(self, point, k:int=1):
        pass

class ExactKnnIndex(KnnIndex):
    def __init__(self, points):
        self.index = KDTree(points)
    
    def query(self, point, k:int=1):
        return self.index.query(point, k=k)

class ApproxKnnIndex(KnnIndex):
    def __init__(self, points):
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.addDataPointBatch(points)
        self.index.createIndex({'post': 2}, print_progress=True)
    
    def query(self, point, k:int=1):
        idx, dist = self.index.knnQuery(point, k=k)
        if k == 1: return dist[0], idx[0]
        return dist, idx



class Library:
    DIST_EPSILON = 1e-1 #1e-8
    UNIQUENESS_MAX_DECIMALS = 1 #8

    def __init__(self, size:int, max_depth:int, data):
        """
        A total of 'size' random trees are generated:
            * algebraic simplification of trees
            * semantics containing NaN or infty values are ignored (extra test on 0.0, 1.0 and -1.0)
            * constant semantics are ignored
            * only unique semantics, the smaller (#nodes) is kept
        """

        self.data = data
        self.stree_index = []
        self.lib_data = []

        solutionCreator = gp.RandomSolutionCreator(nvars=data.nvars)
        extra_trees = size
        X_extra = np.array([
            [ 0.0] * data.nvars,
            [ 1.0] * data.nvars,
            [-1.0] * data.nvars,
        ])

        all_semantics = {}

        while extra_trees > 0:
            strees = solutionCreator.create_population(extra_trees, max_depth, noconsts=True)
            extra_trees = 0
            
            for t in strees:
                t = t.simplify()  # TODO: ensure executed once.
                st = t(data.X)
                st_extra = t(X_extra)
                st_key = tuple(st.round(Library.UNIQUENESS_MAX_DECIMALS).tolist())

                if np.isnan(st_extra).any() or np.isnan(st_extra).any() or \
                   np.isnan(st).any() or np.isinf(st).any() or (st == st[0]).all():
                    extra_trees += 1
                    continue
                
                if st_key in all_semantics:
                    old_stree_i = all_semantics[st_key]
                    old_tree = self.stree_index[old_stree_i]
                    
                    extra_trees += 1

                    if old_tree.get_nnodes() <= t.get_nnodes():
                        continue
                    self.stree_index[old_stree_i] = t
                    self.lib_data[old_stree_i] = st

                else:
                    all_semantics[st_key] = len(self.stree_index)
                    self.stree_index.append(t)
                    self.lib_data.append(st)

        self.lib_data = np.stack(self.lib_data)
        self.sem_index = ExactKnnIndex(self.lib_data)
        #self.sem_index = ApproxKnnIndex(self.lib_data)

        self.stree_provider = SyntaxTreeCloneProvider(self.stree_index)

        from backprop.pareto_front import SymbolicFrequencies
        self.symbfreq = SymbolicFrequencies()
    
    def query(self, sem) -> backprop.SyntaxTree:
        #const_fit = sem.mean()
        #const_fit_d = np.linalg.norm(const_fit - sem)
        
        d, idx = self.sem_index.query(sem)
        
        if d == np.infty: return None
        #if const_fit_d <= d: return backprop.ConstantSyntaxTree(const_fit)

        stree = self.stree_provider.get_stree(idx)
        #self.symbfreq.add(stree)
        return stree
    
    def multiquery(self, sem, k=4) -> list[tuple[np.array, backprop.SyntaxTree]]:
        d, idx = self.sem_index.query(sem, k=k)
        if d[0] == np.infty: return None  # nearest firts.
        return [ (self.lib_data[__idx], self.stree_provider.get_stree(__idx)) for __idx in idx ]
    
    def find_best_similarity(self):
        min_d = None
        min_i = None
        min_idx = None

        for i in range(self.lib_data.shape[0]):
            d, idx = self.sem_index.query(self.lib_data[i], k=2)
            d = d[1]
            idx = idx[1]

            if min_d is None or d < min_d:
                stree_a = self.stree_index[i]
                stree_b = self.stree_index[idx]
                
                optCollectorA = backprop.SyntaxTreeIneqOperatorCollector()
                stree_a.accept(optCollectorA)
                
                optCollectorB = backprop.SyntaxTreeIneqOperatorCollector()
                stree_b.accept(optCollectorB)

                if optCollectorA.opts == optCollectorB.opts: continue

                min_d = d
                min_i = i
                min_idx = idx

        print(f"{self.stree_index[min_i]} => {self.stree_index[min_idx]} [{min_d}]")
        return self.stree_index[min_i], self.stree_index[min_idx], min_d

