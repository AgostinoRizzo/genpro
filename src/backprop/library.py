import numpy as np
from scipy.spatial import KDTree
from backprop import backprop, gp


class Library:

    def __init__(self, size:int, max_depth:int, data):
        self.data = data

        solutionCreator = gp.RandomSolutionCreator(nvars=data.nvars)
        strees = solutionCreator.create_population(size, max_depth, noconsts=True)
        
        self.stree_index = []
        self.lib_data = []

        for t in strees:
            st = t(data.X)
            self.stree_index.append(t)
            self.lib_data.append(st)
        self.lib_data = np.stack(self.lib_data)

        self.sem_index = KDTree(self.lib_data)
    
    def query(self, sem) -> backprop.SyntaxTree:
        const_fit = sem.mean()
        const_fit_d = np.linalg.norm(const_fit - sem)
        
        d, idx = self.sem_index.query(sem)
        
        if d == np.infty: return None
        if const_fit_d <= d: return backprop.ConstantSyntaxTree(const_fit)

        return self.stree_index[idx].clone()
