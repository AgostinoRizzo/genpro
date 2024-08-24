import numpy as np
from scipy.spatial import KDTree
from backprop import backprop, gp


class Library:

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
                st_key = tuple(st.tolist())

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
        self.sem_index = KDTree(self.lib_data)
    
    def query(self, sem) -> backprop.SyntaxTree:
        const_fit = sem.mean()
        const_fit_d = np.linalg.norm(const_fit - sem)
        
        d, idx = self.sem_index.query(sem)
        
        if d == np.infty: return None
        #if const_fit_d <= d: return backprop.ConstantSyntaxTree(const_fit)

        return self.stree_index[idx].clone()
    
