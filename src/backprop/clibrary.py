import numpy as np
import itertools
import random

from backprop.library import Library, StochasticSemanticIndex, ExactKnnIndex, LibraryLookupError, LibraryKnowledgeLookupError, compute_distance
from symbols.syntax_tree import SyntaxTree
from symbols.const import ConstantSyntaxTree


class ConstrainedLibrary(Library):
    def __init__(self,
            size:int, max_depth:int, max_length:int, data, know, mesh, derivs:list[tuple[int]],
            solutionCreator, symm:bool=None, ext_strees:list=None, stochastic:bool=False):
        super().__init__(size, max_depth, max_length, data, know, solutionCreator, mesh, symm, ext_strees)
        self.max_depth = max_depth
        self.stochastic = stochastic

        #print(f"---> ACTUAL LIB SIZE: {len(self.stree_index)}")

        self.clib = {}
        self.clib_idxmap = {}

        X_extra = np.array([
            [ 0.0] * data.nvars,
            [ 1.0] * data.nvars,
            [-1.0] * data.nvars,
        ])

        def_s_t_image_idx = np.full(mesh.X.shape[0], True, dtype=bool)
        def_s_t_extra_idx = np.full(X_extra.shape[0], True, dtype=bool)
        for i in range(def_s_t_image_idx.size):
            if know.is_undef_at(mesh.X[i]): def_s_t_image_idx[i] = False
        for i in range(def_s_t_extra_idx.size):
            if know.is_undef_at(X_extra[i]): def_s_t_extra_idx[i] = False

        for i, t in enumerate(self.stree_index):
            d_t = t.get_max_depth()
            t_extra = t.at(X_extra)
            t.clear_output()

            S_t = {}
            zero_derivs = []
            for deriv in derivs:
                s = np.sign(t[(mesh.X, deriv)])
                s[~mesh.sign_defspace[deriv]] = np.nan
                S_t[deriv] = s
                if (s[mesh.sign_defspace[deriv]] == 0).all():
                    zero_derivs.append(deriv)
            
            s_t_image = S_t[()]

            noroot = (s_t_image[def_s_t_image_idx] != 0.0).all() and \
                     (t_extra[def_s_t_extra_idx] != 0.0).all() and \
                     not np.isnan(t_extra)[def_s_t_extra_idx].any()
            
            SS_t = [ np.concatenate( [S_t[deriv] for deriv in sorted(S_t.keys())] ) ]
            mesh_size = mesh.X.shape[0]
            if len(zero_derivs) > 0:
                for signs in itertools.product([-1.0, 1.0], repeat=len(zero_derivs)):
                    s_t = []
                    for deriv in sorted(derivs):
                        if deriv in zero_derivs: s_t.append([signs[zero_derivs.index(deriv)]] * mesh_size)
                        else: s_t.append(S_t[deriv])
                    SS_t.append( np.concatenate(s_t) )


            noroot_options = [True, False] if noroot else [False]
            for depth in range(d_t, self.max_depth + 1):
                for noroot_t in noroot_options:
                    for s_t in SS_t:
                        K_t = (depth, noroot_t, s_t.tobytes())
                        if K_t not in self.clib_idxmap: self.clib_idxmap[K_t] = []
                        self.clib_idxmap[K_t].append(i)
        
        for K in self.clib_idxmap.keys():
            self.clib[K] = self._create_index(self.lib_data[self.clib_idxmap[K]])
        
        self.mesh = mesh
    
    def _create_index(self, data):
        if self.stochastic: return StochasticSemanticIndex(data)
        return ExactKnnIndex(data)

    def cquery(self, y, C, max_dist=np.inf, check_constfit:bool=True) -> SyntaxTree:
        # constant fit (when compliant w.r.t. C).
        const_fit = None
        if check_constfit and (not self.stochastic or random.random() < 0.05):
            const_fit = y.mean()
            if np.isnan(const_fit): const_fit = None
            const_fit_dist = np.inf
            if const_fit is not None and C.check_const(const_fit):
                const_fit_dist = compute_distance(const_fit, y)
                max_dist = min(max_dist, const_fit_dist)
        
        max_depth = min(C.get_max_depth(), self.max_depth)

        S_bytes, noroot = C.get_key()
        K = (max_depth, noroot, S_bytes)
        
        #relaxed = self.query(y, max_dist=max_dist)
        #img = np.sign(relaxed[(self.mesh.X, ())])
        #d1 = np.sign(relaxed[(self.mesh.X, (1,))])
        #are_eq = np.array_equal(img, C.origin_pconstrs[()]) and np.array_equal(d1, C.origin_pconstrs[(1,)])
        
        #return relaxed

        try:
            
            dist, local_idx = self.clib[K].query(y, k=1, max_dist=max_dist)
            #dist, global_idx = self.sem_index.query(y, k=1, max_dist=max_dist)

            if const_fit is not None and const_fit_dist <= max_dist and const_fit_dist <= dist:
                return ConstantSyntaxTree(const_fit)
            
            if dist == np.inf:
                raise LibraryLookupError()
            
            global_idx = self.clib_idxmap[K][local_idx]
            constr = self.stree_provider.get_stree(global_idx)
            return constr

        except KeyError:
            #return relaxed
            raise LibraryKnowledgeLookupError()
            #return self.query(y, max_dist=max_dist)
