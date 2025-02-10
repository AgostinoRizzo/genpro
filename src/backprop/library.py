import numpy as np
from scipy.spatial import KDTree
#import nmslib
from multiprocessing import Process, Lock, Condition
import random

from symbols.syntax_tree import SyntaxTree
from symbols.const import ConstantSyntaxTree
from symbols.visitor import SyntaxTreeIneqOperatorCollector
from gp import creator, selector
from backprop.utils import is_symmetric


def compute_distance(p1:np.array, p2:np.array):
    if np.isnan(p1).any() or np.isnan(p2).any():
        return np.inf
    return np.linalg.norm(p1 - p2)

def compute_mse(p1:np.array, p2:np.array):
    return ((p1 - p2) ** 2).sum() / p1.size


class SyntaxTreeCloneProvider:
    def __init__(self, stree_index:list):
        self.stree_index = stree_index
    
    def get_stree(self, idx:int):
        return self.stree_index[idx].clone()
    
    def draw_stree(self):
        return random.choice(self.stree_index).clone()


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
    def query(self, point, k:int=1, max_dist=np.inf):
        pass

class ExactKnnIndex(KnnIndex):
    def __init__(self, points):
        self.index = KDTree(points)
    
    def query(self, point, k:int=1, max_dist=np.inf):
        return self.index.query(point, k=k, p=2, distance_upper_bound=max_dist)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
class ReducedExactKnnIndex(KnnIndex):
    def __init__(self, points):
        self.pca = PCA(n_components=min(20, *points.shape))
        points_proj = self.pca.fit_transform(points)
        self.index = KDTree(points_proj)
    
    def query(self, point, k:int=1, max_dist=np.inf):
        point_proj = self.pca.transform(np.array([point]))[0]
        return self.index.query(point_proj, k=k, p=2, distance_upper_bound=max_dist)

"""class ApproxKnnIndex(KnnIndex):
    def __init__(self, points):
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.addDataPointBatch(points)
        self.index.createIndex({'post': 2}, print_progress=True)
    
    def query(self, point, k:int=1, max_dist=np.inf):
        idx, dist = self.index.knnQuery(point, k=k)
        if k == 1: return dist[0], idx[0]
        return dist, idx"""

class ApproxKnnIndex(KnnIndex):
    def __init__(self, points, nanchors:int=8):
        self.anchors = [np.random.uniform(points.min(), points.max(), points.shape[1])]
        anchored_points = []
        for i in range(points.shape[0]):
            anchored_points.append( [compute_distance(points[i], a) for a in self.anchors] )
        self.index = KDTree(np.asmatrix(anchored_points))
    
    def query(self, point, k:int=1, max_dist=np.inf):
        anchored_point = np.asarray( [compute_distance(point, a) for a in self.anchors] )
        return self.index.query(anchored_point, k=k, p=2, distance_upper_bound=max_dist)


class LibraryError(RuntimeError):
    pass

class LibraryLookupError(LibraryError):
    pass

class LibraryKnowledgeLookupError(LibraryError):
    pass


class Library:
    def __init__(self, size:int, max_depth:int, max_length:int, data, know, solutionCreator, mesh=None, symm:bool=None, ext_strees:list=None):
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

        if ext_strees is None:
            extra_trees = size

            X_extra = []
            for x_val in [0.0, 1.0, -1.0]:
                x = [x_val] * data.nvars
                if np.greater_equal(x, data.xl).all() and np.less_equal(x, data.xu).all():
                    X_extra.append(x)
            X_extra = np.array(X_extra)

            def_st_idx = np.full(data.X.shape[0], True, dtype=bool)
            def_st_extra_idx = np.full(X_extra.shape[0], True, dtype=bool)
            for i in range(def_st_idx.size):
                if know.is_undef_at(data.X[i]): def_st_idx[i] = False
            for i in range(def_st_extra_idx.size):
                if know.is_undef_at(X_extra[i]): def_st_extra_idx[i] = False

            all_semantics = {}
            std_min = data.y.std() * 0.001  # at least 1% of the variance of the dataset.
            max_trials = 2
            ntrials = 0

            while extra_trees > 0 and ntrials < max_trials:
                strees = solutionCreator.create_population(extra_trees, max_depth, max_length, create_consts=False)
                extra_trees = 0
                ntrials += 1
                
                for t in strees:
                    t = t.simplify()  # TODO: ensure executed once.
                    st = t(data.X)
                    st_extra = t.at(X_extra)
                    st_key = tuple(st.tolist())
                    
                    if symm is not None and symm != is_symmetric(t[(mesh.X, ())], mesh.symm_Y_Ids):
                        extra_trees += 1
                        continue

                    if not np.isfinite(st**2)[def_st_idx].all():
                        extra_trees += 1
                        continue
                    
                    if (~np.isfinite(st_extra))[def_st_extra_idx].any() or \
                    (~np.isfinite(st))[def_st_idx].any() or (st == st[0]).all():
                        extra_trees += 1
                        continue
                    
                    if st.std() < std_min:
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
        else:
            for t in ext_strees:
                st = t(data.X)
                self.stree_index.append(t)
                self.lib_data.append(st)

        self.lib_data = np.stack(self.lib_data)
        self.sem_index = ExactKnnIndex(self.lib_data)
        #self.sem_index = ApproxKnnIndex(self.lib_data)

        self.stree_provider = SyntaxTreeCloneProvider(self.stree_index)

        from backprop.pareto_front import SymbolicFrequencies
        self.symbfreq = SymbolicFrequencies()
    
    def query(self, sem, max_dist=np.inf) -> SyntaxTree:
        const_fit = sem.mean()
        const_fit_d = np.inf
        if np.isnan(const_fit): const_fit = None
        else:
            const_fit_d = compute_distance(const_fit, sem)
            max_dist = min(max_dist, const_fit_d)
        
        d, idx = self.sem_index.query(sem, max_dist=max_dist)
        
        if not np.isfinite(d) and not np.isfinite(const_fit_d):
            raise LibraryLookupError()
        if np.isfinite(const_fit_d) and const_fit_d <= d:
            return ConstantSyntaxTree(const_fit)

        stree = self.stree_provider.get_stree(idx)
        #self.symbfreq.add(stree)
        return stree
    
    def multiquery(self, sem, k=4, max_dist=np.inf) -> list[tuple[np.array, SyntaxTree]]:
        d, idx = self.sem_index.query(sem, k=k, max_dist=max_dist)
        if d[0] == np.infty: raise LibraryLookupError()  # nearest firts.
        return [ (self.lib_data[__idx], self.stree_provider.get_stree(__idx)) for __idx in idx ]
    
    def draw(self) -> SyntaxTree:
        return self.stree_provider.draw_stree()

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
                
                optCollectorA = SyntaxTreeIneqOperatorCollector()
                stree_a.accept(optCollectorA)
                
                optCollectorB = SyntaxTreeIneqOperatorCollector()
                stree_b.accept(optCollectorB)

                if optCollectorA.opts == optCollectorB.opts: continue

                min_d = d
                min_i = i
                min_idx = idx

        print(f"{self.stree_index[min_i]} => {self.stree_index[min_idx]} [{min_d}]")
        return self.stree_index[min_i], self.stree_index[min_idx], min_d


class ConstrainedLibrary(Library):
    def __init__(self, size:int, max_depth:int, max_length:int, data, know, mesh, derivs:list[tuple[int]], solutionCreator):
        super().__init__(size, max_depth, max_length, data, know, solutionCreator)
        self.max_depth = max_depth
        
        K_none = (None, None)

        self.clibs        = {d: {} for d in range(max_depth+1)}
        self.clibs_idxmap = {d: {} for d in range(max_depth+1)}
        self.clibs_negmap = {d: {} for d in range(max_depth+1)}
        for d in range(max_depth+1):
            self.clibs       [d][K_none] = []
            self.clibs_idxmap[d][K_none] = []
            self.clibs_negmap[d][K_none] = []

        X_extra = np.array([
            [ 0.0] * data.nvars,
            [ 1.0] * data.nvars,
            [-1.0] * data.nvars,
        ])

        def_k_t_image_idx = np.full(mesh.X.shape[0], True, dtype=bool)
        def_k_t_extra_idx = np.full(X_extra.shape[0], True, dtype=bool)
        for i in range(def_k_t_image_idx.size):
            if know.is_undef_at(mesh.X[i]): def_k_t_image_idx[i] = False
        for i in range(def_k_t_extra_idx.size):
            if know.is_undef_at(X_extra[i]): def_k_t_extra_idx[i] = False

        for i, t in enumerate(self.stree_index):
            d_t = t.get_max_depth()
            k_t_image = np.sign(t[(mesh.X, ())])
            k_t = np.concatenate( [np.sign(t[(mesh.X, d)]) for d in sorted(derivs)] )  # important to sort them!
            t_extra = t.at(X_extra)
            t.clear_output()

            noroot = (k_t_image[def_k_t_image_idx] != 0.0).all() and \
                     (t_extra[def_k_t_extra_idx] != 0.0).all() and \
                     not np.isnan(t_extra)[def_k_t_extra_idx].any()

            for sign in [1.0]:  # TODO: [1.0, -1.0]:
                
                K_t = ((k_t * sign).tobytes(), noroot)  # TODO: better to unify with constraints.BackpropConstraints.
                if K_t not in self.clibs_idxmap[d_t]:
                    self.clibs_idxmap[d_t][K_t] = []
                    self.clibs_negmap[d_t][K_t] = []
                self.clibs_idxmap[d_t][K_t].append(i)
                self.clibs_negmap[d_t][K_t].append(sign < 0.0)

                # default bucket.
                self.clibs_idxmap[d_t][K_none].append(i)
                self.clibs_negmap[d_t][K_none].append(sign < 0.0)
        
        for d_t in self.clibs_idxmap.keys():
            for K_t in self.clibs_idxmap[d_t].keys():
                self.clibs[d_t][K_t] = ExactKnnIndex(self.lib_data[self.clibs_idxmap[d_t][K_t]])
            self.clibs[d_t][K_none] = ExactKnnIndex(self.lib_data[self.clibs_idxmap[d_t][(None, None)]])  # default library.

    def cquery(self, y, C, max_dist=np.inf) -> SyntaxTree:
        
        matching_libs = self.__get_matching_libs(C)  # 10s
        if len(matching_libs) == 0:
            return None
        
        best_d = None
        best_i_mlib = None
        best_local_idx = None
        for i_mlib, (mlib, mlib_idxmap, mlib_negmap) in enumerate(matching_libs):  # 6s

            d, idx = mlib.query(y, max_dist=max_dist)  # 5s
            if d == np.infty: continue
            if best_d is None or d < best_d:
                best_d = d
                best_i_mlib = i_mlib
                best_local_idx = idx

        if best_d is None:
            return None
        
        mlib, mlib_idxmap, mlib_negmap = matching_libs[best_i_mlib]
        global_idx = best_local_idx if mlib_idxmap is None else mlib_idxmap[best_local_idx]
        if mlib_negmap is not None and mlib_negmap[best_local_idx]:
            return self.stree_provider.get_stree(global_idx).scale(-1.0)
        return self.stree_provider.get_stree(global_idx)
    
    def __get_matching_libs(self, C) -> list:
        matching_libs = []

        for d in range(min(C.get_max_depth(), self.max_depth)+1):
        
            # partially constrained.
            if C.are_partial():
                for K_lib in self.clibs[d].keys():
                    if K_lib == (None, None): continue
                    if C.match_key(K_lib):  # arg K_lib must not contain NaN values apart from undef points (given in input)! (see match_key).  # 10s
                        matching_libs.append((self.clibs[d][K_lib], self.clibs_idxmap[d][K_lib], self.clibs_negmap[d][K_lib]))

            # not constrained.
            elif C.are_none():
                K_none = (None, None)
                matching_libs.append((self.clibs[d][K_none], self.clibs_idxmap[d][K_none], self.clibs_negmap[d][K_none]))

            # totally constrained.
            else:
                K = C.get_key()
                if K in self.clibs[d]:
                    matching_libs.append((self.clibs[d][K], self.clibs_idxmap[d][K], self.clibs_negmap[d][K]))
        
        return matching_libs

    """
    def cquery_brute(self, y, K, max_dist=np.inf, w:np.array=None, S_train=None) -> SyntaxTree:
        
        if K not in self.clibs:
            return None
        
        data = self.clibs[K].index.data
        
        yy = S_train.y
        ddata = (-0.05*S_train.X[:,0]) / data
        q = yy - ddata
        idx = None
        if w is None:
            idx = np.argmin( np.sqrt( np.sum(q ** 2, axis=1) ) )
        else:
            idx = np.argmin( np.sqrt( np.sum(w * (q ** 2), axis=1) ) )

        global_idx = self.clibs_idxmap[K][idx]
        if self.clibs_negmap[K][idx]:
            return self.stree_provider.get_stree(global_idx).scale(-1.0)
        return self.stree_provider.get_stree(global_idx)
    
    def cquery_stoch(self, y, K, max_dist=np.inf) -> SyntaxTree:
        
        if K not in self.clibs:
            return None
        
        idx = random.randrange(self.clibs[K].index.data.shape[0])
        global_idx = self.clibs_idxmap[K][idx]
        if self.clibs_negmap[K][idx]:
            return self.stree_provider.get_stree(global_idx).scale(-1.0)
        return self.stree_provider.get_stree(global_idx)
    """


class IterativeConstrainedLibrary(Library):
    def __init__(self, size:int, max_depth:int, max_length:int, data, know, mesh, derivs:list[tuple[int]], solutionCreator):
        super().__init__(size, max_depth, max_length, data, know, solutionCreator)
        self.C_map = [None] * len(self.stree_index)
        self.nqueries = 0
        self.nfaults = 0

        X_extra = np.array([
            [ 0.0] * data.nvars,
            [ 1.0] * data.nvars,
            [-1.0] * data.nvars,
        ])

        def_k_t_image_idx = np.full(mesh.X.shape[0], True, dtype=bool)
        def_k_t_extra_idx = np.full(X_extra.shape[0], True, dtype=bool)
        for i in range(def_k_t_image_idx.size):
            if know.is_undef_at(mesh.X[i]): def_k_t_image_idx[i] = False
        for i in range(def_k_t_extra_idx.size):
            if know.is_undef_at(X_extra[i]): def_k_t_extra_idx[i] = False

        for i, t in enumerate(self.stree_index):
            d_t = t.get_max_depth()
            k_t_image = np.sign(t[(mesh.X, ())])
            k_t = np.concatenate( [np.sign(t[(mesh.X, d)]) for d in sorted(derivs)] )  # important to sort them!
            t_extra = t.at(X_extra)
            t.clear_output()

            noroot = (k_t_image[def_k_t_image_idx] != 0.0).all() and \
                     (t_extra[def_k_t_extra_idx] != 0.0).all() and \
                     not np.isnan(t_extra)[def_k_t_extra_idx].any()
                
            C_t = (d_t, k_t.tobytes(), noroot)
            self.C_map[i] = C_t

    def cquery(self, y, C, max_dist=np.inf) -> SyntaxTree:
        self.nqueries += 1
        lib = self.sem_index
        offset_i = 0

        for k in [32, 128, 512]:
            d, idx = lib.query(y, k=k, max_dist=max_dist)
            
            for i in range(offset_i, k):
                if d[i] == np.inf: return None
                idx_i = idx[i]
                d_i, k_i, noroot_i = self.C_map[idx_i]

                if d_i <= C.get_max_depth() and C.match_key((k_i, noroot_i)):
                    return self.stree_provider.get_stree(idx_i)
            
            offset_i = k
        
        # radius fault.
        self.nfaults += 1
        return None
    
    def get_radius_faults(self):
        return self.nfaults / self.nqueries


class HierarchicalConstrainedLibrary(Library):
    def __init__(self, size:int, max_depth:int, max_length:int, data, know, mesh, derivs:list[tuple[int]], solutionCreator, symm:bool=None, ext_strees:list=None):
        super().__init__(size, max_depth, max_length, data, know, solutionCreator, mesh, symm, ext_strees)
        self.max_depth = max_depth
        
        K_none = (None, None)

        self.clib = {}
        self.clib_idxmap = {}
        self.C_map = [None] * len(self.stree_index)

        X_extra = np.array([
            [ 0.0] * data.nvars,
            [ 1.0] * data.nvars,
            [-1.0] * data.nvars,
        ])

        def_k_t_image_idx = np.full(mesh.X.shape[0], True, dtype=bool)
        def_k_t_extra_idx = np.full(X_extra.shape[0], True, dtype=bool)
        for i in range(def_k_t_image_idx.size):
            if know.is_undef_at(mesh.X[i]): def_k_t_image_idx[i] = False
        for i in range(def_k_t_extra_idx.size):
            if know.is_undef_at(X_extra[i]): def_k_t_extra_idx[i] = False

        for i, t in enumerate(self.stree_index):
            d_t = t.get_max_depth()
            k_t_image = np.sign(t[(mesh.X, ())])
            k_t = np.concatenate( [np.sign(t[(mesh.X, d)]) for d in sorted(derivs)] )
            t_extra = t.at(X_extra)
            t.clear_output()

            k_t_image[~mesh.sign_defspace[()]] = np.nan

            noroot = (k_t_image[def_k_t_image_idx] != 0.0).all() and \
                     (t_extra[def_k_t_extra_idx] != 0.0).all() and \
                     not np.isnan(t_extra)[def_k_t_extra_idx].any()

            noroot_options = [True] if noroot else [False, True]
            for depth in range(d_t, self.max_depth + 1):
                for noroot_t in noroot_options:
                    for K_t in [(depth,), (depth, noroot_t), (depth, noroot_t, k_t_image.tobytes())]:
                        if K_t not in self.clib_idxmap: self.clib_idxmap[K_t] = []
                        self.clib_idxmap[K_t].append(i)
            
            C_t = (d_t, noroot, k_t.tobytes())
            self.C_map[i] = C_t
        
        for K in self.clib_idxmap.keys():
            self.clib[K] = ExactKnnIndex(self.lib_data[self.clib_idxmap[K]])

    def cquery(self, y, C, max_dist=np.inf, check_constfit:bool=True) -> SyntaxTree:
        # constant fit (when compliant w.r.t. C).
        const_fit = None
        if check_constfit:
            const_fit = y.mean()
            if np.isnan(const_fit): const_fit = None
            const_fit_dist = np.inf
            if const_fit is not None and C.check_const(const_fit):
                const_fit_dist = compute_distance(const_fit, y)
                max_dist = min(max_dist, const_fit_dist)
        
        max_depth = min(C.get_max_depth(), self.max_depth)

        if C.are_none():
            try:
                K = (max_depth,)
                stree, dist = self.__local_query(self.clib[K], self.clib_idxmap[K], y, max_dist)
                if const_fit is not None and const_fit_dist <= max_dist and const_fit_dist <= dist:
                    return ConstantSyntaxTree(const_fit)
                if stree is None:
                    raise LibraryLookupError()
                return stree
            except KeyError:
                raise LibraryLookupError()

        k_bytes, noroot = C.get_key_image()

        for K in [(max_depth, noroot, k_bytes), (max_depth, noroot), (max_depth,)]:
        
            try:
            
                check_image = len(K) < 3
                stree, dist = self.__local_cquery(self.clib[K], self.clib_idxmap[K], y, C, max_dist, check_image)
                if const_fit is not None and const_fit_dist <= max_dist and const_fit_dist <= dist:
                    return ConstantSyntaxTree(const_fit)
                if stree is None:
                    raise LibraryLookupError()
                return stree

            except KeyError:
                continue
        
        raise LibraryLookupError()  # never here.
    
    def __local_query(self, lib, clib_idxmap, y, max_dist) -> tuple[SyntaxTree,float]:
        dist, local_idx = lib.query(y, max_dist=max_dist)
        if dist == np.infty: return None, dist
        global_idx = clib_idxmap[local_idx]
        return self.stree_provider.get_stree(global_idx), dist
    
    def __local_cquery(self, lib, clib_idxmap, y, C, max_dist, check_image:bool, k:int=128) -> tuple[SyntaxTree,float]:
        offset_i = 0

        while True:
            dist, local_idx = lib.query(y, k=k, max_dist=max_dist)
            
            for i in range(offset_i, k):
                dist_i = dist[i]
                if dist_i == np.inf: return None, dist_i
                
                global_idx_i = clib_idxmap[local_idx[i]]
                depth_i, noroot_i, k_i = self.C_map[global_idx_i]

                if C.match_key((k_i, noroot_i), check_image):
                    return self.stree_provider.get_stree(global_idx_i), dist_i
            
            offset_i = k
            k += k

        
class DynamicConstrainedLibrary:
    def __init__(self, population:list, eval_map:dict, selector:selector.Selector, data, mesh):
        parents = selector.select(population, eval_map, len(population))
        self.strees = [random.choice(p.cache.nodes) for p in parents]
        self.sem    = [t(data.X) for t in self.strees]
        self.lib  = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
        self.clib = {0:{}, 1:{}, 2:{}, 3:{}, 4:{}, 5:{}}

        X_extra = np.array([
            [ 0.0] * data.nvars,
            [ 1.0] * data.nvars,
            [-1.0] * data.nvars,
        ])

        for i, t in enumerate(self.strees):
            k_t = np.sign(t[(mesh.X, ())])
            t.clear_output()
            t_extra = t(X_extra)
            t.clear_output()

            noroot = (k_t != 0.0).all() and (t_extra != 0.0).all() and not np.isnan(t_extra).any()

            K_t = (k_t.tobytes(), noroot)

            max_depth = t.get_max_depth()
            for d in range(max_depth, 6):
                if K_t not in self.clib[d]:
                    self.clib[d][K_t] = []
                self.clib[d][K_t].append(i)
                self.lib[d].append(i)
        
    def query(self, max_depth:int, y=None):
        #idx = random.choice(self.lib[max_depth])
        #return self.strees[idx].clone()
        return self.__query(y, self.lib[max_depth])
    
    def cquery(self, K, max_depth:int, y=None):
        k_t, noroot = K
        
        k_t_nan = np.isnan(k_t)
        if k_t_nan.all():
            return self.query(max_depth)
        if k_t_nan.any():
            return self.query(max_depth)  # TODO: linear/brute search.
        
        K = (k_t.tobytes(), noroot)
        if K not in self.clib[max_depth]:
            return self.query(max_depth)
        #idx = random.choice(self.clib[max_depth][K])
        #return self.strees[idx].clone()
        return self.__query(y, self.clib[max_depth][K])
    
    def __query(self, y, indices:list):
        if y is None or not np.isfinite(y).all() or True:
            return self.strees[random.choice(indices)].clone()
        
        best_i = None
        best_d = None

        for i in indices:
            y0 = self.sem[i]
            d = compute_distance(y, y0)
            if best_i is None or d < best_d:
                best_i = i
                best_d = d
        
        return self.strees[best_i].clone()


def get_semiasymm_strees(symm_strees:list, mesh) -> list:
    semiasymm_strees = []

    for stree in symm_strees:
        stree.set_parent()
        
        for n in stree.cache.nodes:
            if type(n) is ConstantSyntaxTree: continue
            if is_symmetric(n[(mesh.X, ())], mesh.symm_Y_Ids): continue

            semiasymm_strees.append(n)
    
    return semiasymm_strees
