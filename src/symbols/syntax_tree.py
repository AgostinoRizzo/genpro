import numpy as np


class PullError(RuntimeError):
    pass

class PullViolation(RuntimeError):
    pass


class SyntaxTreeInfo:
    X_data = None

    @staticmethod
    def set_problem(data):
        SyntaxTreeInfo.X_data = data.X

    def __init__(self, stree):
        self.stree = stree
        
        self.__nnodes = None
        self.__nodes = None
        self.__backprop_nodes = None
        self.__y = None

    @property
    def nnodes(self):
        if self.__nnodes is None:
            self.__nnodes = self.stree.get_nnodes()
        return self.__nnodes
        
    @property
    def nodes(self):
        if self.__nodes is None:
            self.__nodes = []
            self.stree.get_nodes(self.__nodes)

            self.__nnodes = len(self.__nodes)
        return self.__nodes

    @property
    def backprop_nodes(self):
        if self.__backprop_nodes is None:
            self.__backprop_nodes = []
            for n in self.nodes:
                if SyntaxTree.is_invertible_path(n):
                    self.__backprop_nodes.append(n)
        return self.__backprop_nodes

    @property
    def y(self):
        if self.__y is None:
            self.__y = self.stree(SyntaxTreeInfo.X_data)
        return self.__y
    
    def clear(self):
        self.__nnodes = None
        self.__nodes = None
        self.__backprop_nodes = None
        self.__y = None


class SyntaxTree:
    CACHE_MAX_DEPTH = 0
    CACHE_NNODES = 1

    def __init__(self):
        self.parent = None
        self.output = None  # TODO: change name to y_data.
        self.y_know = {}
        self.output_stash = None
        self.y_know_stash = None
        #self.cache = [None] * 2
        self.parents = None

        self.sat = True
        self.sat_y = True
        self.match_r2 = 1.0
        self.best_match_r2 = 1.0

        self.cache = SyntaxTreeInfo(self)
    
    def __call__(self, x): pass
    def __getitem__(self, x_d): pass

    def at(self, x): pass

    def invalidate_output(self):
        self.output = None
        self.y_know = {}  # important: create new.
        if self.parent is not None:
            self.parent.invalidate_output()
    
    def clear_output(self):
        self.output = None
        self.y_know = {}  # important: create new.
    
    def stash_output(self):
        self.output_stash = self.output
        self.y_know_stash = self.y_know
        self.output = None
        self.y_know = {}  # important: create new.
        if self.parent is not None:
            self.parent.stash_output()
    
    def backup_output(self):
        self.output = self.output_stash
        self.y_know = self.y_know_stash
        if self.parent is not None:
            self.parent.backup_output()
    
    def copy_output_from(self, other):
        self.output = other.output
        self.y_know = other.y_know

    def has_parent(self) -> bool:
        return self.parent is not None

    def clone(self): return None
    def set_parent(self, parent=None): self.parent = parent
    def validate(self) -> bool: return True
    def simplify(self): return self
    def __str__(self) -> str: return ''
    def __eq__(self, other) -> bool: return False
    def diff(self, varidx:int=0): return None
    def is_const(self) -> bool: return False
    def is_const_wrt(self, varidx) -> bool: return False
    
    def pull_output(self, target_output:np.array, child=None, flatten:bool=False) -> np.array:
        if self.parent is None:
            return utils.flatten(target_output) if flatten else target_output
        return self.parent.pull_output(target_output, self, flatten)
    
    def pull_know(self, k_target:np.array, noroot_target:bool=False, child=None, track:dict={}) -> tuple[np.array,bool]:
        if self.parent is None:
            return k_target, noroot_target
        return self.parent.pull_know(k_target, noroot_target, self, track)
    
    def pull_know_deriv(self, image_track:dict, derividx:int, k_target:np.array, child=None) -> np.array:
        if self.parent is None:
            return k_target
        return self.parent.pull_know_deriv(image_track, derividx, k_target, self)
    
    def get_coeffs(self, coeffs:list):
        pass
    
    def set_coeffs(self, coeffs:list, start:int=0):
        pass

    def get_unknown_stree(self, unknown_stree_label:str): return None
    def set_unknown_model(self, model_label:str, model, coeffs_mask:list[float]=None, constrs:dict=None): pass
    def set_all_unknown_models(self, model): pass
    def count_unknown_model(self, model_label:str) -> int: return 0
    def accept(self, visitor): pass
    def to_sympy(self, dps:int=None): pass
    def get_max_depth(self) -> int: return 0
    def get_nnodes(self) -> int: return 1
    def get_nodes(self, nodes:list):
        nodes.append(self)
    def match(self, trunk) -> bool: return False
    def is_linear(self) -> bool: return False
    def is_invertible(self) -> bool: return False
    #TODO: def scale(self, l): return BinaryOperatorSyntaxTree('*', ConstantSyntaxTree(l), self)
    def is_scalable(self, l) -> bool: return False

    def get_depth(self) -> int:
        depth = 0
        p = self.parent
        while p is not None:
            depth += 1
            p = p.parent
        return depth
    
    def is_subtree(self, other) -> bool:
        if id(self) == id(other): return True
        if self.parent is None: return False
        return self.parent.is_subtree(other)

    def clear_cache(self):
        #for i in range(len(self.cache)): self.cache[i] = None
        pass
    
    @staticmethod
    def is_invertible_path(node) -> bool:
        p = node.parent
        while p is not None:
            if not p.is_invertible(): return False
            p = p.parent
        return True
    
    @staticmethod
    def diff(stree, deriv:tuple[int]):
        stree_deriv = stree
        for varidx in deriv:
            stree_deriv = stree_deriv.diff(varidx).simplify()
        return stree_deriv
    
    @staticmethod
    def diff_all(stree, derivs:list[tuple[int]], include_zeroth:bool=True) -> dict:
        derivs_map = {(): stree.simplify()}
        for deriv in sorted(derivs):
            if len(deriv) == 0: continue
            derivs_map[deriv] = derivs_map[deriv[:-1]].diff(deriv[-1]).simplify()
        if not include_zeroth:
            del derivs_map[()]
        return derivs_map
