import numpy as np


class Evaluation:
    def __init__(self, minimize:bool=True):
        self.minimize = minimize
    def better_than(self, other) -> bool: return False
    def get_quality(self): return self


class RealEvaluation(Evaluation):
    def __init__(self, value, minimize:bool=True, name:str='', isfeasible:bool=True):
        super().__init__(minimize)
        self.value = value
        self.isfeasible = isfeasible
        self.name = name
    
    def better_than(self, other) -> bool:
        if np.isnan(other.value):
            if np.isnan(self.value): return False
            return True
        if self.minimize: return self.value < other.value
        return self.value > other.value
    
    def get_value(self):
        return self.value
    
    def __str__(self) -> str:
        if len(self.name) > 0: return f"{self.name}: {self.value}"
        return f"{self.value}"


class LayeredEvaluation(Evaluation):
    def __init__(self, n, nv, data_eval, stree, know_pressure:float=1.0):
        self.fea_ratio = (1.0 - (nv / n)) if data_eval.isfeasible else 0.0
        self.data_eval = data_eval
        self.stree = stree
        self.know_pressure = know_pressure
    
    def better_than(self, other) -> bool:
        if self.know_pressure > 0:  # TODO: add probability.
            if self.fea_ratio > other.fea_ratio: return True
            if self.fea_ratio < other.fea_ratio: return False

        if self.data_eval.better_than(other.data_eval): return True
        if other.data_eval.better_than(self.data_eval): return False

        return self.stree.get_nnodes() < other.stree.get_nnodes()
    
    def get_value(self):
        return self.data_eval.value
    
    def get_quality(self):
        return self.data_eval
    
    def __str__(self) -> str:
        return f"fea: {self.fea_ratio}\n" + \
               f"{self.data_eval}"
