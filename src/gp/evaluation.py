class Evaluation:
    def __init__(self, minimize:bool=True):
        self.minimize = minimize
    def better_than(self, other) -> bool: return False


class RealEvaluation(Evaluation):
    def __init__(self, value, minimize:bool=True):
        super().__init__(minimize)
        self.value = value
    def better_than(self, other) -> bool:
        if np.isnan(other.value):
            if np.isnan(self.value): return False
            return True
        if self.minimize: return self.value < other.value
        return self.value > other.value
    def get_value(self):
        return self.value
    def __str__(self) -> str:
        return f"{self.value}"


class LayeredEvaluation(Evaluation):
    def __init__(self, n, nv, r2):
        self.fea_ratio = 1.0 - (nv / n)
        self.r2 = r2
    
    def better_than(self, other) -> bool:        
        if self.r2 == 0.0: return False

        if self.fea_ratio > other.fea_ratio: return True
        if self.fea_ratio < other.fea_ratio: return False

        return self.r2 > other.r2
    
    def get_value(self):
        return self.r2
    
    def __str__(self) -> str:
        return f"fea: {self.fea_ratio}\n" + \
               f"r2:  {self.r2}"
