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

class FUEvaluation(Evaluation):
    def __init__(self, know_mse, know_nv, know_n, know_ls, know_sat, data_r2, nnodes):
        self.know_mse  = know_mse
        self.know_nv   = know_nv
        self.know_n    = know_n
        self.know_ls   = know_ls
        self.know_sat  = know_sat
        self.data_r2   = data_r2
        self.nnodes    = nnodes
        self.crowdist  = 0

        self.fea_ratio = 1. - (know_nv / know_n)
        self.fea_bucket = int( self.fea_ratio * 10. )
        self.feasible  = know_nv == 0
    
        self.data_nnodes = data_r2 / nnodes

        self.genidx = 0
    
    def better_than(self, other) -> bool:        
        if self.data_r2 == 0.0: return False

        if self.fea_ratio > other.fea_ratio: return True
        if self.fea_ratio < other.fea_ratio: return False

        return self.data_r2 > other.data_r2
    
    def get_value(self):
        return self.data_r2
    
    def __str__(self) -> str:
        return f"know_mse:  {self.know_mse }\n" + \
               f"know_nv:   {self.know_nv  }\n" + \
               f"know_n:    {self.know_n   }\n" + \
               f"know_ls:   {self.know_ls  }\n" + \
               f"fea_ratio: {self.fea_ratio}\n" + \
               f"data_r2:   {self.data_r2  }"