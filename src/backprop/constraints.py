import numpy as np


class Constraints:
    def __init__(self, max_depth:int):
        self.max_depth = max_depth
    
    def get_max_depth(self) -> int:
        return self.max_depth

class BackpropConstraints(Constraints):
    def __init__(self, max_depth:int, pconstrs:np.array, noroot:bool):
        super().__init__(max_depth)
        self.pconstrs = pconstrs
        self.noroot = noroot

        # precompute meta-info.
        self.key = (pconstrs.tobytes(), noroot)
        pconstrs_nan = np.isnan(pconstrs)
        self.partial = not noroot or pconstrs_nan.any()
        self.none = not noroot and pconstrs_nan.all()
        self.pconstrs_mask = ~np.isnan(pconstrs)
    
    def get_key(self) -> tuple:
        return self.key
    
    def are_partial(self) -> bool:  # partially constrained.
        return self.partial
    
    def are_none(self) -> bool:  # unconstrained.
        return self.none
    
    def match_key(self, K_other) -> bool:
        """
        it is assumed K_other does not contain NaN values!
        """
        pconstrs_other, noroot_other = K_other
        pconstrs_other = np.frombuffer(pconstrs_other)
        
        if self.noroot and not noroot_other:
            return False 

        return np.array_equal(self.pconstrs[self.pconstrs_mask], pconstrs_other[self.pconstrs_mask])