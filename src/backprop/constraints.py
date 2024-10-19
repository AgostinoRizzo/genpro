import numpy as np


class Constraints:
    def __init__(self, max_depth:int, max_length:int):
        self.max_depth = max_depth
        self.max_length = max_length
    
    def get_max_depth(self) -> int:
        return self.max_depth
    
    def get_max_length(self) -> int:
        return self.max_length


class BackpropConstraints(Constraints):
    def __init__(self, max_depth:int, max_length:int, pconstrs:dict[tuple,np.array], noroot:bool):
        super().__init__(max_depth, max_length)
        self.origin_pconstrs = pconstrs
        self.pconstrs_image = pconstrs[()]
        self.pconstrs = np.concatenate( [pconstrs[deriv] for deriv in sorted(pconstrs.keys())] )  # important order with sorted!
        self.noroot = noroot

        # precompute meta-info.
        self.key = (self.pconstrs.tobytes(), noroot)
        self.key_image = (self.pconstrs_image.tobytes(), noroot)
        pconstrs_nan = np.isnan(self.pconstrs)
        self.partial = not noroot or pconstrs_nan.any()
        self.none = not noroot and pconstrs_nan.all()
        self.pconstrs_mask = ~pconstrs_nan
        self.pconstrs_image_size = self.pconstrs_image.size
    
    def get_key(self) -> tuple:
        return self.key
    
    def get_key_image(self) -> tuple:
        return self.key_image
    
    def are_partial(self) -> bool:  # partially constrained.
        return self.partial
    
    def are_none(self) -> bool:  # unconstrained.
        return self.none
    
    def match_key(self, K_other, check_image:bool=True) -> bool:
        """
        it is assumed K_other does not contain NaN values apart from undef points (given in input).
        """
        pconstrs_other, noroot_other = K_other
        pconstrs_other = np.frombuffer(pconstrs_other)
        
        if self.noroot and not noroot_other:
            return False 

        offset = 0 if check_image else self.pconstrs_image_size
        return np.array_equal(
            self.pconstrs[offset:][self.pconstrs_mask[offset:]],
            pconstrs_other[offset:][self.pconstrs_mask[offset:]], equal_nan=True)
    
    def project(self, y):
        image_pconstr = self.origin_pconstrs[()]
        if   (image_pconstr > 0.0).all(): y[y < 0.0] = 0.0
        elif (image_pconstr < 0.0).all(): y[y > 0.0] = 0.0
    
    def __str__(self) -> str:
        out = ''
        if self.noroot:
            out += "noroot\n"
        for deriv in sorted(self.origin_pconstrs.keys()):
            if len(deriv) > 0:
                out += f"{str(deriv)}: "
            out += f"{self.origin_pconstrs[deriv]}\n"
        return out

    def format(self, X_mesh, funcname:str='f') -> str:
        out = ''
        
        if self.noroot:
            out += f" {funcname}(x) ≠ 0\n"
        
        def __equal_val__(v1, v2):
            if np.isnan(v1) and np.isnan(v2): return True
            return v1 == v2
        
        for deriv in sorted(self.origin_pconstrs.keys()):
            f = f" {funcname}"
            derivdeg = len(deriv)
            assert derivdeg <= 1
            if derivdeg == 1:
                f = f"∂{funcname}"
            
            curr_val = self.origin_pconstrs[deriv][0]
            l = 0
            for i, val in enumerate(self.origin_pconstrs[deriv].tolist() + [np.nan]):
                if curr_val is not None and not __equal_val__(val, curr_val):
                    u = i - 1
                    
                    constr_opt = '='
                    if curr_val > 0: constr_opt = '≥'
                    elif curr_val < 0: constr_opt = '≤'
                    else:
                        l = i
                        curr_val = val
                        continue
                    
                    xl = "{:.2f}".format(X_mesh[l])
                    xu = "{:.2f}".format(X_mesh[u])
                    out += f"{f}(x) {constr_opt} 0  [{xl},{xu}]\n"
                    
                    l = i
                    curr_val = val
        return out