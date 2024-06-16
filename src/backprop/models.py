import numpy as np
import math


class Model:
    def __init__(self):
        self.derivs: dict[tuple[int,...],callable] = {}
    
    def get_deriv(self, vars:tuple) -> callable: return self.derivs[vars]
    def get_coeffs(self) -> np.array: return None
    def set_coeffs(self, coeffs:np.array): pass
    def set_coeff(self, idx, c): pass
    def __call__(self, x): pass


class Poly1d(Model):
    def __init__(self, deg:int):  # coeffs in decreasing powers.
        assert deg >= 0
        super().__init__()
        self.P = np.poly1d(np.ones(deg + 1))
        self.set_coeffs(np.zeros(self.P.c.size))
    
    def get_coeffs(self) -> np.array:
        return self.P.c
    
    def set_coeffs(self, coeffs:np.array):
        assert coeffs.size == self.P.c.size
        self.P.c[:] = coeffs
        self.derivs[(   )] = self.P
        self.derivs[(0, )] = np.polyder(self.P, m=1)
        self.derivs[(0,0)] = np.polyder(self.P, m=2)
    
    def set_coeff(self, idx, c):
        assert type(idx) == int
        self.P.c[idx] = c

    def __call__(self, x):
        return self.P(x)


class Polynd(Model):
    CIDX_MAP = {}
    
    def __init__(self, nvars:int, deg:int):
        assert nvars > 0 and deg >= 0
        super().__init__()
        self.nvars = nvars
        self.deg = deg
        self.C = np.zeros((deg+1,) * nvars)
        self.cidx = Polynd.__get_cidx(nvars, deg)
    
    def get_coeffs(self) -> np.array:
        return self.C[self.cidx]
    
    def set_coeffs(self, coeffs:np.array):
        assert coeffs.size == self.cidx[0].size
        self.C[self.cidx] = coeffs
        self.derivs[()] = self
        
        # compute all 1st derivatives.
        for var_idx in range(self.nvars):
            C = np.polynomial.polynomial.polyder(self.C, m=1, axis=var_idx)
            self.derivs[(var_idx,)] = lambda x, C=C, nvars=self.nvars : Polynd.__evaluate(C, x, nvars)
        
        # compute all 2nd derivatives.
        for var1_idx in range(self.nvars):
            C1 = np.polynomial.polynomial.polyder(self.C, m=1, axis=var1_idx)
            for var2_idx in range(self.nvars):
                C2 = np.polynomial.polynomial.polyder(C1, m=1, axis=var2_idx)
                self.derivs[(var1_idx,var2_idx)] = lambda x, C=C2, nvars=self.nvars : Polynd.__evaluate(C, x, nvars)
    
    def set_coeff(self, idx, c):
        assert type(idx) == tuple and len(idx) == self.nvars and sum(idx) <= self.deg
        self.C[idx] = c
    
    def __call__(self, x):
        return Polynd.__evaluate(self.C, x, self.nvars)
    
    @staticmethod
    def __get_cidx(nvars:int, deg:int) -> np.array:
        if (nvars, deg) in Polynd.CIDX_MAP:
            return Polynd.CIDX_MAP[(nvars, deg)]
        cidx = (np.empty( math.comb(nvars+deg, deg), dtype=int ),) * nvars
        i = 0
        size = deg + 1
        for idx in np.ndindex( (size,) * nvars ):
            if sum(idx) < size:
                for v_idx in range(nvars):
                    cidx[v_idx][i] = idx[v_idx]
                i += 1
        Polynd.CIDX_MAP[(nvars, deg)] = cidx
        return cidx
    
    @staticmethod
    def __evaluate(C:np.ndarray, x, nvars:int):
        assert x.ndim <= 2 and x.size > 0 and x.shape[0] == nvars

        it = iter(x)
        x0 = next(it)
        __C = np.polynomial.polynomial.polyval(x0, C)
        for xi in it:
            __C = np.polynomial.polynomial.polyval(xi, __C, tensor=False)

        return __C  # (numpy.array of) numpy.float*


class ModelFactory:
    @staticmethod
    def create_poly(deg:int, nvars:int=1) -> Model:
        assert deg >= 0 and nvars >= 1
        if nvars == 1: return Poly1d(deg)
        return Polynd(nvars, deg)
