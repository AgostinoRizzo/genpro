import numpy as np
import sympy
import math


class Model:
    def __init__(self):
        self.derivs: dict[tuple[int,...],_] = {}
    
    def get_deriv(self, vars:tuple): return self.derivs[vars]
    def get_coeffs(self) -> np.array: return None
    def get_coeffs_size(self) -> int: return 0
    def set_coeffs(self, coeffs): pass
    def set_coeff(self, idx, c): pass
    def __call__(self, x): pass
    def is_poly(self) -> bool: return False
    def get_degree(self) -> int: return 0
    def to_sympy(self, dps:int=None): return None
    def __str__(self) -> str: return str(self.to_sympy())


class Poly(Model):
    def simplify_from_qp(self, constrs:dict):  # constrs:dict[qp.Constraints]
        pass
    def is_poly(self) -> bool:
        return True


class Poly1d(Poly):
    def __init__(self, deg:int, P:np.poly1d=None, compute_derivs:bool=True):  # coeffs in decreasing powers.
        assert deg >= 0
        super().__init__()
        self.deg = deg
        if P is None: self.P = np.poly1d(np.ones(deg + 1))
        else: self.P = P
        self.compute_derivs = compute_derivs
        self.set_coeffs(0. if P is None else P.c)
    
    def get_coeffs(self) -> np.array:
        return self.P.c
    
    def get_coeffs_size(self) -> int:
        return self.P.c.size
    
    def set_coeffs(self, coeffs):
        assert np.isscalar(coeffs) or coeffs.size == self.P.c.size
        self.P.c[:] = coeffs
        self.derivs[(   )] = self
        if self.compute_derivs:
            self.derivs[(0, )] = Poly1d(max(self.deg-1, 0), P=np.polyder(self.P, m=1), compute_derivs=False)
            self.derivs[(0,0)] = Poly1d(max(self.deg-2, 0), P=np.polyder(self.P, m=2), compute_derivs=False)
    
    def set_coeff(self, idx, c):
        assert type(idx) == int
        self.P.c[idx] = c

    def __call__(self, x):
        return self.P(x)
    
    def get_degree(self) -> int:
        return self.deg
    
    def to_sympy(self, dps:int=None):
        if self.P.c.size == 0: return sympy.Integer(0)
        x = sympy.Symbol('x')
        P_sympy = sympy.Float(self.P.c[0], dps=dps) * (x**(self.P.c.size-1))
        i = 1
        for power in range(self.P.c.size-2, -1, -1):
            P_sympy += sympy.Float(self.P.c[i], dps=dps) * (x**power)
            i += 1
        return P_sympy
    
    def simplify_from_qp(self, constrs:dict):
        canBeZeroCoeffs = [True] * self.P.c.size
        for derivdeg, constrs in constrs.items():
            if constrs.noroot: canBeZeroCoeffs[-derivdeg-1] = False
        
        for i in range(self.P.c.size):
            if abs(self.P.c[i]) < 1e-8 and canBeZeroCoeffs[i]:  # TODO: fix epsilon.
                self.P.c[i] = 0


class Polynd(Poly):
    CIDX_MAP = {}
    
    def __init__(self, nvars:int, deg:int, C:np.ndarray=None, compute_derivs:bool=True):
        assert nvars > 0 and deg >= 0
        super().__init__()
        self.nvars = nvars
        self.deg = deg
        if C is None: self.C = np.zeros((deg+1,) * nvars)
        else:
            assert C.size <= (deg+1) ** nvars
            self.C = C
            if compute_derivs: self.set_coeffs(self.get_coeffs())
            else: self.derivs[()] = self
        self.compute_derivs = compute_derivs
        self.cidx = Polynd.__get_cidx(nvars, deg)
    
    def get_coeffs(self) -> np.array:
        return self.C[self.cidx]
    
    def get_coeffs_size(self) -> int:
        return self.cidx[0].size
    
    def set_coeffs(self, coeffs):
        self.C[self.cidx] = coeffs
        self.derivs[()] = self
        
        if self.compute_derivs:
            # compute all 1st derivatives.
            for var_idx in range(self.nvars):
                C = np.polynomial.polynomial.polyder(self.C, m=1, axis=var_idx)
                self.derivs[(var_idx,)] = Polynd(self.nvars, max(C.shape)-1, C=C, compute_derivs=False)
            
            # compute all 2nd derivatives.
            for var1_idx in range(self.nvars):
                C1 = np.polynomial.polynomial.polyder(self.C, m=1, axis=var1_idx)
                for var2_idx in range(self.nvars):
                    C2 = np.polynomial.polynomial.polyder(C1, m=1, axis=var2_idx)
                    self.derivs[(var1_idx,var2_idx)] = Polynd(self.nvars, max(C2.shape)-1, C=C2, compute_derivs=False)
    
    def set_coeff(self, idx, c):
        assert type(idx) == tuple and len(idx) == self.nvars and sum(idx) <= self.deg
        self.C[idx] = c
    
    def __call__(self, x):
        return Polynd.__evaluate(self.C, x, self.nvars)
    
    def get_degree(self) -> int:
        return self.deg
    
    def to_sympy(self, dps:int=None):
        cidx_size = self.cidx[0].size
        if cidx_size == 0: return sympy.Integer(0)
        x = [sympy.Symbol(f"x{i}") for i in range(self.nvars)]

        P_sympy = sympy.Float(0., dps=dps)
        for i in range(cidx_size):
            powers = tuple([self.cidx[j][i] for j in range(self.nvars)])
            if self.C[powers] == 0.: continue
            
            P_sympy_sub = sympy.Float(self.C[powers], dps=dps)
            for i_var in range(self.nvars):
                if powers[i_var] != 0: P_sympy_sub *= x[i_var]**powers[i_var]
            P_sympy += P_sympy_sub
        
        return P_sympy
    
    def simplify_from_qp(self, constrs:dict):
        raise RuntimeError('No implementation (yet).')
    
    @staticmethod
    def __get_cidx(nvars:int, deg:int) -> np.array:
        if (nvars, deg) in Polynd.CIDX_MAP:
            return Polynd.CIDX_MAP[(nvars, deg)]
        cidx_size = math.comb(nvars+deg, deg)
        cidx = tuple( [np.empty( cidx_size, dtype=int ) for _ in range(nvars)] )

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
    def create_poly(deg:int, nvars:int=1) -> Poly:
        assert deg >= 0 and nvars >= 1
        if nvars == 1: return Poly1d(deg)
        return Polynd(nvars, deg)
