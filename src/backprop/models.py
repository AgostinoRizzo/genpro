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
    def simplify_from_qp(self, constrs_map:dict):  # constrs_map:dict[tuple[int],qp.Constraints]
        pass
    def is_poly(self) -> bool:
        return True
    def as_virtual(self, X, polydeg:int):
        return None


class Poly1d(Poly):
    POWERS_MAP:dict[int,np.array] = {}

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

    def __call__(self, X):
        if np.isscalar(X): return self.P(X)
        assert X.ndim <= 2
        return self.P( X if X.ndim == 1 else X[:,0] )  # TODO: check X matrix as Fortran style more efficient (*).
    
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
    
    def simplify_from_qp(self, constrs_map:dict):
        canBeZeroCoeffs = [True] * self.P.c.size
        for deriv, constrs in constrs_map.items():
            derivdeg = len(deriv)
            if constrs.noroot: canBeZeroCoeffs[-derivdeg-1] = False
        
        for i in range(self.P.c.size):
            if abs(self.P.c[i]) < 1e-8 and canBeZeroCoeffs[i]:  # TODO: fix epsilon.
                self.P.c[i] = 0
    
    def as_virtual(self, X, polydeg:int):
        assert X.ndim <= 2 and self.deg <= polydeg
        ncoeffs = polydeg + 1
        x = X if X.ndim <= 1 else X[:,0]
        V = self.P.c * ( np.tile(x, (self.deg+1,1)).T ** Poly1d.get_powers(self.deg) )
        n_zero_coeffs = ncoeffs - V.shape[1]
        if n_zero_coeffs == 0: return V
        return np.hstack(( V, np.zeros((V.shape[0], n_zero_coeffs)) ))
    
    @staticmethod
    def get_powers(deg:int):
        if deg in Poly1d.POWERS_MAP:
            return Poly1d.POWERS_MAP[deg]
        powers = np.array( [p for p in range(deg, -1, -1)] )
        Poly1d.POWERS_MAP[deg] = powers
        return powers


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
        self.cidx, self.CIDX = Polynd.__get_cidx(nvars, deg)
    
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
    
    def __call__(self, X):
        assert X.ndim == 2 and X.shape[1] == self.nvars
        return Polynd.__evaluate(self.C, X.T, self.nvars)
    
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
    
    def simplify_from_qp(self, constrs_map:dict):
        raise RuntimeError('No implementation (yet).')
    
    def as_virtual(self, X, polydeg:int):
        ncoeffs = polydeg + 1
        assert X.ndim == 2 and X.shape[1] == self.nvars and self.CIDX.shape[0] <= ncoeffs
        # i = # constrol points (mesh size)
        # j = # coeffs
        n_zero_cols = ncoeffs - self.CIDX.shape[0]
        V_rows = X.shape[0]
        V_cols = n_zero_cols + self.CIDX.shape[0]
        
        V = np.empty((V_rows, V_cols))
        V[:,0:n_zero_cols] = 0

        for i in range(V_rows):  # TODO: make it more efficient totally using numpy.
            for j in range(n_zero_cols, V_cols):
                V[i,j] = np.prod( X[i,:] ^ self.CIDX[:,j] )
        
        return V
    
    @staticmethod
    def __get_cidx(nvars:int, deg:int) -> tuple[tuple[np.array],np.ndarray]:
        if (nvars, deg) in Polynd.CIDX_MAP:
            return Polynd.CIDX_MAP[(nvars, deg)]
        cidx_size = math.comb(nvars+deg, deg)
        cidx = tuple( [np.empty( cidx_size, dtype=int ) for _ in range(nvars)] )
        CIDX = np.empty((nvars, cidx_size), dtype=int)  # TODO: as Fortran is more efficient for as_virtual?!

        i = 0
        size = deg + 1
        for idx in np.ndindex( (size,) * nvars ):
            if sum(idx) < size:
                for v_idx in range(nvars):
                    cidx[v_idx][i] = idx[v_idx]
                    CIDX[v_idx][i] = idx[v_idx]
                i += 1
        Polynd.CIDX_MAP[(nvars, deg)] = (cidx, CIDX)
        return cidx, CIDX
    
    @staticmethod
    def __evaluate(C:np.ndarray, X, nvars:int):
        assert X.ndim <= 2 and X.size > 0 and X.shape[0] == nvars

        it = iter(X)
        x0 = next(it)
        __C = np.polynomial.polynomial.polyval(x0, C)
        for xi in it:
            __C = np.polynomial.polynomial.polyval(xi, __C, tensor=False)  # TODO: check X matrix as Fortran style more efficient (*).

        return __C  # (numpy.array of) numpy.float*


class ModelFactory:
    @staticmethod
    def create_poly(deg:int, nvars:int=1) -> Poly:
        assert deg >= 0 and nvars >= 1
        if nvars == 1: return Poly1d(deg)
        return Polynd(nvars, deg)
