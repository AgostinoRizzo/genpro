import numpy as np
import sympy
import dataset

SPEED_OF_LIGHT = 2.99792458e8
PLANCK_CONSTANT = 6.626e-34
ELECTRIC_CONSTANT = 8.854e-12


class FeynmanICh6Eq20a(dataset.Dataset1d):
    def __init__(self) -> None:
        super().__init__(xl=-4., xu=4.)
        self.yl =  0.
        self.yu =  0.5

        INFTY = self.numlims.INFTY

        # intersection points
        self.knowledge.add_deriv(0, dataset.DataPoint(0, self.func(0)))
        self.knowledge.add_deriv(0, dataset.DataPoint(self.xl, self.func(self.xl)))
        self.knowledge.add_deriv(0, dataset.DataPoint(self.xu, self.func(self.xu)))

        # known (first) derivatives
        self.knowledge.add_deriv(1, dataset.DataPoint(0, 0))
        
        # known positivity/negativity
        self.knowledge.add_sign(0, -INFTY, INFTY, '+')
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, -INFTY, 0, '+')
        self.knowledge.add_sign(1, 0, INFTY, '-')

        # concavity/convexity
        self.knowledge.add_sign(2, -INFTY, -1, '+')
        self.knowledge.add_sign(2, -1, 1, '-')
        self.knowledge.add_sign(2, 1, INFTY, '+')

        # symmetry
        self.knowledge.add_symm(0, 0, iseven=True )
        self.knowledge.add_symm(1, 0, iseven=False)
        self.knowledge.add_symm(2, 0, iseven=True )

    def func(self, x: float) -> float:
        return np.exp(-((x**2) / 2)) / np.sqrt(2 * np.pi)
    
    def get_sympy(self, evaluated:bool=False):
        x  = sympy.Symbol('theta')
        return sympy.exp(-((x**2) / 2)) / sympy.sqrt(2 * sympy.pi)
    
    def get_name(self) -> str:
        return 'feynman-i.6.20a'
    
    def get_xlabel(self) -> str:
        return 'theta (x)'


class FeynmanICh29Eq4(dataset.Dataset1d):
    def __init__(self) -> None:
        super().__init__(xl=0., xu=1.)
        self.yl = 0
        self.yu = 1

        self._yl = 0
        self._yu = 4e-9

        INFTY = self.numlims.INFTY

        # intersection points
        self.knowledge.add_deriv(0, dataset.DataPoint(0, 0))
        
        # known positivity/negativity
        self.knowledge.add_sign(0, -INFTY, 0, '-')
        self.knowledge.add_sign(0, 0, INFTY, '+')
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, -INFTY, INFTY, '+')

        # symmetry
        self.knowledge.add_symm(0, 0, iseven=False)
        self.knowledge.add_symm(1, 0, iseven=True )
        self.knowledge.add_symm(2, 0, iseven=True )

    def func(self, x: float) -> float:
        y = x / SPEED_OF_LIGHT
        return self._ymap(y)
    
    def get_sympy(self, evaluated:bool=False):
        x = sympy.Symbol('omega')
        expr = x / SPEED_OF_LIGHT
        if evaluated: return self._ymap(expr)
        return expr
    
    def get_name(self) -> str:
        return 'feynman-i.29.4'
    
    def get_xlabel(self) -> str:
        return 'omega (x)'
    
    def is_yscaled(self) -> bool:
        return True


class FeynmanICh34Eq27(dataset.Dataset1d):
    def __init__(self) -> None:
        super().__init__(xl=0., xu=1.)
        self.yl = 0
        self.yu = 1

        self._yl = 0
        self._yu = 1.1e-34

        INFTY = self.numlims.INFTY

        # intersection points
        self.knowledge.add_deriv(0, dataset.DataPoint(0, 0))
        
        # known positivity/negativity
        self.knowledge.add_sign(0, -INFTY, 0, '-')
        self.knowledge.add_sign(0, 0, INFTY, '+')
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, -INFTY, INFTY, '+')

        # symmetry
        self.knowledge.add_symm(0, 0, iseven=False)
        self.knowledge.add_symm(1, 0, iseven=True )
        self.knowledge.add_symm(2, 0, iseven=True )

    def func(self, x: float) -> float:
        y = (PLANCK_CONSTANT / (2 * np.pi)) * x
        return self._ymap(y)
    
    def get_sympy(self, evaluated:bool=False):
        x  = sympy.Symbol('omega')
        expr = (PLANCK_CONSTANT / (2 * sympy.pi)) * x
        if evaluated: return self._ymap(expr)
        return expr
    
    def get_name(self) -> str:
        return 'feynman-i.34.27'
    
    def get_xlabel(self) -> str:
        return 'omega (x)'
    
    def is_yscaled(self) -> bool:
        return True


class FeynmanIICh8Eq31(dataset.Dataset1d):
    def __init__(self) -> None:
        super().__init__(xl=0., xu=1.)
        self.yl = 0
        self.yu = 1

        self._yl = 0
        self._yu = 4.5e-12

        INFTY = self.numlims.INFTY

        # intersection points
        self.knowledge.add_deriv(0, dataset.DataPoint(0, 0))
        
        # known positivity/negativity
        self.knowledge.add_sign(0, -INFTY, INFTY, '+')
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, -INFTY, 0, '-')
        self.knowledge.add_sign(1, 0, INFTY, '+')

        # concavity/convexity
        self.knowledge.add_sign(2, -INFTY, INFTY, '+')

        # symmetry
        self.knowledge.add_symm(0, 0, iseven=True )
        self.knowledge.add_symm(1, 0, iseven=False)
        self.knowledge.add_symm(2, 0, iseven=True )

    def func(self, x: float) -> float:
        y = (ELECTRIC_CONSTANT * (x ** 2)) / 2
        return self._ymap(y)
    
    def get_sympy(self, evaluated:bool=False):
        x  = sympy.Symbol('Ef')
        expr = (ELECTRIC_CONSTANT * (x ** 2)) / 2
        if evaluated: return self._ymap(expr)
        return expr
    
    def get_name(self) -> str:
        return 'feynman-ii.8.31'
    
    def get_xlabel(self) -> str:
        return 'Ef (x)'
    
    def is_yscaled(self) -> bool:
        return True


class FeynmanIICh27Eq16(dataset.Dataset1d):
    def __init__(self) -> None:
        super().__init__(xl=0., xu=1.)
        self.yl = 0
        self.yu = 1

        self._yl = 0
        self._yu = 0.003

        INFTY = self.numlims.INFTY

        # intersection points
        self.knowledge.add_deriv(0, dataset.DataPoint(0, 0))
        
        # known positivity/negativity
        self.knowledge.add_sign(0, -INFTY, INFTY, '+')
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, -INFTY, 0, '-')
        self.knowledge.add_sign(1, 0, INFTY, '+')

        # concavity/convexity
        self.knowledge.add_sign(2, -INFTY, INFTY, '+')

        # symmetry
        self.knowledge.add_symm(0, 0, iseven=True )
        self.knowledge.add_symm(1, 0, iseven=False)
        self.knowledge.add_symm(2, 0, iseven=True )

    def func(self, x: float) -> float:
        y = ELECTRIC_CONSTANT * SPEED_OF_LIGHT * (x ** 2)
        return self._ymap(y)
    
    def get_sympy(self, evaluated:bool=False):
        x  = sympy.Symbol('E_f')
        expr = ELECTRIC_CONSTANT * SPEED_OF_LIGHT * (x ** 2)
        if evaluated: return self._ymap(expr)
        return expr
    
    def get_name(self) -> str:
        return 'feynman-ii.27.16'
    
    def get_xlabel(self) -> str:
        return 'Ef (x)'
    
    def is_yscaled(self) -> bool:
        return True
