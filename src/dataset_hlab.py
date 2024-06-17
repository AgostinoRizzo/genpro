import numpy as np
import sympy
import dataset

class NguyenF1(dataset.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xl = -1.
        self.xu =  1.
        self.yl = -1.
        self.yu =  3.

        INFTY = self.numlims.INFTY

        # intersection points
        self.knowledge.add_deriv(0, dataset.DataPoint(0, 0))
        
        # known positivity/negativity
        self.knowledge.add_sign(0, -INFTY, 0, '-')
        self.knowledge.add_sign(0, 0, INFTY, '+')
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, -INFTY, INFTY, '+')

        # concavity/convexity
        self.knowledge.add_sign(2, -INFTY, -0.3333, '-')
        self.knowledge.add_sign(2, -0.3333, INFTY, '+')

    def func(self, x: float) -> float:
        return x**3 + x**2 + x
    
    def get_sympy(self, evaluated:bool=False):
        x  = sympy.Symbol('x')
        return x**3 + x**2 + x
    
    def get_name(self) -> str:
        return 'nguyen-f1'


class NguyenF4(dataset.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xl = -1.
        self.xu =  1.
        self.yl = -0.4
        self.yu =  6.

        INFTY = self.numlims.INFTY

        # intersection points
        self.knowledge.add_deriv(0, dataset.DataPoint(0, 0))
        
        # known positivity/negativity
        self.knowledge.add_sign(0, -INFTY, -1, '+')
        self.knowledge.add_sign(0, -1, 0, '-')
        self.knowledge.add_sign(0, 0, INFTY, '+')
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, -INFTY, -0.6703, '-')
        self.knowledge.add_sign(1, -0.6703, INFTY, '+')

        # concavity/convexity
        self.knowledge.add_sign(2, -INFTY, INFTY, '+')

    def func(self, x: float) -> float:
        return x**6 + x**5 + x**4 + x**3 + x**2 + x
    
    def get_sympy(self, evaluated:bool=False):
        x  = sympy.Symbol('x')
        return x**6 + x**5 + x**4 + x**3 + x**2 + x
    
    def get_name(self) -> str:
        return 'nguyen-f4'


class NguyenF7(dataset.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xl = 0.01
        self.xu = 2.
        self.yl = 0.01
        self.yu = 2.8

        INFTY = self.numlims.INFTY

        # intersection points
        self.knowledge.add_deriv(0, dataset.DataPoint(0, 0))
        
        # known positivity/negativity
        self.knowledge.add_sign(0, -1, 0, '-')  # TODO: the function is not defined for x < -1.
        self.knowledge.add_sign(0, 0, INFTY, '+')
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, -1, INFTY, '+')

        # concavity/convexity
        self.knowledge.add_sign(2, -1, -0.2333, '-')
        self.knowledge.add_sign(2, -0.2333, 0.7712, '+')
        self.knowledge.add_sign(2, 0.7712, INFTY, '-')

    def func(self, x: float) -> float:
        return np.log(x+1) + np.log((x**2)+1)
    
    def get_sympy(self, evaluated:bool=False):
        x  = sympy.Symbol('x')
        return sympy.log(x+1) + sympy.log((x**2)+1)
    
    def get_name(self) -> str:
        return 'nguyen-f7'


class Keijzer7(dataset.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xl = 1.
        self.xu = 100.
        self.yl = 0.
        self.yu = 5.

        INFTY = self.numlims.INFTY

        # intersection points
        self.knowledge.add_deriv(0, dataset.DataPoint(1, 0))
        
        # known positivity/negativity
        self.knowledge.add_sign(0, 0, 1, '-')  # TODO: the function is not defined for x < 0.
        self.knowledge.add_sign(0, 1, INFTY, '+')
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, 0, INFTY, '+')

        # concavity/convexity
        self.knowledge.add_sign(2, 0, INFTY, '-')

    def func(self, x: float) -> float:
        return np.log(x)
    
    def get_sympy(self, evaluated:bool=False):
        x  = sympy.Symbol('x')
        return sympy.log(x)
    
    def get_name(self) -> str:
        return 'keijzer-7'


class Keijzer8(dataset.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xl = 0.
        self.xu = 100.
        self.yl = 0.
        self.yu = 10.

        INFTY = self.numlims.INFTY

        # intersection points
        self.knowledge.add_deriv(0, dataset.DataPoint(0, 0))
        
        # known positivity/negativity
        self.knowledge.add_sign(0, 0, INFTY, '+')  # TODO: the function is not defined for x < 0.
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, 0, INFTY, '+')

        # concavity/convexity
        self.knowledge.add_sign(2, 0, INFTY, '-')

    def func(self, x: float) -> float:
        return np.sqrt(x)
    
    def get_sympy(self, evaluated:bool=False):
        x  = sympy.Symbol('x')
        return sympy.sqrt(x)
    
    def get_name(self) -> str:
        return 'keijzer-8'
