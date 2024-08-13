import numpy as np
import sympy

from dataset import Dataset1d, DataPoint


class MockDataset(Dataset1d): 
    def __init__(self) -> None:
        super().__init__(xl=0., xu=1.)
        self.yl = 0.
        self.yu = 1.
     
    def func(self, x: float) -> float:
        return (x**3 -2*x + 1) / (x*3 + x -1) #np.sin(x) + 1  #x / (x**2)#(x+2) / (x**2 + x + 1)
    
    def get_name(self) -> str:
        return 'Mock'


class PolyDataset(Dataset1d):   
    def __init__(self) -> None:
        super().__init__(xl=0.01, xu=4.)
        self.yl = 0.01
        self.yu = 16.
     
    def func(self, x: float) -> float:
        return x **2
        #return 0.2*x**4 -1*x**2 + 1.3
    
    def get_name(self) -> str:
        return 'Poly'


class TrigonDataset(Dataset1d):   
    def __init__(self) -> None:
        super().__init__(xl=-5., xu=5.)
        self.yl = -1.
        self.yu = 1.

        """self.knowledge.add_deriv(0, DataPoint( 0., 0.))
        self.knowledge.add_deriv(0, DataPoint(  .5*math.pi,  1.))
        self.knowledge.add_deriv(0, DataPoint( -.5*math.pi, -1.))
        self.knowledge.add_deriv(0, DataPoint( 1.5*math.pi, -1.))
        self.knowledge.add_deriv(0, DataPoint(-1.5*math.pi,  1.))

        self.knowledge.add_deriv(1, DataPoint(  .5*math.pi,  0.))
        self.knowledge.add_deriv(1, DataPoint( -.5*math.pi,  0.))
        self.knowledge.add_deriv(1, DataPoint( 1.5*math.pi,  0.))
        self.knowledge.add_deriv(1, DataPoint(-1.5*math.pi,  0.))"""
     
    def func(self, x: float) -> float:
        return np.sin(x)
    
    def get_name(self) -> str:
        return 'Trigon'


class MagmanDataset(Dataset1d):
    def __init__(self) -> None:
        super().__init__(xl=-0.075, xu=0.075)
        self.yl = -0.25
        self.yu =  0.25
        
        self.c1 = .00032
        self.c2 = .000305
        self.i  = .000004
        peak_x  = 0.00788845
        
        # intersection points
        """self.knowledge.add_deriv(0, DataPoint( 0., 0.))
        self.knowledge.add_deriv(0, DataPoint(-peak_x, self.func(-peak_x)))
        self.knowledge.add_deriv(0, DataPoint( peak_x, self.func( peak_x)))
        self.knowledge.add_deriv(0, DataPoint(self.xl, self.func(self.xl)))
        self.knowledge.add_deriv(0, DataPoint(self.xu, self.func(self.xu)))"""

        # known (first) derivatives
        """self.knowledge.add_deriv(1, DataPoint(-peak_x,  0.))
        self.knowledge.add_deriv(1, DataPoint( peak_x,  0.))"""

        #
        # positivity/negativity contraints
        #
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, -0.00001, '+')
        self.knowledge.add_sign(0, 0.00001, self.xu, '-')
    
        # monotonically increasing/decreasing
        """self.knowledge.add_sign(1, self.xl, -0.01, '+')
        #self.knowledge.add_sign(1, -peak_x+0.1, peak_x-0.1, '-')
        self.knowledge.add_sign(1, -0.01, self.xu, '+')"""

        # concavity/convexity
        """self.knowledge.add_sign(2, self.xl, -0.01, '+')
        self.knowledge.add_sign(2, 0.01, self.xu, '-')"""

    def func(self, x: float) -> float:
        return -self.i*self.c1*x / (x**2 + self.c2)**3
    
    def get_sympy(self, evaluated:bool=False):
        x  = sympy.Symbol(self.get_varnames()[0])
        i  = sympy.Symbol('i')
        c1 = sympy.Symbol('c1')
        c2 = sympy.Symbol('c2')
        expr = -i*c1*x / (x**2 + c2)**3
        if evaluated: return expr.subs( {i:self.i, c1:self.c1, c2:self.c2} )
        return expr
        #return '-\frac{i \cdot c_1 \cdot x}{\left(x^2 + c_2\right)^3}'
    
    def get_name(self) -> str:
        return 'magman'
    
    def get_xlabel(self, xidx:int=0) -> str:
        return 'distance [m] (x)'

    def get_ylabal(self) -> str:
        return 'force [N] (y)'


class MagmanDatasetScaled(Dataset1d):
    def __init__(self) -> None:
        super().__init__(xl=-2., xu=2.)
        self.yl = -2.
        self.yu = 2.

        self._xl = -0.075
        self._xu =  0.075
        self._yl = -0.25
        self._yu =  0.25

        #self.c1 = 1.4
        #self.c2 = 1.2
        #self.i = 7.
        #peak_x = 0.5

        self.c1 = .00032
        self.c2 = .000305
        self.i = .000004
        peak_x = self._xmap(0.00781024967)
        infl_x = self._xmap(0.01352774925)
        #peak_x = 0.20827333333333353
        
        """# intersection points
        self.knowledge.add_deriv(0, DataPoint( 0., 0.))
        self.knowledge.add_deriv(0, DataPoint(-peak_x, self.func(-peak_x)))
        self.knowledge.add_deriv(0, DataPoint( peak_x, self.func( peak_x)))
        self.knowledge.add_deriv(0, DataPoint(self.xl, self.func(self.xl)))
        self.knowledge.add_deriv(0, DataPoint(self.xu, self.func(self.xu)))"""
        #infty = 10
        #self.knowledge.add_deriv(0, DataPoint(-infty, 0))
        #self.knowledge.add_deriv(0, DataPoint(+infty, 0))

        # known (first) derivatives
        """self.knowledge.add_deriv(1, DataPoint(-peak_x,  0.))
        self.knowledge.add_deriv(1, DataPoint( peak_x,  0.))"""

        #
        # positivity/negativity contraints
        #
        INFTY = self.numlims.INFTY

        # known positivity/negativity
        #self.knowledge.add_sign(0, self.xl, -0.001, '+')
        #self.knowledge.add_sign(0, 0.001, self.xu, '-')
        self.knowledge.add_sign(0, -INFTY, 0, '+')
        self.knowledge.add_sign(0, 0, INFTY, '-')
    
        # monotonically increasing/decreasing
        #self.knowledge.add_sign(1, self.xl, -peak_x, '+')
        #self.knowledge.add_sign(1, -peak_x, peak_x, '-')
        #self.knowledge.add_sign(1, peak_x, self.xu, '+')
        self.knowledge.add_sign(1, -INFTY, -peak_x, '+')
        self.knowledge.add_sign(1, -peak_x, peak_x, '-')
        self.knowledge.add_sign(1, peak_x, INFTY, '+')

        """# concavity/convexity
        #self.knowledge.add_sign(2, self.xl, -0.4, '+')
        #self.knowledge.add_sign(2, 0.4, self.xu, '-')
        
        self.knowledge.add_sign(2, -INFTY, -infl_x, '+')
        self.knowledge.add_sign(2, -infl_x, 0, '-')
        self.knowledge.add_sign(2, 0, infl_x, '+')
        self.knowledge.add_sign(2, infl_x, INFTY, '-')

        """# symmetry
        self.knowledge.add_symm(0, 0, iseven=False)
        self.knowledge.add_symm(1, 0, iseven=True )
        self.knowledge.add_symm(2, 0, iseven=False)

    def func(self, x: float) -> float:
        x = self._xmap(x, toorigin=True)
        y = -self.i*self.c1*x / (x**2 + self.c2)**3
        return self._ymap(y)
    
    def get_sympy(self, evaluated:bool=False):
        x  = sympy.Symbol(self.get_varnames()[0])
        i  = sympy.Symbol('i')
        c1 = sympy.Symbol('c1')
        c2 = sympy.Symbol('c2')
        if evaluated:
            x = self._xmap(x, toorigin=True)
        expr = -i*c1*x / (x**2 + c2)**3
        if evaluated:
            expr = self._ymap(expr)
            return expr.subs( {i:self.i, c1:self.c1, c2:self.c2} )
        return expr
        #return '-\frac{i \cdot c_1 \cdot x}{\left(x^2 + c_2\right)^3}'

    def deriv(self, x: float) -> float:
        x = self._xmap(x, toorigin=True)
        y = (6.4e-9 * x**2 - 3.904e-13) / (x**2 + 0.000305) ** 4
        return self._ymap(y)
    
    def get_name(self) -> str:
        return 'magman'
    
    def get_xlabel(self, xidx:int=0) -> str:
        return 'distance [m] (x)'

    def get_ylabal(self) -> str:
        return 'force [N] (y)'
    
    def is_xscaled(self) -> bool:
        return True
    
    def is_yscaled(self) -> bool:
        return True


class HEADataset(Dataset1d):
    def __init__(self) -> None:
        super().__init__(xl=0., xu=10.)
        self.yl = -1.
        self.yu = 1.
    
    def func(self, x: float) -> float:
        if type(x) == float: return self.__func(x)
        y = []
        for _x in x: y.append(self.__func(_x))
        return y

    def __func(self, x: float) -> float:
        return math.e**(-x) * x**3 * math.cos(x) * math.sin(x) * (math.cos(x) * math.sin(x)**2 - 1)
    
    def get_name(self) -> str:
        return 'hea'


class ABSDataset(Dataset1d):
    def __init__(self) -> None:
        super().__init__(xl=0., xu=1.)
        self.yl = 0.
        self.yu = 0.45

        self.m = 6.67 #407.75
        self.g = 0.15 #9.81
        self.b = 55.56
        self.c = 1.35
        self.d = 0.4
        self.e = 0.52

        #
        # prior knowledge
        #

        peak_x = 0.0618236
        infl_x = 0.10629991
        peak_y = self.func(peak_x)

        # intersection points
        self.knowledge.add_deriv(0, DataPoint(0., 0.))
        self.knowledge.add_deriv(0, DataPoint(peak_x, peak_y))
        self.knowledge.add_deriv(1, DataPoint(peak_x, 0.))
        
        # known positivity/negativity
        #self.knowledge.add_sign(0, self.xl, self.numlims.INFTY, '+')
        self.knowledge.add_sign(0, self.xl, self.xu, '+')

        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, self.xl, peak_x, '+')
        #self.knowledge.add_sign(1, peak_x, self.numlims.INFTY, '-')
        self.knowledge.add_sign(1, peak_x, self.xu, '-')

        # concavity
        self.knowledge.add_sign(2, self.xl, infl_x, '-')
        #self.knowledge.add_sign(2, peak_x, self.numlims.INFTY, '+')
        self.knowledge.add_sign(2, infl_x, self.xu, '+')
    
    def func(self, x: float) -> float:
        return self.m * self.g * self.d * np.sin(self.c * np.arctan(self.b * (1 - self.e) * x + self.e * np.arctan(self.b * x)))
    
    def get_sympy(self, evaluated:bool=False):
        x = sympy.Symbol(self.get_varnames()[0])
        b = sympy.Symbol('b')
        c = sympy.Symbol('c')
        d = sympy.Symbol('d')
        e = sympy.Symbol('e')
        g = sympy.Symbol('g')
        m = sympy.Symbol('m')
        expr = m * g * d * sympy.sin(c * sympy.atan(b * (1 - e) * x + e * sympy.atan(b * x)))
        if evaluated: return expr.subs( {b:self.b, c:self.c, d:self.d, e:self.e, g:self.g, m:self.m} )
        return expr
        #return 'm \cdot g \cdot d \cdot \sin\left( c \cdot \arctan\left( b \cdot (1-e)^x + e^{\arctan\left(b \cdot x\right)}\right)\right)'
    
    def get_name(self) -> str:
        return 'abs'
    
    def get_xlabel(self, xidx:int=0) -> str:
        return 'k'

    def get_ylabal(self) -> str:
        return 'F(k) [N]'
    
    def get_varnames(self) -> dict[int,str]:
        return {0: 'kappa'}


class ABSDatasetScaled(Dataset1d):
    def __init__(self) -> None:
        super().__init__(xl=0., xu=1.)
        self.yl = 0.
        self.yu = 10

        self._xl = 0.
        self._xu = 1.
        self._yl = 0.
        self._yu = 0.45

        self.m = 6.67
        self.g = 0.15
        self.b = 55.56
        self.c = 1.35
        self.d = 0.4
        self.e = 0.52

        #
        # prior knowledge
        #

        peak_x = self._xmap(0.0618236)
        infl_x = self._xmap(0.10629991)
        peak_y = self.func(peak_x)

        # intersection points
        self.knowledge.add_deriv(0, DataPoint( 0., 0.))
        self.knowledge.add_deriv(0, DataPoint(peak_x, peak_y))
        self.knowledge.add_deriv(1, DataPoint(peak_x, 0.))
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.numlims.INFTY, '+')

        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, self.xl, peak_x, '+')
        self.knowledge.add_sign(1, peak_x, self.numlims.INFTY, '-')

        # concavity
        self.knowledge.add_sign(2, self.xl, infl_x, '-')
        self.knowledge.add_sign(2, infl_x, self.numlims.INFTY, '+')
    
    def func(self, x: float) -> float:
        x = self._xmap(x, toorigin=True)
        y = self.m * self.g * self.d * np.sin(self.c * np.arctan(self.b * (1 - self.e) * x + self.e * np.arctan(self.b * x)))
        return self._ymap(y)
    
    def get_sympy(self, evaluated:bool=False):
        x = sympy.Symbol(self.get_varnames()[0])
        b = sympy.Symbol('b')
        c = sympy.Symbol('c')
        d = sympy.Symbol('d')
        e = sympy.Symbol('e')
        g = sympy.Symbol('g')
        m = sympy.Symbol('m')
        if evaluated:
            x = self._xmap(x, toorigin=True)
        expr = m * g * d * sympy.sin(c * sympy.atan(b * (1 - e) * x + e * sympy.atan(b * x)))
        if evaluated:
            expr = self._ymap(expr)
            return expr.subs( {b:self.b, c:self.c, d:self.d, e:self.e, g:self.g, m:self.m} )
        return expr
        #return 'm \cdot g \cdot d \cdot \sin\left( c \cdot \arctan\left( b \cdot (1-e)^x + e^{\arctan\left(b \cdot x\right)}\right)\right)'
    
    def get_name(self) -> str:
        return 'abs'
    
    def get_xlabel(self, xidx:int=0) -> str:
        return 'k'

    def get_ylabal(self) -> str:
        return 'F(k) [N]'
    
    def is_yscaled(self) -> bool:
        return True
    
    def get_varnames(self) -> dict[int,str]:
        return {0: 'kappa'}


class OneOverXDataset(Dataset1d):
    def __init__(self) -> None:
        super().__init__(xl=1e-10, xu=5.)
        self.yl = 0.
        self.yu = 5.

        #
        # prior knowledge
        #

        # intersection points
        self.knowledge.add_deriv(0, DataPoint( 1., 1.))
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.numlims.INFTY, '+')

        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, self.xl, self.numlims.INFTY, '-')

        # concavity
        self.knowledge.add_sign(2, self.xl, self.numlims.INFTY, '+')
    
    def func(self, x: float) -> float:
        return 1 / x
    
    def get_sympy(self, evaluated:bool=False):
        x = sympy.Symbol(self.get_varnames()[0])
        return 1 / x
        #return '\frac{1}{x}'
    
    def get_name(self) -> str:
        return '1/x'