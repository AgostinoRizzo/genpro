import numpy as np
import sympy

from dataset import Datasetnd, DataPoint


class Resistance2(Datasetnd):
    def __init__(self) -> None:
        super().__init__(xl=[0.0001,0.0001], xu=[20.,20.])
        self.def_xu = np.array([10.,10.], dtype=float)  # partial domain definition
        self.yl =  0.
        self.yu = 10.
    
        # intersection points
        # self.knowledge.add_deriv(0, DataPoint( [0.,0.], 0. ))  TODO: add undef (0,0)
        self.knowledge.add_deriv(0, DataPoint( [self.xl[0],self.xu[1]], 0. ))
        self.knowledge.add_deriv(0, DataPoint( [self.xu[0],self.xl[1]], 0. ))

        # undef points.
        self.knowledge.add_undef(np.zeros(2))
        
        INFTY = self.numlims.INFTY
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, [INFTY, INFTY], '+')
        
        # monotonically increasing/decreasing
        self.knowledge.add_sign((0,), self.xl, [INFTY, INFTY], '+')
        self.knowledge.add_sign((1,), self.xl, [INFTY, INFTY], '+')

        # concavity
        #self.knowledge.add_sign((0,0), self.xl, [INFTY, INFTY], '-')
        #self.knowledge.add_sign((1,1), self.xl, [INFTY, INFTY], '-')

        # symmetry (w.r.t. variables).
        self.knowledge.add_symmvars((0,1))
        self.knowledge.add_symmvars((1,0))
        
    def func(self, X) -> float:
        x0 = X[:,0]
        x1 = X[:,1]
        with np.errstate(divide='ignore', invalid='ignore'):
            return (x0*x1) / (x0+x1)
    
    def get_sympy(self, evaluated:bool=False):
        varnames = self.get_varnames()
        x0 = sympy.Symbol(varnames[0])
        x1 = sympy.Symbol(varnames[1])
        return (x0*x1) / (x0+x1)
    
    def get_name(self) -> str:
        return 'Resistance2'
    
    def get_varnames(self) -> dict[int,str]:
        return {0: 'r_1', 1: 'r_2'}


class Keijzer14(Datasetnd):
    def __init__(self) -> None:
        super().__init__(xl=[-10.,-10.], xu=[10.,10.])
        self.def_xl = np.array([-4.,-4.], dtype=float)  # partial domain definition
        self.def_xu = np.array([ 4., 4.], dtype=float)
        self.yl = 0.
        self.yu = 0.
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.xu, '+')

        # symmetry (w.r.t. variables).
        self.knowledge.add_symmvars((0,1))
        self.knowledge.add_symmvars((1,0))
    
    def get_varnames(self) -> dict[int,str]:
        return {0: 'x', 1: 'y'}
    
    def func(self, X) -> float:
        x = X[:,0]
        y = X[:,1]
        with np.errstate(divide='ignore', invalid='ignore'):
            return 8.0 / (2.0 + x**2 + y**2)
    
    def get_sympy(self, evaluated:bool=False):
        varnames = self.get_varnames()
        x = sympy.Symbol(varnames[0])
        y = sympy.Symbol(varnames[1])
        return 8.0 / (2.0 + x**2 + y**2)
    
    def get_name(self) -> str:
        return 'keijzer14'


class Pagie1(Datasetnd):
    def __init__(self) -> None:
        super().__init__(xl=[-10.,-10.], xu=[10.,10.])
        self.yl = 0.
        self.yu = 2.
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.xu, '+')

        # symmetry (w.r.t. variables).
        self.knowledge.add_symmvars((0,1))
        self.knowledge.add_symmvars((1,0))
    
    def get_varnames(self) -> dict[int,str]:
        return {0: 'x', 1: 'y'}
    
    def func(self, X) -> float:
        x = X[:,0]
        y = X[:,1]
        with np.errstate(divide='ignore', invalid='ignore'):
            return (1.0 / (1.0 + x**(-4))) + (1.0 / (1.0 + y**(-4)))
    
    def get_sympy(self, evaluated:bool=False):
        varnames = self.get_varnames()
        x = sympy.Symbol(varnames[0])
        y = sympy.Symbol(varnames[1])
        return (1.0 / (1.0 + x**(-4))) + (1.0 / (1.0 + y**(-4)))
    
    def get_name(self) -> str:
        return 'pagie1'