import numpy as np
import sympy

from dataset import Datasetnd, DataPoint


class Gravity(Datasetnd):
    def __init__(self) -> None:
        super().__init__(xl=[0.0001,0.0001,0.0001], xu=[20.,20.,20.])
        self.def_xu = np.array([10.,10.,10.], dtype=float)  # partial domain definition
        self.yl = 0.
        self.yu = 2.6696
    
        # undef points.
        self.knowledge.add_undef(np.zeros(3))
        
        INFTY = self.numlims.INFTY
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, [INFTY, INFTY, INFTY], '+')
        
        # monotonically increasing/decreasing
        self.knowledge.add_sign((0,), self.xl, [INFTY, INFTY, INFTY], '+')
        self.knowledge.add_sign((1,), self.xl, [INFTY, INFTY, INFTY], '+')

        # symmetry (w.r.t. variables).
        self.knowledge.add_symmvars((0,1,2))
        self.knowledge.add_symmvars((1,0,2))

    def func(self, X) -> float:
        x0 = X[:,0]
        x1 = X[:,1]
        x2 = X[:,2]
        with np.errstate(divide='ignore', invalid='ignore'):
            return (6.674e-11 * x0 * x1) / (x2**2)
    
    def get_sympy(self, evaluated:bool=False):
        x0 = sympy.Symbol('m1')
        x1 = sympy.Symbol('m2')
        x1 = sympy.Symbol('r')
        return (6.674e-11 * x0 * x1) / (x2**2)
    
    def get_name(self) -> str:
        return 'Gravity'


class Resistance3(Datasetnd):
    def __init__(self) -> None:
        super().__init__(xl=[0.0001,0.0001,0.0001], xu=[20.,20.,20.])
        self.def_xu = np.array([10.,10.,10.], dtype=float)  # partial domain definition
        self.yl = 0.
        self.yu = 6.7
    
        # undef points.
        self.knowledge.add_undef(np.zeros(3))
        
        INFTY = self.numlims.INFTY
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, [INFTY, INFTY, INFTY], '+')
        
        # monotonically increasing/decreasing
        #self.knowledge.add_sign((0,), self.xl, [INFTY, INFTY, INFTY], '+')
        #self.knowledge.add_sign((1,), self.xl, [INFTY, INFTY, INFTY], '+')
        #self.knowledge.add_sign((3,), self.xl, [INFTY, INFTY, INFTY], '+')

        # symmetry (w.r.t. variables).
        self.knowledge.add_symmvars((0,1,2))
        self.knowledge.add_symmvars((1,0,2))
        self.knowledge.add_symmvars((2,1,0))
        self.knowledge.add_symmvars((0,2,1))

    def func(self, X) -> float:
        x0 = X[:,0]
        x1 = X[:,1]
        x2 = X[:,2]
        with np.errstate(divide='ignore', invalid='ignore'):
            return (x0 * x1 * x2) / (x0*x1 + x0*x2 + x1*x2)
    
    def get_sympy(self, evaluated:bool=False):
        x0 = sympy.Symbol('m1')
        x1 = sympy.Symbol('m2')
        x1 = sympy.Symbol('r')
        return (6.674e-11 * x0 * x1) / (x2**2)
    
    def get_name(self) -> str:
        return 'Resistance3'