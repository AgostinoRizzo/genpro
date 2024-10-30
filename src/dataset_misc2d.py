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