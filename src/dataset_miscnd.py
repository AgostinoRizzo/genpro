import numpy as np
import sympy

from dataset import Datasetnd


class WavePower(Datasetnd):
    def __init__(self) -> None:
        super().__init__(
            xl=[1.0, 1.0, 1.0, 1.0, 1.0],
            xu=[2.0, 2.0, 5.0, 5.0, 2.0]
        )
        self.yl = 0.0
        self.yu = None
    
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.xu, '-')
        
        # monotonically increasing/decreasing
        for i in [0,2,3]:
            self.knowledge.add_sign((i,), self.xl, self.xu, '-')
        for i in [1,4]:
            self.knowledge.add_sign((i,), self.xl, self.xu, '+')
        
        # symmetry (w.r.t. variables).
        self.knowledge.add_symmvars((0,1,2,3,4))
        self.knowledge.add_symmvars((0,1,3,2,4))  # m2, m1
        #self.knowledge.add_symmvars((0,4,2,3,1))  # r, c
        #self.knowledge.add_symmvars((0,4,3,2,1))  # m2, m1, r, c

    def func(self, X) -> float:
        G  = X[:,0]
        c  = X[:,1]
        m1 = X[:,2]
        m2 = X[:,3]
        r  = X[:,4]
        return (((m1*m2)**2 * (m1+m2)) / (r**5))
        return (-32.0/5.0) * ((G**4)/(c**5)) * (((m1*m2)**2 * (m1+m2)) / (r**5))
    
    def get_sympy(self, evaluated:bool=False):
        G  = sympy.Symbol('G')
        c  = sympy.Symbol('c')
        m1 = sympy.Symbol('m1')
        m2 = sympy.Symbol('m2')
        r  = sympy.Symbol('r')
        return (-32.0/5.0) * ((G**4)/(c**5)) * (((m1*m2)**2 * (m1+m2)) / (r**5))
    
    def get_name(self) -> str:
        return 'Wave power'