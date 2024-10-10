import numpy as np
import sympy
import dataset

SPEED_OF_LIGHT = 2.99792458e8
PLANCK_CONSTANT = 6.626e-34
ELECTRIC_CONSTANT = 8.854e-12


class FeynmanICh6Eq20(dataset.Datasetnd):
    def __init__(self) -> None:
        super().__init__(xl=[-1.,0.0001], xu=[1.,1.])
        self.yl = 0.
        self.yu = 2.
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.xu, '+')
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign((0,), [0., 0.], self.xu, '-')
        self.knowledge.add_sign((0,), [self.xl[0], 0.], [0., self.xu[1]], '+')

    def func(self, X) -> float:
        x0 = X[:,0]
        x1 = X[:,1]
        return np.exp(-(x0/x1)**2 / 2.0) / (np.sqrt(2.0 * np.pi) * x1)
    
    def get_sympy(self, evaluated:bool=False):
        x0 = sympy.Symbol('theta')
        x1 = sympy.Symbol('sigma')
        return sympy.exp(-(x0/x1)**2 / 2.0) / (sympy.sqrt(2.0 * sympy.pi) * x1)
    
    def get_name(self) -> str:
        return 'feynman-i.6.20'