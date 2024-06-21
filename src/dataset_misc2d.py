import numpy as np
import sympy

from dataset import Datasetnd


class Resistance2(Datasetnd):
    def __init__(self) -> None:
        super().__init__(xl=[0.,0.], xu=[20.,20.])
        self.yl =  0.
        self.yu = 10.
    
    def func(self, X) -> float:
        x0 = X[:,0]
        x1 = X[:,1]
        with np.errstate(divide='ignore', invalid='ignore'):
            return (x0*x1) / (x0+x1)
    
    def get_sympy(self, evaluated:bool=False):
        x0 = sympy.Symbol('x0')
        x1 = sympy.Symbol('x1')
        return (x0*x1) / (x0+x1)
    
    def get_name(self) -> str:
        return 'Resistance2'