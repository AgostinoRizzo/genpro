import numpy as np
import sympy
import dataset


class FeynmanICh6Eq20(dataset.Datasetnd):
    def __init__(self) -> None:
        super().__init__(xl=[1.,1.], xu=[3.,3.])
        self.yl = 0.
        self.yu = np.inf
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.xu, '+')
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign((1,), self.xl, self.xu, '-')

    def func(self, X) -> float:
        sigma = X[:,0]
        theta = X[:,1]
        return np.exp(-(theta/sigma)**2 / 2.0) / (np.sqrt(2.0 * np.pi) * sigma)
    
    def get_sympy(self, evaluated:bool=False):
        sigma = sympy.Symbol('sigma')
        theta = sympy.Symbol('theta')
        return sympy.exp(-(theta/sigma)**2 / 2.0) / (sympy.sqrt(2.0 * sympy.pi) * sigma)
    
    def get_name(self) -> str:
        return 'I.6.20'
    
    def get_varnames(self) -> dict[int,str]:
        return {0: 'sigma', 1: 'theta'}