import numpy as np
import sympy
import math
from dataset import Datasetnd


class AircraftLift(Datasetnd):
    def __init__(self) -> None:
        super().__init__(
            xl=[0.4,  5.0, 0.4,  5.0, 1.0, 5.0],
            xu=[0.8, 10.0, 0.8, 10.0, 1.5, 7.0]
        )
        self.yl =  0.0
        self.yu = 20.0

        self.a0 = -2.0
    
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.xu, '+')
        
        # monotonically increasing/decreasing
        for i in range(0, 5):
            self.knowledge.add_sign((i,), self.xl, self.xu, '+')
        self.knowledge.add_sign((5,), self.xl, self.xu, '-')

    def func(self, X) -> float:
        C_La   = X[:,0]
        a      = X[:,1]
        C_Ld_e = X[:,2]
        d_e    = X[:,3]
        S_HT   = X[:,4]
        S_ref  = X[:,5]
        return C_La * (a - self.a0) + C_Ld_e * d_e * S_HT / S_ref
    
    def get_sympy(self, evaluated:bool=False):
        C_La   = sympy.Symbol('C_La')
        a      = sympy.Symbol('a')
        C_Ld_e = sympy.Symbol('C_Ld_e')
        d_e    = sympy.Symbol('d_e')
        S_HT   = sympy.Symbol('S_HT')
        S_ref  = sympy.Symbol('S_ref')
        return C_La * (a - self.a0) + C_Ld_e * d_e * S_HT / S_ref
    
    def get_name(self) -> str:
        return 'Aircraft lift'


class RocketFuelFlow(Datasetnd):
    def __init__(self) -> None:
        super().__init__(
            xl=[4.0e5, 0.5, 250.0],
            xu=[6.0e5, 1.5, 260.0]
        )
        self.yl = 0.
        self.yu = 100000.

        R = 287.0
        γ = 1.4
        self.c = math.sqrt((γ / R) * (2 / (γ + 1)) ** ((γ + 1) / (γ - 1)))
    
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.xu, '+')
        
        # monotonically increasing/decreasing
        self.knowledge.add_sign((0,), self.xl, self.xu, '+')
        self.knowledge.add_sign((1,), self.xl, self.xu, '+')
        self.knowledge.add_sign((2,), self.xl, self.xu, '-')

        # symmetry (w.r.t. variables).
        #self.knowledge.add_symmvars((0,1,2))
        #self.knowledge.add_symmvars((1,0,2))  # p0, A

    def func(self, X) -> float:
        p0 = X[:,0]
        A  = X[:,1]
        T0 = X[:,2]
        return ((p0 * A) / np.sqrt(T0)) * self.c
    
    def get_sympy(self, evaluated:bool=False):
        p0 = sympy.Symbol('p0')
        A  = sympy.Symbol('A')
        T0 = sympy.Symbol('T0')
        return ((p0 * A) / sympy.sqrt(T0)) * self.c
    
    def get_name(self) -> str:
        return 'Rocket fuel flow'
