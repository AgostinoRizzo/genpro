import numpy as np
import sympy
import dataset


class FeynmanICh41Eq16(dataset.Datasetnd):
    def __init__(self) -> None:
        super().__init__(xl=[1.]*5, xu=[5.]*5)
        self.yl = 0.
        self.yu = np.inf
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.xu, '+')
    
        # monotonically increasing/decreasing
        for i in [1, 3]:
            self.knowledge.add_sign((i,), self.xl, self.xu, '+')
        for i in [2, 4]:
            self.knowledge.add_sign((i,), self.xl, self.xu, '-')

    def func(self, X) -> float:
        omega = X[:,0]
        T     = X[:,1]
        h     = X[:,2]
        kb    = X[:,3]
        c     = X[:,4]
        return (h * omega**3) / (np.pi**2 * c**2 * (np.exp((h*omega)/(kb*T))-1))
    
    def get_sympy(self, evaluated:bool=False):
        omega = sympy.Symbol('omega')
        T     = sympy.Symbol('T')
        h     = sympy.Symbol('h')
        kb    = sympy.Symbol('kb')
        c     = sympy.Symbol('c')
        return (h * omega**3) / (sympy.pi**2 * c**2 * (sympy.exp((h*omega)/(kb*T))-1))
    
    def get_name(self) -> str:
        return 'I.41.16'
    
    def get_varnames(self) -> dict[int,str]:
        return {0: 'omega', 1: 'T', 2: 'h', 3: 'kb', 4: 'c'}


class FeynmanICh48Eq20(dataset.Datasetnd):
    def __init__(self) -> None:
        super().__init__(xl=[1.,1.,3.], xu=[5.,2.,20.])
        self.yl = 0.
        self.yu = np.inf
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.xu, '+')
    
        # monotonically increasing/decreasing
        for i in [0, 1, 2]:
            self.knowledge.add_sign((i,), self.xl, self.xu, '+')

    def func(self, X) -> float:
        m = X[:,0]
        v = X[:,1]
        c = X[:,2]
        return (m * c**2) / np.sqrt(1 - ((v**2)/(c**2)))
    
    def get_sympy(self, evaluated:bool=False):
        m = sympy.Symbol('m')
        v = sympy.Symbol('v')
        c = sympy.Symbol('c')
        return (m * c**2) / sympy.sqrt(1 - ((v**2)/(c**2)))
    
    def get_name(self) -> str:
        return 'I.48.20'
    
    def get_varnames(self) -> dict[int,str]:
        return {0: 'm', 1: 'v', 2: 'c'}


class FeynmanIICh6Eq15a(dataset.Datasetnd):
    def __init__(self) -> None:
        super().__init__(xl=[1.]*6, xu=[3.]*6)
        self.yl = 0.
        self.yu = np.inf
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.xu, '+')
    
        # monotonically increasing/decreasing
        for i in [1, 3, 4, 5]:
            self.knowledge.add_sign((i,), self.xl, self.xu, '+')
        for i in [0, 2]:
            self.knowledge.add_sign((i,), self.xl, self.xu, '-')

    def func(self, X) -> float:
        epsilon = X[:,0]
        p_d     = X[:,1]
        r       = X[:,2]
        x       = X[:,3]
        y       = X[:,4]
        z       = X[:,5]
        return ( (p_d*3*z/(4*np.pi*epsilon)) / r**5 ) * np.sqrt(x**2 + y**2)
    
    def get_sympy(self, evaluated:bool=False):
        epsilon = sympy.Symbol('epsilon')
        p_d     = sympy.Symbol('p_d')
        r       = sympy.Symbol('r')
        x       = sympy.Symbol('x')
        y       = sympy.Symbol('y')
        z       = sympy.Symbol('z')
        return ( (p_d*3*z/(4*sympy.pi*epsilon)) / r**5 ) * sympy.sqrt(x**2 + y**2)
    
    def get_name(self) -> str:
        return 'II.6.15a'
    
    def get_varnames(self) -> dict[int,str]:
        return {0: 'epsilon', 1: 'p_d', 2: 'r', 3: 'x', 4: 'y', 5: 'z'}


class FeynmanIICh11Eq27(dataset.Datasetnd):
    def __init__(self) -> None:
        super().__init__(xl=[0.,0.,1.,1.], xu=[1.,1.,2.,2.])
        self.yl = 0.
        self.yu = np.inf
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.xu, '+')
    
        # monotonically increasing/decreasing
        for i in [0, 1, 2, 3]:
            self.knowledge.add_sign((i,), self.xl, self.xu, '+')

    def func(self, X) -> float:
        n       = X[:,0]
        alpha   = X[:,1]
        epsilon = X[:,2]
        Ef      = X[:,3]
        return (n*alpha/(1-n*alpha/3)) * epsilon * Ef
    
    def get_sympy(self, evaluated:bool=False):
        n       = sympy.Symbol('n')
        alpha   = sympy.Symbol('alpha')
        epsilon = sympy.Symbol('epsilon')
        Ef      = sympy.Symbol('Ef')
        return (n*alpha/(1-n*alpha/3)) * epsilon * Ef
    
    def get_name(self) -> str:
        return 'II.11.27'
    
    def get_varnames(self) -> dict[int,str]:
        return {0: 'n', 1: 'alpha', 2: 'epsilon', 3: 'Ef'}


class FeynmanIICh11Eq28(dataset.Datasetnd):
    def __init__(self) -> None:
        super().__init__(xl=[0.,0.], xu=[1.,1.])
        self.yl = 0.
        self.yu = np.inf
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.xu, '+')
    
        # monotonically increasing/decreasing
        for i in [0, 1]:
            self.knowledge.add_sign((i,), self.xl, self.xu, '+')

    def func(self, X) -> float:
        n       = X[:,0]
        alpha   = X[:,1]
        return 1 + (n*alpha/(1-n*alpha/3))
    
    def get_sympy(self, evaluated:bool=False):
        n       = sympy.Symbol('n')
        alpha   = sympy.Symbol('alpha')
        return 1 + (n*alpha/(1-n*alpha/3))
    
    def get_name(self) -> str:
        return 'II.11.28'
    
    def get_varnames(self) -> dict[int,str]:
        return {0: 'n', 1: 'alpha'}


class FeynmanIIICh10Eq19(dataset.Datasetnd):
    def __init__(self) -> None:
        super().__init__(xl=[1.]*4, xu=[5.]*4)
        self.yl = 0.
        self.yu = np.inf
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.xu, '+')
    
        # monotonically increasing/decreasing
        for i in [0, 1, 2, 3]:
            self.knowledge.add_sign((i,), self.xl, self.xu, '+')

    def func(self, X) -> float:
        mom = X[:,0]
        Bx  = X[:,1]
        By  = X[:,2]
        Bz  = X[:,3]
        return mom * np.sqrt(Bx**2 + By**2 + Bz**2)
    
    def get_sympy(self, evaluated:bool=False):
        mom = sympy.Symbol('mom')
        Bx  = sympy.Symbol('Bx')
        By  = sympy.Symbol('By')
        Bz  = sympy.Symbol('Bz')
        return mom * np.sqrt(Bx**2 + By**2 + Bz**2)
    
    def get_name(self) -> str:
        return 'III.10.19'
    
    def get_varnames(self) -> dict[int,str]:
        return {0: 'mom', 1: 'Bx', 2: 'By', 3: 'Bz'}
