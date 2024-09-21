from symbols.syntax_tree import SyntaxTree
import numlims

class Derivative:
    @staticmethod
    def create(f:SyntaxTree, deriv:tuple[int], nvars:int, numlims:numlims.NumericLimits):
        assert nvars > 0
        derivdeg = len(deriv)
        if derivdeg == 0: return f
        if derivdeg == 1: return FirstDerivative(f, deriv[0], nvars, numlims)
        if derivdeg == 2:
            if nvars == 1: return UnidimSecondDerivative(f, nvars, numlims)
            else:
                dx1, dx2 = deriv
                if dx1 == dx2: return MultidimSingleVarSecondDerivative(f, dx1, nvars, numlims)
                return MultidimSecondDerivative(f, dx1, dx2, nvars, numlims)
        raise RuntimeError(f"{derivdeg}th derivative not supported.")
    
    @staticmethod
    def create_all(f:SyntaxTree, derivs:list[tuple[int]], nvars:int, numlims:numlims.NumericLimits):
        all_derivatives = {}
        for d in derivs:
            all_derivatives[d] = Derivative.create(f, d, nvars, numlims)
        return all_derivatives

class FirstDerivative(Derivative):
    def __init__(self, f:SyntaxTree, dx:int, nvars:int, numlims:numlims.NumericLimits):
        self.f = f
        self.h_scal = numlims.STEPSIZE
        self.h = np.zeros(nvars)
        self.h[dx] = self.h_scal
    
    def __call__(self, x):
        return (self.f(x + self.h) - self.f(x)) / self.h_scal

class UnidimSecondDerivative(Derivative):
    def __init__(self, f:SyntaxTree, nvars:int, numlims:numlims.NumericLimits):
        self.f = f
        self.h_scal_2 = numlims.STEPSIZE ** 2
        self.h = numlims.STEPSIZE
    
    def __call__(self, x):
        return (self.f(x + self.h) - 2*self.f(x) + self.f(x - self.h)) / self.h_scal_2

class MultidimSingleVarSecondDerivative(Derivative):
    def __init__(self, f:SyntaxTree, dx:int, nvars:int, numlims:numlims.NumericLimits):
        self.f = f
        self.h_scal_2 = numlims.STEPSIZE ** 2
        self.h = np.zeros(nvars)
        self.h[dx] = numlims.STEPSIZE
    
    def __call__(self, x):
        return (self.f(x + self.h) - 2*self.f(x) + self.f(x - self.h)) / self.h_scal_2

class MultidimSecondDerivative(Derivative):
    def __init__(self, f:SyntaxTree, dx1:int, dx2:int, nvars:int, numlims:numlims.NumericLimits):
        self.f = f
        self.h_scal_2_4 = (numlims.STEPSIZE ** 2) * 4

        self.h_dx1dx2   = np.zeros(nvars)
        self.h_dx1_dx2  = np.zeros(nvars)

        self.h_dx1dx2  [dx1] = numlims.STEPSIZE
        self.h_dx1dx2  [dx2] = numlims.STEPSIZE

        self.h_dx1_dx2 [dx1] = numlims.STEPSIZE
        self.h_dx1_dx2 [dx2] = -numlims.STEPSIZE
    
    def __call__(self, x):
        return (self.f(x + self.h_dx1dx2) - self.f(x + self.h_dx1_dx2) - self.f(x - self.h_dx1_dx2) + self.f(x - self.h_dx1dx2)) / self.h_scal_2_4
