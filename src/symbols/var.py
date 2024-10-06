import sympy
import numpy as np
from symbols.syntax_tree import SyntaxTree


class VariableSyntaxTree(SyntaxTree):
    def __init__(self, idx:int=0):
        super().__init__()
        self.idx = idx
    
    def clone(self):
        c = VariableSyntaxTree(self.idx)
        c.copy_output_from(self)
        return c
    
    def __call__(self, x):
        if self.output is None:
            self.output = x[:,self.idx] if x.ndim == 2 else x
        return self.output
    
    def __getitem__(self, x_d):
        x, d = x_d
        if d not in self.y_know:
            if d == ():
                self.y_know[d] = x[:,self.idx] if x.ndim == 2 else x
            elif len(d) == 1 and d[0] == self.idx:
                self.y_know[d] = np.ones(x.shape[0])
            else:
                self.y_know[d] = np.zeros(x.shape[0])
        return self.y_know[d]
    
    def __str__(self) -> str:
        return f"x{self.idx}"
    
    def __eq__(self, other) -> bool:
        if type(other) is not VariableSyntaxTree: return False
        return self.idx == other.idx
    
    def at(self, x):
        return x[:,self.idx] if x.ndim == 2 else x
    
    def diff(self, varidx:int=0) -> SyntaxTree:
        return ConstantSyntaxTree(0.0 if self.is_const_wrt(varidx) else 1.0)
    
    def is_const(self) -> bool:
        return False
    
    def is_const_wrt(self, varidx):
        return self.idx != varidx
    
    def accept(self, visitor):
        visitor.visitVariable(self)
    
    def to_sympy(self, dps:int=None):
        return sympy.Symbol(str(self))
    
    def match(self, trunk) -> bool:
        if type(trunk) is UnknownSyntaxTree: return True
        if type(self) is not type(trunk): return False
        if self.idx != trunk.idx: return False
        return True