import numpy as np
import sympy
from symbols.syntax_tree import SyntaxTree


class ConstantSyntaxTree(SyntaxTree):
    def __init__(self, val:float):
        super().__init__()
        self.val = val
    
    def clone(self):
        c = ConstantSyntaxTree(self.val)
        c.copy_output_from(self)
        return c
    
    def __call__(self, x):
        if self.output is None:
            self.output = np.full(x.shape[0], self.val)
        return self.output
    
    def __getitem__(self, x_d):
        x, d = x_d
        if d not in self.y_know:
            self.y_know[d] = np.full(x.shape[0], self.val)
        return self.y_know[d]
    
    def get_coeffs(self, coeffs:list):
        coeffs.append(self.val)
    
    def set_coeffs(self, coeffs:list, start:int=0):
        self.val = coeffs[start]
        start += 1
    
    def __str__(self) -> str:
        return "%.2f" % self.val
    
    def __eq__(self, other) -> bool:
        if type(other) is not ConstantSyntaxTree: return False
        return self.val == other.val
    
    def diff(self, varidx:int=0) -> SyntaxTree:
        return ConstantSyntaxTree(0.0)
    
    def is_const(self) -> bool:
        return True
    
    def is_const_wrt(self, varidx):
        return True
    
    def validate(self) -> bool:
        res = not np.isnan(self.val)
        return res
    
    def accept(self, visitor):
        visitor.visitConstant(self)
    
    def to_sympy(self, dps:int=None):
        return sympy.Float(self.val, dps=dps)

    def match(self, trunk) -> bool:
        if type(trunk) is UnknownSyntaxTree: return True
        if type(self) is not type(trunk): return False
        if self.val != trunk.val: return False
        return True
    
    def scale(self, l):
        self.val *= l
        return self
    
    def is_scalable(self, l) -> bool:
        return True