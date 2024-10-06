import sympy
from symbols.syntax_tree import SyntaxTree
from backprop import utils


class FunctionSyntaxTree(SyntaxTree):
    def __init__(self, f):
        super().__init__()
        self.f = f
    
    def clone(self):
        c = FunctionSyntaxTree(self.f.clone())
        c.copy_output_from(self)
        return c
    
    def __call__(self, x):
        if self.output is None:
            self.output = self.f(x)
        return self.output
    
    def __getitem__(self, x_d):
        x, d = x_d
        if d not in self.y_know:
            self.y_know[d] = self.f(x)
        return self.y_know[d]
    
    def at(self, x):
        return self.f(x)
    
    def __str__(self) -> str:
        return 'f(X)'
    
    def __eq__(self, other) -> bool:
        if type(other) is not FunctionSyntaxTree: return False
        return self.f == other.f
    
    def diff(self, varidx:int=0) -> SyntaxTree:
        return FunctionSyntaxTree(self.f.get_deriv((varidx,)).clone())
    
    def is_const(self) -> bool:  # TODO
        return False
    
    def is_const_wrt(self, varidx):  # TODO
        return False
    
    def accept(self, visitor):
        visitor.visitFunction(self)
    
    def to_sympy(self, dps:int=None):
        return self.f.to_sympy()
    
    def match(self, trunk) -> bool:
        return False


class SemanticSyntaxTree(SyntaxTree):
    def __init__(self, sem):
        super().__init__()
        self.sem = sem
    
    def clone(self):
        c = SemanticSyntaxTree(self.sem)
        c.copy_output_from(self)
        return c
    
    def __call__(self, x):
        if self.output is None:
            self.output = self.sem
        return self.output
    
    def __getitem__(self, x):
        raise NotImplementedError()
    
    def __str__(self) -> str:
        return f"{self.sem}"#'SST(X)'
    
    def __eq__(self, other) -> bool:
        if id(self) == id(other): return True
        if type(other) is not SemanticSyntaxTree: return False
        return (self.sem == other.sem).all()
    
    def at(self, x):
        return self.sem
    
    def diff(self, varidx:int=0) -> SyntaxTree:
        raise NotImplementedError()
    
    def is_const(self) -> bool:
        return False
    
    def is_const_wrt(self, varidx):
        return False
    
    def accept(self, visitor):
        visitor.visitSemantic(self)
    
    def to_sympy(self, dps:int=None):
        raise NotImplementedError()
    
    def match(self, trunk) -> bool:
        return False
        

class UnknownSyntaxTree(SyntaxTree):
    def __init__(self, name:str='A', deriv:tuple[int]=(), nvars:int=1, model=None, coeffs_mask=None, constrs=None):
        assert nvars > 0
        super().__init__()
        self.name = name
        self.deriv = deriv
        self.nvars = nvars
        self.model = model
        self.coeffs_mask = coeffs_mask  # TODO: remove it?! (no more used).
        self.constrs = constrs
        self.label = f"{utils.deriv_to_string(self.deriv)}{self.name}"
        self.knowledge = None
    
    def clone(self):
        c = UnknownSyntaxTree(self.name, self.deriv, self.nvars, model=self.model)  # TODO: clone model
        c.copy_output_from(self)
        return c
    
    def __call__(self, x):
        if self.output is None:
            if self.model is None:
                raise RuntimeError('None unknown model.')
            self.output = self.model(x)
        return self.output
    
    def __getitem__(self, x_d):
        x, d = x_d
        if d not in self.y_know:
            if self.model is None:
                raise RuntimeError('None unknown model.')
            self.y_know[d] = self.model.get_deriv(d)(x)
        return self.y_know[d]
    
    def __str__(self) -> str:
        xs = ''
        for i in range(self.nvars): xs += f"x{i},"
        return f"{self.label}({xs[:-1]})"
    
    def __eq__(self, other) -> bool:
        if type(other) is not UnknownSyntaxTree: return False
        return self.name == other.name and self.deriv == other.deriv
    
    def at(self, x):
        if self.model is None:
            raise RuntimeError('None unknown model.')
        return self.model(x) 
    
    def diff(self, varidx:int=0) -> SyntaxTree:
        assert varidx < self.nvars
        return UnknownSyntaxTree(name=self.name, deriv=self.deriv+(varidx,), nvars=self.nvars)
    
    def is_const(self) -> bool:
        if self.model is None: return False
        return False  # TODO: #self.model.is_const()
    
    def is_const_wrt(self, varidx):
        if self.model is None: return False
        return False  # TODO: #self.model.is_const_wrt(varidx)
    
    def get_unknown_stree(self, unknown_stree_label:str):
        if self.label == unknown_stree_label: return self
        return None

    def set_unknown_model(self, model_label:str, model, coeffs_mask:list[float]=None, constrs:dict=None):
        if self.label == model_label:
            self.model = model
            self.coeffs_mask = coeffs_mask
            self.constrs = constrs
    
    def set_all_unknown_models(self, model):
        self.model = model
    
    def count_unknown_model(self, model_label:str) -> int:
        return 1 if self.label == model_label else 0
    
    def accept(self, visitor):
        visitor.visitUnknown(self)
    
    def to_sympy(self, dps:int=None):
        if self.model is None:
            xs = [sympy.Symbol('x')] if self.nvars == 1 else [sympy.Symbol(f"x{i}") for i in range(self.nvars)]
            return sympy.Function(self.label)(*xs)
        return self.model.to_sympy(dps)
    
    def get_max_depth(self) -> int:
        return 0  # TODO: is it ok? or consider self.model?
    
    def match(self, trunk) -> bool:
        return True