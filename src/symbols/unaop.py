import numpy as np
import sympy
from symbols.syntax_tree import SyntaxTree
from symbols.const import ConstantSyntaxTree
from symbols import simplifier
from backprop.bperrors import KnowBackpropError


class UnaryOperatorSyntaxTree(SyntaxTree):
    OPERATORS = ['exp', 'log', 'sqrt', 'square', 'cube']
    INVERTIBLE_OPERATORS = ['exp', 'log', 'sqrt', 'cube']

    def __init__(self, operator:str, inner:SyntaxTree):
        super().__init__()
        self.operator = operator
        self.inner = inner
    
    def clone(self):
        c = UnaryOperatorSyntaxTree(self.operator, self.inner.clone())
        c.copy_output_from(self)
        return c
    
    def __call__(self, x):
        if self.output is None:
            self.output = self.operate(self.inner(x))
        return self.output
    
    def __getitem__(self, x_d):
        x, d = x_d
        if d not in self.y_know:
            if d == ():
                self.y_know[d] = self.operate(self.inner[x_d])
            elif len(d) == 1:
                self.y_know[d] = self.operate_deriv(self.inner[(x,())], self.inner[x_d])
            else:
                raise RuntimeError(f"Derivative {d} not supported.")
        return self.y_know[d]
    
    def at(self, x):
        return self.operate(self.inner.at(x))
    
    def clear_output(self):
        super().clear_output()
        self.inner.clear_output()
    
    def set_parent(self, parent=None):
        super().set_parent(parent)
        self.inner.set_parent(self)
    
    def validate(self) -> bool:
        return self.inner.validate()

    def simplify(self):
        self.inner = self.inner.simplify()
        return simplifier.simplify_unary_stree(self)
    
    def operate(self, inner:np.array) -> np.array:
        if self.operator == 'exp'   : return np.exp (inner)
        if self.operator == 'log'   : return np.log (inner)
        if self.operator == 'sqrt'  : return np.sqrt(inner)
        if self.operator == 'square': return inner ** 2
        if self.operator == 'cube'  : return inner ** 3
        raise RuntimeError(f"Operation not defined for operator {self.operator}.")
    
    def operate_inv(self, output:np.array) -> np.array:
        if self.operator == 'exp'   : return np.log(output)
        if self.operator == 'log'   : return np.exp(output)
        if self.operator == 'sqrt'  : return output ** 2
        if self.operator == 'square': return np.sqrt(output)
        if self.operator == 'cube'  : return np.cbrt(output)
        
        raise RuntimeError(f"Inverse operation not defined for operator {self.operator}.")
    
    def operate_deriv(self, inner:np.array, inner_deriv:np.array) -> np.array:
        if self.operator == 'exp'   : return np.exp(inner) * inner_deriv
        if self.operator == 'log'   : return inner_deriv / inner
        if self.operator == 'sqrt'  : return inner_deriv / (2.0 * np.sqrt(inner))
        if self.operator == 'square': return 2.0 * inner * inner_deriv
        if self.operator == 'cube'  : return 3.0 * (inner**2) * inner_deriv
        raise RuntimeError(f"Derivative not defined for operator {self.operator}.")
    
    def __str__(self) -> str:
        return f"{self.operator}({str(self.inner)})"
    
    def __eq__(self, other) -> bool:
        if type(other) is not UnaryOperatorSyntaxTree: return False
        return \
            self.operator == other.operator and \
            self.inner == other.inner
    
    """
    def diff(self, varidx:int=0) -> SyntaxTree:
        if self.is_const_wrt(varidx):
            return ConstantSyntaxTree(0)
        
        g = self.inner

        if self.operator == 'exp':
            return BinaryOperatorSyntaxTree( '*',
                UnaryOperatorSyntaxTree('exp', g.clone() ),
                g.diff(varidx)
                )
        
        if self.operator == 'log':
            return BinaryOperatorSyntaxTree( '/',
                g.diff(varidx),
                g.clone()
            )
        
        if self.operator == 'sqrt':
            return BinaryOperatorSyntaxTree( '/',
                g.diff(varidx),
                BinaryOperatorSyntaxTree( '*',
                    ConstantSyntaxTree(2),
                    UnaryOperatorSyntaxTree('sqrt', g.clone())
                )
            )
        
        if self.operator == 'square':
            return BinaryOperatorSyntaxTree( '*',
                BinaryOperatorSyntaxTree( '*',
                    ConstantSyntaxTree(2),
                    g.clone()
                ),
                g.diff(varidx)
            )

        if self.operator == 'cube':
            return BinaryOperatorSyntaxTree( '*',
                BinaryOperatorSyntaxTree( '*',
                    ConstantSyntaxTree(3),
                    UnaryOperatorSyntaxTree( 'square', g.clone() )
                ),
                g.diff(varidx)
            )
        
        raise RuntimeError(f"Differentiation not defined for operator {self.operator}.")
    """
    
    def is_const(self) -> bool:
        return self.inner.is_const()
    
    def is_const_wrt(self, varidx) -> bool:
        return self.inner.is_const_wrt(varidx)
    
    def pull_output(self, target_output:np.array, child=None, flatten:bool=False) -> np.array:
        pulled_output = super().pull_output(target_output, child, flatten)
        if child is None or pulled_output is None: return pulled_output
        if id(child) == id(self.inner):
            pulled_output = self.operate_inv(pulled_output)
            if flatten: utils.flatten(pulled_output)
            return pulled_output
        raise RuntimeError('Invalid child.')
    
    def pull_know(self, k_target:np.array, noroot_target:bool=False, symm_target:tuple[bool,np.array]=None, child=None, track:dict={}) -> tuple[np.array,bool,tuple[bool,np.array]]:
        
        k_pulled, noroot_pulled, symm_pulled = super().pull_know(k_target, noroot_target, symm_target, track=track)
        if child is None:
            return k_pulled, noroot_pulled, symm_pulled
        
        if id(child) != id(self.inner):
            raise RuntimeError('Invalid child.')
        
        k_target = k_pulled
        noroot_target = noroot_pulled
        symm_target = symm_pulled
        
        k_pulled = np.full(k_target.shape, np.nan)
        noroot_pulled = False
        # symmetry (w.r.t. variables) backprop.
        if symm_target is not None and self.operator == 'square':
            symm_pulled = (None, symm_target[1])

        if self.operator == 'square':
            if (k_target < 0.0).any():
                raise KnowBackpropError()
            if noroot_target:
                noroot_pulled = True
        
        elif self.operator == 'cube':
            k_pulled[:] = k_target
            if noroot_target:
                noroot_pulled = True
        
        elif self.operator == 'sqrt':
            if (k_target < 0.0).any():
                raise KnowBackpropError()
            k_pulled[:] = +1.0
            if noroot_target:
                noroot_pulled = True
        
        elif self.operator == 'exp':
            if (k_target < 0.0).any():
                raise KnowBackpropError()
        
        elif self.operator == 'log':
            k_pulled[:] = +1.0
            noroot_pulled = True
        
        else:
            raise RuntimeError('Invalid operator.')
        
        track[id(self.inner)] = (k_pulled, noroot_pulled, symm_pulled)
        return k_pulled, noroot_pulled, symm_pulled
    
    def pull_know_deriv(self, image_track:dict, derividx:int, k_target:np.array, child=None) -> np.array:
        
        k_pulled = super().pull_know_deriv(image_track, derividx, k_target)
        if child is None:
            return k_pulled
        
        if id(child) != id(self.inner):
            raise RuntimeError('Invalid child.')
        
        k_inner = image_track[id(self.inner)][0]
        k_inner_isknown = ~np.isnan(k_inner)
        
        k_target = k_pulled
        k_pulled = np.full(k_target.shape, np.nan)

        if self.operator == 'square':
            mask = k_inner_isknown & (k_target > 0.0)
            k_pulled[mask] = k_inner[mask]

            mask = k_inner_isknown & (k_target < 0.0)
            k_pulled[mask] = -k_inner[mask]
        
        elif self.operator == 'cube':
            k_pulled[:] = k_target[:]
        
        elif self.operator == 'sqrt':
            if (k_inner == 0.0).any():
                raise KnowBackpropError()
            k_pulled[:] = k_target[:]
        
        elif self.operator == 'exp':
            k_pulled[:] = k_target[:]
        
        elif self.operator == 'log':
            k_pulled[:] = k_target[:]
        
        else:
            raise RuntimeError('Invalid operator.')
        
        return k_pulled
    
    def get_coeffs(self, coeffs:list):
        self.inner.get_coeffs(coeffs)
    
    def set_coeffs(self, coeffs:list, start:int=0):
        self.inner.set_coeffs(coeffs, start)
    
    def get_unknown_stree(self, unknown_stree_label:str):
        return self.inner.get_unknown_stree(unknown_stree_label)

    def set_unknown_model(self, model_label:str, model, coeffs_mask:list[float]=None, constrs:dict=None):
        self.inner.set_unknown_model(model_label, model, coeffs_mask, constrs)
    
    def set_all_unknown_models(self, model):
        self.inner.set_all_unknown_models(model)
    
    def count_unknown_model(self, model_label:str) -> int:
        return self.inner.count_unknown_model(model_label)
    
    def accept(self, visitor):
        visitor.visitUnaryOperator(self)
        self.inner.accept(visitor)
    
    def to_sympy(self, dps:int=None):
        inner_sympy = self.inner.to_sympy(dps)
        if   self.operator == 'exp'   : return sympy.exp (inner_sympy)
        elif self.operator == 'log'   : return sympy.log (inner_sympy)
        elif self.operator == 'sqrt'  : return sympy.sqrt(inner_sympy)
        elif self.operator == 'square': return inner_sympy ** 2
        elif self.operator == 'cube'  : return inner_sympy ** 3
        raise RuntimeError(f"Conversion to sympy not defined for operator {self.operator}.")
    
    def get_max_depth(self) -> int:
        #if self.cache[SyntaxTree.CACHE_MAX_DEPTH] is None: 
        #    self.cache[SyntaxTree.CACHE_MAX_DEPTH] = 1 + self.inner.get_max_depth()
        #return self.cache[SyntaxTree.CACHE_MAX_DEPTH]
        return 1 + self.inner.get_max_depth()
    
    def get_nnodes(self) -> int:
        #if self.cache[SyntaxTree.CACHE_NNODES] is None: 
        #    self.cache[SyntaxTree.CACHE_NNODES] = 1 + self.inner.get_nnodes()
        #return self.cache[SyntaxTree.CACHE_NNODES]
        return 1 + self.inner.get_nnodes()
    
    def get_nodes(self, nodes:list):
        super().get_nodes(nodes)
        self.inner.get_nodes(nodes)
    
    def match(self, trunk) -> bool:
        if type(trunk) is UnknownSyntaxTree: return True
        if type(self) is not type(trunk): return False
        if self.operator != trunk.operator: return False
        return self.inner.match(trunk.inner)
    
    def is_invertible(self) -> bool:
        return self.operator in UnaryOperatorSyntaxTree.INVERTIBLE_OPERATORS
    
    def scale(self, l):
        if self.operator == 'sqrt' and l >= 0 and self.inner.is_scalable(l):
            self.inner.scale(l ** 2)
            return self
        return super().scale(l)
    
    def is_scalable(self, l) -> bool:
        return self.operator == 'sqrt' and l >= 0 and self.inner.is_scalable(l)