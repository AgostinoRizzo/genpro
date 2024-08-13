import numpy as np
import sympy
import string
import random
import re

import numlims
from backprop import utils


class PropositionalConstraint:
    def __init__(self, f, g, opt:str='>', lb:float=-numlims.INFTY, ub:float=numlims.INFTY):
        self.f = f
        self.g = g
        self.opt = opt
        self.lb = lb
        self.ub = ub
    
    def __str__(self) -> str:
        lb_str = numlims.tostr(self.lb)
        ub_str = numlims.tostr(self.ub)
        return f"{self.f}{self.opt}{self.g} [{lb_str},{ub_str}]"
    
    def is_equivalent(self, other) -> bool:
        return \
            self.f == other.f and self.g == other.g and self.opt == other.opt and \
            self.lb == other.lb and self.ub == other.ub
        
    def is_dominated_by(self, other) -> bool:
        return \
            self.f == other.f and self.g == other.g and self.opt == other.opt and \
            self.lb >= other.lb and self.ub <= other.ub
    
    def fights(self, other) -> bool:
        if self.lb >= other.ub or self.ub <= other.lb: return False  # not interval overlap
        if not (self.f == other.f and self.g == other.g): return False
        if (self.opt == '>' and other.opt == '<') or \
           (self.opt == '<' and other.opt == '>'): return True
        return False
    
    def simplify(self):
        if type(self.f) is ConstantSyntaxTree and type(self.g) is ConstantSyntaxTree:
            if self.opt == '>': return None if self.f.val > self.g.val else False
            if self.opt == '<': return None if self.f.val < self.g.val else False
            if self.opt == '%': return None if self.f.val % 2 == self.g.val else False
            raise RuntimeError(f"Constant simplification not defined for {self.opt}")
        return self


class SymbolMapper:
    def __init__(self):
        self.symb_to_prop = {}
        self.prop_to_symb = {}
        self.symb_idx = 0
    
    def map_as_symbol(self, prop:PropositionalConstraint):
        prop = prop.simplify()
        if prop is None: return True
        if not prop: return False
        for prop_k in self.prop_to_symb.keys():
            if prop.is_equivalent( prop_k ): return self.prop_to_symb[prop_k]

        symb = sympy.Symbol(f"X{self.symb_idx}")
        self.symb_idx += 1
        self.symb_to_prop[symb] = prop
        self.prop_to_symb[prop] = symb
        return symb


class Relopt:  # immutable class
    def __init__(self, opt:str='='):
        self.opt = opt
    def neg(self):
        neg_opt = Relopt(self.opt)
        if self.opt == '>=': neg_opt.opt = '<='
        if self.opt == '<=': neg_opt.opt = '>='
        if self.opt == '>' : neg_opt.opt = '<'
        if self.opt == '<' : neg_opt.opt = '>'
        return neg_opt
    def strict(self):
        strict_opt = Relopt(self.opt)
        if self.opt == '>=': strict_opt.opt = '>'
        if self.opt == '<=': strict_opt.opt = '<'
        return strict_opt
    def __eq__(self, other) -> bool:
        return self.opt == other.opt
    def check(self, a:float, b:float) -> bool:
        if self.opt == '=' : return a == b
        if self.opt == '>=': return a >= b
        if self.opt == '<=': return a <= b
        if self.opt == '>' : return a >  b
        if self.opt == '<' : return a <  b
        raise RuntimeError(f"Operator {self.opt} not supported.")


class PullError(RuntimeError):
    pass

class PullViolation(RuntimeError):
    pass


class SyntaxTree:
    CACHE_MAX_DEPTH = 0
    CACHE_NNODES = 1

    def __init__(self):
        self.parent = None
        self.output = None
        #self.cache = [None] * 2
        self.parents = None

        self.sat = True
        self.sat_y = True
        self.match_r2 = 1.0
        self.best_match_r2 = 1.0
    
    def clone(self): return None
    def compute_output(self, x): return None
    def __call__(self, x): return self.compute_output(x)
    def set_parent(self, parent=None): self.parent = parent
    def validate(self) -> bool: return True
    def simplify(self): return self
    def __str__(self) -> str: return ''
    def __eq__(self, other) -> bool: return False
    def diff(self, varidx:int=0): return None
    def is_const(self) -> bool: return False
    def is_const_wrt(self, varidx) -> bool: return False
    def backprop_sign(self, symbmapper:SymbolMapper, lb:float=-numlims.INFTY, ub:float=numlims.INFTY, sign:str='+'): pass

    def pull_output(self, target_output:np.array, relopt:Relopt=Relopt('='), child=None, flatten:bool=False) -> tuple[np.array, Relopt]:
        if self.parent is None:
            return ((utils.flatten(target_output) if flatten else target_output), relopt)
        return self.parent.pull_output(target_output, relopt, self, flatten)
    
    def get_unknown_stree(self, unknown_stree_label:str): return None
    def set_unknown_model(self, model_label:str, model, coeffs_mask:list[float]=None, constrs:dict=None): pass
    def set_all_unknown_models(self, model): pass
    def count_unknown_model(self, model_label:str) -> int: return 0
    def accept(self, visitor): pass
    def to_sympy(self, dps:int=None): pass
    def get_max_depth(self) -> int: return 0
    def get_nnodes(self) -> int: return 1
    def match(self, trunk) -> bool: return False
    def is_linear(self) -> bool: return False
    def is_invertible(self) -> bool: return False
    def scale(self, l): return BinaryOperatorSyntaxTree('*', ConstantSyntaxTree(l), self)
    def is_scalable(self, l) -> bool: return False

    def get_depth(self) -> int:
        depth = 0
        p = self.parent
        while p is not None:
            depth += 1
            p = p.parent
        return depth
    
    def is_subtree(self, other) -> bool:
        if id(self) == id(other): return True
        if self.parent is None: return False
        return self.parent.is_subtree(other)

    def clear_cache(self):
        #for i in range(len(self.cache)): self.cache[i] = None
        pass
    
    @staticmethod
    def is_invertible_path(node) -> bool:
        p = node.parent
        while p is not None:
            if not p.is_invertible(): return False
            p = p.parent
        return True
    
    @staticmethod
    def diff(stree, deriv:tuple[int]):
        stree_deriv = stree
        for varidx in deriv:
            stree_deriv = stree_deriv.diff(varidx).simplify()
        return stree_deriv
    
    @staticmethod
    def diff_all(stree, derivs:list[tuple[int]], include_zeroth:bool=True) -> dict:
        derivs_map = {(): stree.simplify()}
        for deriv in sorted(derivs):
            if len(deriv) == 0: continue
            derivs_map[deriv] = derivs_map[deriv[:-1]].diff(deriv[-1]).simplify()
        if not include_zeroth:
            del derivs_map[()]
        return derivs_map


class BinaryOperatorSyntaxTree(SyntaxTree):
    OPERATORS = ['+', '-', '*', '/'] #, '^']
    INVERTIBLE_OPERATORS = ['+', '-', '*', '/']

    def __init__(self, operator:str, left:SyntaxTree, right:SyntaxTree):
        super().__init__()
        self.operator = operator
        self.left = left
        self.right = right
    
    def clone(self):
        return BinaryOperatorSyntaxTree(self.operator, self.left.clone(), self.right.clone())
    
    def compute_output(self, x):
        self.output  = None
        left_output  = self.left.compute_output(x)
        right_output = self.right.compute_output(x)
        if left_output is None or right_output is None: return None
        self.output = self.__operate(left_output, right_output)
        return self.output
    
    def set_parent(self, parent=None):
        super().set_parent(parent)
        self.left.set_parent(self)
        self.right.set_parent(self)
    
    def validate(self) -> bool:
        if self.operator == '^' and (type(self.right) is not ConstantSyntaxTree or self.right.val not in [2.0,3.0,4.0]):  # TODO: only ^2 managed.
            return False
        
        if self.operator == '/' and type(self.right) is ConstantSyntaxTree and self.right.val == 0.0:
            return False

        return self.left.validate() and self.right.validate()

    def simplify(self):
        self.left = self.left.simplify()
        self.right = self.right.simplify()
        is_left_const  = type(self.left)  is ConstantSyntaxTree
        is_right_const = type(self.right) is ConstantSyntaxTree

        if is_left_const and is_right_const and (self.operator != '/' or self.right.val != 0.):
            return ConstantSyntaxTree( self.__operate(self.left.val, self.right.val) )
        
        if self.operator == '*':
            if (is_left_const and self.left.val == 0) or (is_right_const and self.right.val == 0):
                return ConstantSyntaxTree(0.0)
            if (is_left_const and self.left.val == 1):
                return self.right
            if (is_right_const and self.right.val == 1):
                return self.left
        
        if self.operator == '/':
            if self.left == self.right: return ConstantSyntaxTree(1.0)
            if is_left_const and self.left.val == 0: return ConstantSyntaxTree(0.0)
        
        if self.operator == '+':
            if is_left_const  and self.left.val  == 0: return self.right
            if is_right_const and self.right.val == 0: return self.left
        if self.operator == '-':
            if is_right_const and self.right.val == 0: return self.left
            if self.left == self.right: return ConstantSyntaxTree(0.0)
        
        if self.operator == '^' and is_right_const:
            if self.right.val == 0:
                return ConstantSyntaxTree(1.0)
            
            if self.right.val == 1:
                return self.left
            
            if self.right.val == 2 and type(self.left) is UnaryOperatorSyntaxTree and self.left.operator == 'sqrt':
                return self.left.inner
            
            if type(self.right) is ConstantSyntaxTree:
                if self.right.val == 2: return UnaryOperatorSyntaxTree('square', self.left)
                if self.right.val == 3: return UnaryOperatorSyntaxTree('cube', self.left)
            
            if type(self.left) is BinaryOperatorSyntaxTree and self.left.operator == '^' and \
               type(self.left.right) is ConstantSyntaxTree:
                self.right = ConstantSyntaxTree(self.right.val * self.left.right.val)
                self.left = self.left.left
                return self
        
        return self
    
    def __operate(self, left:np.array, right:np.array) -> np.array:
        if   self.operator == '/': return left / right
        elif self.operator == '*': return left * right
        elif self.operator == '+': return left + right
        elif self.operator == '-': return left - right
        elif self.operator == '^': return left ** right
        raise RuntimeError(f"Operation not defined for operator {self.operator}.")
    
    def __operate_inv(self, output:np.array, output_relopt:Relopt, get_left:bool=True) -> tuple[np.array, Relopt]:
        if   self.operator == '/':
            self.right.output[self.right.output == 0.0] = np.nan
            
            # get the numerator.
            if get_left:
                return  output * self.right.output, \
                       (output_relopt if np.all(self.right.output >= 0) else output_relopt.neg())  # TODO: check np.all
            
            # get the denominator.
            if output_relopt.opt == '=':
                return self.left.output / output, \
                       output_relopt
            
            #raise RuntimeError(f"Inverse for {self.left.output}/x{output_relopt.opt}{output} not defined.")
            return np.full(output.shape, np.nan), \
                   output_relopt

            """if output != 0. or output_relopt.opt == '=':
                if output_relopt.opt == '=': raise PullViolation()
                raise RuntimeError(f"Inverse for {self.left.output}/x{output_relopt.opt}{output} not defined.")
            if self.left.output >= 0.: return 0., output_relopt.strict()
            return 0., output_relopt.neg().strict()"""
            #if output == 0.: return numlims.INFTY, Relopt('=')
        
        elif self.operator == '*':
            num = output
            den = self.right.output if get_left else self.left.output
            return  num / den, \
                   (output_relopt if np.all(den > 0) else output_relopt.neg())  # TODO: check np.all

        elif self.operator == '+': return ((output - self.right.output) if get_left else (output - self.left.output)), output_relopt
        elif self.operator == '-': return ((output + self.right.output) if get_left else (self.left.output - output)), output_relopt
        elif self.operator == '^':
            if get_left:
                if np.all(self.right.output == 2.): return np.sqrt(output), output_relopt
                if np.all(self.right.output == 3.): return np.cbrt(output), output_relopt
                raise RuntimeError(f"Inverse power not defined for exponent {self.right.output}")
            else:
                np.log(output, self.left.output), output_relopt            
        
        raise RuntimeError(f"Inverse operation not defined for operator {self.operator}.")
    
    def __str__(self) -> str:
        return f"({str(self.left)} {self.operator} {str(self.right)})"
    
    def __eq__(self, other) -> bool:
        if type(other) is not BinaryOperatorSyntaxTree: return False
        return \
            self.operator == other.operator and \
            self.left == other.left and self.right == other.right
    
    def diff(self, varidx:int=0) -> SyntaxTree:
        if self.is_const_wrt(varidx):
            return ConstantSyntaxTree(0)
        
        f = self.left
        g = self.right

        if self.operator == '/':
            return BinaryOperatorSyntaxTree( '/',
                BinaryOperatorSyntaxTree('-',
                    BinaryOperatorSyntaxTree('*', f.diff(varidx), g.clone()),
                    BinaryOperatorSyntaxTree('*', f.clone(), g.diff(varidx))
                ),
                UnaryOperatorSyntaxTree('square', g.clone())
            )
        
        if self.operator == '*':
            return BinaryOperatorSyntaxTree( '+',
                BinaryOperatorSyntaxTree('*', f.diff(varidx), g.clone()),
                BinaryOperatorSyntaxTree('*', f.clone(), g.diff(varidx))
            )
        
        elif self.operator == '+' or self.operator == '-':
            return BinaryOperatorSyntaxTree( self.operator, f.diff(varidx), g.diff(varidx) )
        
        elif self.operator == '^':
            if type(g) is not ConstantSyntaxTree:
                raise RuntimeError(f"Differentiation not defined for operator {self.operator} and non-constant exponent.{g}")
            return BinaryOperatorSyntaxTree( '*',
                BinaryOperatorSyntaxTree( '*',
                    g.clone(),
                    BinaryOperatorSyntaxTree( '^',
                        f.clone(),
                        BinaryOperatorSyntaxTree( '-',
                            g.clone(),
                            ConstantSyntaxTree(1)
                        )
                    )
                ),
                f.diff(varidx)
            )
        
        raise RuntimeError(f"Differentiation not defined for operator {self.operator}.")
    
    def is_const(self) -> bool:
        return self.left.is_const() and self.right.is_const()
    
    def is_const_wrt(self, varidx) -> bool:
        return self.left.is_const_wrt(varidx) and self.right.is_const_wrt(varidx)
    
    def backprop_sign(self, symbmapper:SymbolMapper, lb:float=-numlims.INFTY, ub:float=numlims.INFTY, sign:str='+'):
        f = self.left
        g = self.right
        f_pos = f.backprop_sign(symbmapper, lb, ub, '+')
        f_neg = f.backprop_sign(symbmapper, lb, ub, '-')
        g_pos = g.backprop_sign(symbmapper, lb, ub, '+')
        g_neg = g.backprop_sign(symbmapper, lb, ub, '-')
        
        if self.operator == '*' or self.operator == '/':
            return (f_pos & g_pos) | (f_neg & g_neg) if sign == '+' else \
                   (f_pos & g_neg) | (f_neg & g_pos)
        
        elif self.operator == '+':
            #return f_pos & g_pos if sign == '+' else f_neg & g_neg
            return ~(f_neg & g_neg) if sign == '+' else ~(f_pos & g_pos)
        
        elif self.operator == '-':
            """f_greater_g_constr = PropositionalConstraint(f, g, '>', lb, ub)
            f_lower_g_constr   = PropositionalConstraint(f, g, '<', lb, ub)
            return (
                symbmapper.map_as_symbol(f_greater_g_constr) if sign == '+' else
                symbmapper.map_as_symbol(f_lower_g_constr)
            )"""
            #return f_pos & g_neg if sign == '+' else f_neg & g_pos
            return ~(f_neg & g_pos) if sign == '+' else ~(f_pos & g_neg)
        
        elif self.operator == '^':
            g_even_constr = PropositionalConstraint(g, ConstantSyntaxTree(0), '%', lb, ub)
            g_odd_constr  = PropositionalConstraint(g, ConstantSyntaxTree(1), '%', lb, ub)
            g_even_symb = symbmapper.map_as_symbol(g_even_constr)
            g_odd_symb  = symbmapper.map_as_symbol(g_odd_constr)
            return (
                g_even_symb | (g_odd_symb & f_pos) if sign == '+' else
                g_odd_symb & f_neg
            )

        raise RuntimeError(f"Backprop not defined for operator {self.operator}.")

    def pull_output(self, target_output:np.array, relopt:Relopt=Relopt('='), child=None, flatten:bool=False) -> tuple[np.array, Relopt]:
        pulled_output, pulled_relopt = super().pull_output(target_output, relopt, child, flatten)
        if child is None or pulled_output is None:
            return pulled_output, pulled_relopt
        if   id(child) == id(self.left):
            pulled_output, pulled_relopt = self.__operate_inv(pulled_output, pulled_relopt, get_left=True)
        elif id(child) == id(self.right):
            pulled_output, pulled_relopt = self.__operate_inv(pulled_output, pulled_relopt, get_left=False)
        else:
            raise RuntimeError('Invalid child.')
        if flatten:
            pulled_output = utils.flatten(pulled_output)
        return pulled_output, pulled_relopt
    
    def get_unknown_stree(self, unknown_stree_label:str):
        stree = self.left.get_unknown_stree(unknown_stree_label)
        if stree is not None: return stree
        return self.right.get_unknown_stree(unknown_stree_label)

    def set_unknown_model(self, model_label:str, model, coeffs_mask:list[float]=None, constrs:dict=None):
        self.left .set_unknown_model(model_label, model, coeffs_mask, constrs)
        self.right.set_unknown_model(model_label, model, coeffs_mask, constrs)
    
    def set_all_unknown_models(self, model):
        self.left .set_all_unknown_models(model_label, model)
        self.right.set_all_unknown_models(model_label, model)
    
    def count_unknown_model(self, model_label:str) -> int:
        return self.left.count_unknown_model(model_label) + \
               self.right.count_unknown_model(model_label)
    
    def accept(self, visitor):
        visitor.visitBinaryOperator(self)
        self.left.accept(visitor)
        self.right.accept(visitor)
    
    def to_sympy(self, dps:int=None):
        left_sympy  = self.left.to_sympy(dps)
        right_sympy = self.right.to_sympy(dps)
        if self.operator == '+': return left_sympy +  right_sympy
        if self.operator == '-': return left_sympy -  right_sympy
        if self.operator == '*': return left_sympy *  right_sympy
        if self.operator == '/': return left_sympy /  right_sympy
        if self.operator == '^': return left_sympy ** right_sympy
        raise RuntimeError(f"Conversion to sympy not defined for operator {self.operator}.")
    
    def get_max_depth(self) -> int:
        #if self.cache[SyntaxTree.CACHE_MAX_DEPTH] is None: 
        #    self.cache[SyntaxTree.CACHE_MAX_DEPTH] = 1 + max(self.left.get_max_depth(), self.right.get_max_depth())
        #return self.cache[SyntaxTree.CACHE_MAX_DEPTH]
        return 1 + max(self.left.get_max_depth(), self.right.get_max_depth())
    
    def get_nnodes(self) -> int:
        #if self.cache[SyntaxTree.CACHE_NNODES] is None: 
        #    self.cache[SyntaxTree.CACHE_NNODES] = 1 + self.left.get_nnodes() + self.right.get_nnodes()
        #return self.cache[SyntaxTree.CACHE_NNODES]
        return 1 + self.left.get_nnodes() + self.right.get_nnodes()
    
    def match(self, trunk) -> bool:
        if type(trunk) is UnknownSyntaxTree: return True
        if type(self) is not type(trunk): return False
        if self.operator != trunk.operator: return False
        return self.left.match(trunk.left) and self.right.match(trunk.right)
    
    def is_invertible(self) -> bool:
        return self.operator in BinaryOperatorSyntaxTree.INVERTIBLE_OPERATORS
    
    def scale(self, l):
        if   self.operator == '/': self.left = self.left.scale(l)
        
        elif self.operator == '*':
            if   self.left.is_scalable(l): self.left = self.left.scale(l)
            elif self.right.is_scalable(l): self.right = self.right.scale(l)
            elif self.left.get_nnodes() < self.right.get_nnodes(): self.left = self.left.scale(l)
            else: self.right = self.right.scale(l)
        
        elif self.operator == '+':
            self.left = self.left.scale(l)
            self.right = self.right.scale(l)
        
        elif self.operator == '-':
            self.left = self.left.scale(l)
            self.right = self.right.scale(l)
        
        elif self.operator == '^': raise NotImplementedError()

        return self
    
    def is_scalable(self, l) -> bool:
        if   self.operator == '/': return self.left.is_scalable(l)
        elif self.operator == '*': return self.left.is_scalable(l) or self.right.is_scalable(l)
        elif self.operator == '+': return self.left.is_scalable(l) and self.right.is_scalable(l)
        elif self.operator == '-': return self.left.is_scalable(l) and self.right.is_scalable(l)
        elif self.operator == '^': raise NotImplementedError()
        return False


class UnaryOperatorSyntaxTree(SyntaxTree):
    OPERATORS = ['exp', 'log', 'sqrt', 'square', 'cube']
    INVERTIBLE_OPERATORS = ['exp', 'log', 'sqrt', 'cube']

    def __init__(self, operator:str, inner:SyntaxTree):
        super().__init__()
        self.operator = operator
        self.inner = inner
    
    def clone(self):
        return UnaryOperatorSyntaxTree(self.operator, self.inner.clone())
    
    def compute_output(self, x):
        self.output  = None
        inner_output  = self.inner.compute_output(x)
        if inner_output is None: return None
        self.output = self.__operate(inner_output)
        return self.output
    
    def set_parent(self, parent=None):
        super().set_parent(parent)
        self.inner.set_parent(self)
    
    def validate(self) -> bool:
        return self.inner.validate()

    def simplify(self):
        self.inner = self.inner.simplify()

        if type(self.inner) is ConstantSyntaxTree:
            return ConstantSyntaxTree( self.__operate(self.inner.val) )
        
        if self.operator == 'exp' and type(self.inner) is UnaryOperatorSyntaxTree and self.inner.operator == 'log':
            return self.inner.inner
        
        if self.operator == 'log' and type(self.inner) is UnaryOperatorSyntaxTree and self.inner.operator == 'exp':
            return self.inner.inner
        
        if self.operator == 'sqrt' and type(self.inner) is BinaryOperatorSyntaxTree and \
           self.inner.operator == '^' and type(self.inner.right) is ConstantSyntaxTree and self.inner.right.val == 2:
            return self.inner.left
        
        if self.operator == 'sqrt' and type(self.inner) is UnaryOperatorSyntaxTree and self.inner.operator == 'square':
            return self.inner.inner
        
        if self.operator == 'square' and type(self.inner) is UnaryOperatorSyntaxTree and self.inner.operator == 'sqrt':
            return self.inner.inner
        
        return self
    
    def __operate(self, inner:np.array) -> np.array:
        if self.operator == 'exp'   : return np.exp (inner)
        if self.operator == 'log'   : return np.log (inner)
        if self.operator == 'sqrt'  : return np.sqrt(inner)
        if self.operator == 'square': return inner ** 2
        if self.operator == 'cube'  : return inner ** 3
        raise RuntimeError(f"Operation not defined for operator {self.operator}.")
    
    def __operate_inv(self, output:np.array, output_relopt:Relopt) -> tuple[np.array, Relopt]:
        if self.operator == 'exp'   : return np.log(output), output_relopt
        if self.operator == 'log'   : return np.exp(output), output_relopt
        if self.operator == 'sqrt'  : return output ** 2, output_relopt
        if self.operator == 'square': return np.sqrt(output), output_relopt
        if self.operator == 'cube'  : return np.cbrt(output), output_relopt
        
        raise RuntimeError(f"Inverse operation not defined for operator {self.operator}.")
    
    def __str__(self) -> str:
        return f"{self.operator}({str(self.inner)})"
    
    def __eq__(self, other) -> bool:
        if type(other) is not UnaryOperatorSyntaxTree: return False
        return \
            self.operator == other.operator and \
            self.inner == other.inner
    
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
    
    def is_const(self) -> bool:
        return self.inner.is_const()
    
    def is_const_wrt(self, varidx) -> bool:
        return self.inner.is_const_wrt(varidx)
    
    def backprop_sign(self, symbmapper:SymbolMapper, lb:float=-numlims.INFTY, ub:float=numlims.INFTY, sign:str='+'):
        raise RuntimeError(f"Sign backprop not defined for operator {self.operator}.")  # TODO: not utilized for now.

    def pull_output(self, target_output:np.array, relopt:Relopt=Relopt('='), child=None, flatten:bool=False) -> tuple[np.array, Relopt]:
        pulled_output, pulled_relopt = super().pull_output(target_output, relopt, child, flatten)
        if child is None or pulled_output is None: return pulled_output, pulled_relopt
        if id(child) == id(self.inner):
            pulled_output, pulled_relopt = self.__operate_inv(pulled_output, pulled_relopt)
            if flatten: utils.flatten(pulled_output)
            return pulled_output, pulled_relopt
        raise RuntimeError('Invalid child.')
    
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


class ConstantSyntaxTree(SyntaxTree):
    def __init__(self, val:float):
        super().__init__()
        self.val = val
    
    def clone(self):
        return ConstantSyntaxTree(self.val)
    
    def compute_output(self, x):
        self.output = np.full(x.shape[0], self.val)
        return self.output
    
    def __str__(self) -> str:
        return "%.2f" % self.val
    
    def __eq__(self, other) -> bool:
        if type(other) is not ConstantSyntaxTree: return False
        return self.val == other.val
    
    def diff(self, varidx:int=0) -> SyntaxTree:
        return ConstantSyntaxTree(0)
    
    def is_const(self) -> bool:
        return True
    
    def is_const_wrt(self, varidx):
        return True
    
    def backprop_sign(self, symbmapper:SymbolMapper, lb:float=-numlims.INFTY, ub:float=numlims.INFTY, sign:str='+'):
        return self.val > 0 if sign == '+' else self.val < 0
    
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


class VariableSyntaxTree(SyntaxTree):
    def __init__(self, idx:int=0):
        super().__init__()
        self.idx = idx
    
    def clone(self):
        return VariableSyntaxTree(self.idx)
    
    def compute_output(self, x):
        self.output = x[:,self.idx] if x.ndim == 2 else x
        return self.output
    
    def __str__(self) -> str:
        return f"x{self.idx}"
    
    def __eq__(self, other) -> bool:
        if type(other) is not VariableSyntaxTree: return False
        return self.idx == other.idx
    
    def diff(self, varidx:int=0) -> SyntaxTree:
        return ConstantSyntaxTree(1)
    
    def is_const(self) -> bool:
        return False
    
    def is_const_wrt(self, varidx):
        return self.idx != varidx
    
    def backprop_sign(self, symbmapper:SymbolMapper, lb:float=-numlims.INFTY, ub:float=numlims.INFTY, sign:str='+'):
        raise NotImplementedError()
    
    def accept(self, visitor):
        visitor.visitVariable(self)
    
    def to_sympy(self, dps:int=None):
        return sympy.Symbol(str(self))
    
    def match(self, trunk) -> bool:
        if type(trunk) is UnknownSyntaxTree: return True
        if type(self) is not type(trunk): return False
        if self.idx != trunk.idx: return False
        return True


class FunctionSyntaxTree(SyntaxTree):
    def __init__(self, f):
        super().__init__()
        self.f = f
    
    def clone(self):
        return FunctionSyntaxTree(self.f.clone())
    
    def compute_output(self, x):
        self.output = self.f(x)
        return self.output
    
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
    
    def backprop_sign(self, symbmapper:SymbolMapper, lb:float=-numlims.INFTY, ub:float=numlims.INFTY, sign:str='+'):
        raise NotImplementedError()
    
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
        return SemanticSyntaxTree(self.sem)
    
    def compute_output(self, x):
        self.output = self.sem
        return self.output
    
    def __str__(self) -> str:
        return f"{self.sem}"#'SST(X)'
    
    def __eq__(self, other) -> bool:
        if id(self) == id(other): return True
        if type(other) is not SemanticSyntaxTree: return False
        return (self.sem == other.sem).all()
    
    def diff(self, varidx:int=0) -> SyntaxTree:
        raise NotImplementedError()
    
    def is_const(self) -> bool:
        return False
    
    def is_const_wrt(self, varidx):
        return False
    
    def backprop_sign(self, symbmapper:SymbolMapper, lb:float=-numlims.INFTY, ub:float=numlims.INFTY, sign:str='+'):
        raise NotImplementedError()
    
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
        return UnknownSyntaxTree(self.name, self.deriv, self.nvars, model=self.model)  # TODO: clone model
    
    def compute_output(self, x):
        self.output = None
        if self.model is None:
            raise RuntimeError('None unknown model.')
            #return None
        self.output = self.model(x)
        return self.output
    
    def __str__(self) -> str:
        xs = ''
        for i in range(self.nvars): xs += f"x{i},"
        return f"{self.label}({xs[:-1]})"
    
    def __eq__(self, other) -> bool:
        if type(other) is not UnknownSyntaxTree: return False
        return self.name == other.name and self.deriv == other.deriv
    
    def diff(self, varidx:int=0) -> SyntaxTree:
        assert varidx < self.nvars
        return UnknownSyntaxTree(name=self.name, deriv=self.deriv+(varidx,), nvars=self.nvars)
    
    def is_const(self) -> bool:
        if self.model is None: return False
        return False  # TODO: #self.model.is_const()
    
    def is_const_wrt(self, varidx):
        if self.model is None: return False
        return False  # TODO: #self.model.is_const_wrt(varidx)
    
    def backprop_sign(self, symbmapper:SymbolMapper, lb:float=-numlims.INFTY, ub:float=numlims.INFTY, sign:str='+'):
        constr = PropositionalConstraint( self, ConstantSyntaxTree(0), '>' if sign == '+' else '<', lb, ub )
        return symbmapper.map_as_symbol(constr)
    
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


class SyntaxTreeVisitor:
    def visitUnaryOperator (self, stree:UnaryOperatorSyntaxTree):  pass
    def visitBinaryOperator(self, stree:BinaryOperatorSyntaxTree): pass
    def visitConstant      (self, stree:ConstantSyntaxTree):       pass
    def visitVariable      (self, stree:VariableSyntaxTree):       pass
    def visitFunction      (self, stree:FunctionSyntaxTree):       pass
    def visitUnknown       (self, stree:UnknownSyntaxTree):        pass
    def visitSemantic      (self, stree:SemanticSyntaxTree):       pass


class SyntaxTreeNodeCollector(SyntaxTreeVisitor):
    def __init__(self):
        self.nodes = []
    def visitUnaryOperator (self, stree:UnaryOperatorSyntaxTree):  self.nodes.append(stree)
    def visitBinaryOperator(self, stree:BinaryOperatorSyntaxTree): self.nodes.append(stree)
    def visitConstant      (self, stree:ConstantSyntaxTree):       self.nodes.append(stree)
    def visitVariable      (self, stree:VariableSyntaxTree):       self.nodes.append(stree)
    def visitFunction      (self, stree:FunctionSyntaxTree):       self.nodes.append(stree)
    def visitUnknown       (self, stree:UnknownSyntaxTree):        self.nodes.append(stree)
    def visitSemantic      (self, stree:SemanticSyntaxTree):       self.nodes.append(stree)

class UnknownSyntaxTreeCollector(SyntaxTreeVisitor):
    def __init__(self):
        self.unknown_labels = set()
        self.unknowns = []
    def visitUnknown(self, stree:UnknownSyntaxTree):
        self.unknown_labels.add(stree.label)
        self.unknowns.append(stree)

class ConstantSyntaxTreeCollector(SyntaxTreeVisitor):
    def __init__(self):
        self.constants = []
    def visitConstant(self, stree:ConstantSyntaxTree):
        self.constants.append(stree)


class SyntaxTreeNodeCounter(SyntaxTreeVisitor):
    def __init__(self):
        self.nnodes = 0
    def visitUnaryOperator (self, stree:UnaryOperatorSyntaxTree):  self.nnodes += 1
    def visitBinaryOperator(self, stree:BinaryOperatorSyntaxTree): self.nnodes += 1
    def visitConstant      (self, stree:ConstantSyntaxTree):       self.nnodes += 1
    def visitVariable      (self, stree:VariableSyntaxTree):       self.nnodes += 1
    def visitFunction      (self, stree:FunctionSyntaxTree):       self.nnodes += 1
    def visitUnknown       (self, stree:UnknownSyntaxTree):        self.nnodes += 1
    def visitSemantic      (self, stree:SemanticSyntaxTree):       self.nnodes += 1


class SyntaxTreeOperatorCollector(SyntaxTreeVisitor):
    def __init__(self):
        self.opts = set()
    def visitUnaryOperator (self, stree:UnaryOperatorSyntaxTree):  self.opts.add(stree.operator)
    def visitBinaryOperator(self, stree:BinaryOperatorSyntaxTree): self.opts.add(stree.operator)


class SyntaxTreeNodeSelector(SyntaxTreeVisitor):
    def __init__(self, ith:int):
        self.ith = ith
        self.i = 0
        self.node = None
    def visitUnaryOperator(self, stree:UnaryOperatorSyntaxTree):
        if self.i == self.ith: self.node = stree
        self.i += 1
    def visitBinaryOperator(self, stree:BinaryOperatorSyntaxTree):
        if self.i == self.ith: self.node = stree
        self.i += 1
    def visitConstant(self, stree:ConstantSyntaxTree):
        if self.i == self.ith: self.node = stree
        self.i += 1
    def visitVariable(self, stree:VariableSyntaxTree):
        if self.i == self.ith: self.node = stree
        self.i += 1
    def visitFunction(self, stree:FunctionSyntaxTree):
        if self.i == self.ith: self.node = stree
        self.i += 1
    def visitUnknown(self, stree:UnknownSyntaxTree):
        if self.i == self.ith: self.node = stree
        self.i += 1
    def visitSemantic(self, stree:SemanticSyntaxTree):
        if self.i == self.ith: self.node = stree
        self.i += 1


class SyntaxTreeGenerator:
    OPERATORS = BinaryOperatorSyntaxTree.OPERATORS + UnaryOperatorSyntaxTree.OPERATORS

    def __init__(self, randstate:int=None, nvars:int=1):
        self.unkn_counter = 0
        self.randgen = random.Random() if randstate is None else random.Random(randstate)
        self.nvars = nvars
    
    def create_random(self, max_depth:int, n:int=1,
                      check_duplicates:bool=True, noconsts:bool=False, leaf_types:list[str]=['const', 'var']) -> list[SyntaxTree]:
        assert max_depth >= 0 and n >= 0
        strees = []
        for _ in range(n):
            self.unkn_counter = 0
            new_stree = self.__create_random(self.randgen.randint(0, max_depth), leaf_types)
            while not new_stree.validate() or (check_duplicates and new_stree in strees) or (noconsts and type(new_stree) is ConstantSyntaxTree):
                self.unkn_counter = 0
                new_stree = self.__create_random(self.randgen.randint(0, max_depth), leaf_types)
            strees.append(new_stree)
        return strees

    def __create_random(self, depth:int, leaf_types:list[str]):
        assert self.unkn_counter < len(string.ascii_uppercase)
        if depth <= 0:
            leaf_t = self.randgen.choice(leaf_types)
            if leaf_t == 'const':
                stree = ConstantSyntaxTree(val=self.randgen.uniform(-1., 1.))
                if np.isnan(stree.val): raise RuntimeError('NaN in random tree generation.')
            elif leaf_t == 'var':
                stree = VariableSyntaxTree(idx=self.randgen.randrange(self.nvars))
            elif leaf_t == 'unknown':
                stree = UnknownSyntaxTree(string.ascii_uppercase[self.unkn_counter], nvars=self.nvars)
                self.unkn_counter += 1
            else:
                raise RuntimeError('Cannot generator leaf node.')
            return stree
        
        operator = self.randgen.choice(SyntaxTreeGenerator.OPERATORS)
        
        stree = None
        if operator in UnaryOperatorSyntaxTree.OPERATORS:
            stree = UnaryOperatorSyntaxTree(operator, self.__create_random(depth - 1, leaf_types))

        else:
            stree = BinaryOperatorSyntaxTree(operator,
                        self.__create_random(depth - 1, leaf_types),
                        self.__create_random(depth - 1, leaf_types) if operator != '^' else ConstantSyntaxTree(self.randgen.choice([2.0,3.0,4.0])))  # TODO: only ^2 managed.
        
        return stree.simplify()

