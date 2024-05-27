import sympy
import math
import numbs

class PropositionalConstraint:
    def __init__(self, f, g, opt:str='>', lb:float=-numbs.INFTY, ub:float=numbs.INFTY):
        self.f = f
        self.g = g
        self.opt = opt
        self.lb = lb
        self.ub = ub
    
    def __str__(self) -> str:
        lb_str = numbs.tostr(self.lb)
        ub_str = numbs.tostr(self.ub)
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


class PullViolation(RuntimeError):
    pass


class SyntaxTree:
    def __init__(self):
        self.parent = None
        self.output = None
    def clone(self): return None
    def compute_output(self, x) -> float: return None
    def set_parent(self, parent=None): self.parent = parent
    def simplify(self): return self
    def __str__(self) -> str: return ''
    def __eq__(self, other) -> bool: return False
    def diff(self): return None
    def backprop_sign(self, symbmapper:SymbolMapper, lb:float=-numbs.INFTY, ub:float=numbs.INFTY, sign:str='+'): pass
    def pull_output(self, target_output:float, relopt:Relopt=Relopt('='), child=None): # -> float, Relopt
        return (target_output, relopt) if self.parent is None else self.parent.pull_output(target_output, relopt, self)
    def get_unknown_stree(self, unknown_stree_label:str): return None
    def set_unknown_model(self, model_label:str, model:callable, coeffs_mask:list[float]=None, constrs:dict=None): pass
    def count_unknown_model(self, model_label:str) -> int: return 0
    def accept(self, visitor): pass


class BinaryOperatorSyntaxTree(SyntaxTree):
    def __init__(self, operator:str, left:SyntaxTree, right:SyntaxTree):
        super().__init__()
        self.operator = operator
        self.left = left
        self.right = right
    
    def clone(self):
        return BinaryOperatorSyntaxTree(self.operator, self.left.clone(), self.right.clone())
    
    def compute_output(self, x) -> float:
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

    def simplify(self):
        self.left = self.left.simplify()
        self.right = self.right.simplify()
        is_left_const  = type(self.left)  is ConstantSyntaxTree
        is_right_const = type(self.right) is ConstantSyntaxTree

        if is_left_const and is_right_const:
            return ConstantSyntaxTree( self.__operate(self.left.val, self.right.val) )
        
        if self.operator == '*':
            if (is_left_const and self.left.val == 0) or (is_right_const and self.right.val == 0):
                return ConstantSyntaxTree(0)
            if (is_left_const and self.left.val == 1):
                return self.right
            if (is_right_const and self.right.val == 1):
                return self.left
        
        if self.operator == '+' or self.operator == '-':
            if is_left_const  and self.left.val  == 0: return self.right
            if is_right_const and self.right.val == 0: return self.left
        
        if self.operator == '^' and is_right_const:
            if self.right.val == 1:
                return self.left
            
            if self.right.val == 2 and type(self.left) is UnaryOperatorSyntaxTree and self.left.operator == 'sqrt':
                return self.left.inner
            
            if type(self.left) is BinaryOperatorSyntaxTree and self.left.operator == '^' and \
               type(self.left.right) is ConstantSyntaxTree:
                self.right = ConstantSyntaxTree(self.right.val * self.left.right.val)
                self.left = self.left.left
                return self
        
        return self
    
    def __operate(self, left:float, right:float) -> float:
        if   self.operator == '/': return left / right
        elif self.operator == '*': return left * right
        elif self.operator == '+': return left + right
        elif self.operator == '-': return left - right
        elif self.operator == '^': return left ** right
        raise RuntimeError(f"Operation not defined for operator {self.operator}.")
    
    def __operate_inv(self, output:float, output_relopt:Relopt, get_left:bool=True): # -> float, Relopt
        if   self.operator == '/':
            if get_left: return output * self.right.output, (output_relopt if self.right.output >= 0 else output_relopt.neg())
            if output != 0. and output_relopt.opt == '=':
                return self.left.output / output,  (output_relopt if self.left.output >= 0 else output_relopt.neg())
            if output != 0. or output_relopt.opt == '=':
                if output_relopt.opt == '=': raise PullViolation()
                raise RuntimeError(f"Inverse for {self.left.output}/x{output_relopt.opt}{output} not defined.")
            if self.left.output >= 0.: return 0., output_relopt.strict()
            return 0., output_relopt.neg().strict()
            #if output == 0.: return numbs.INFTY, Relopt('=')
        
        elif self.operator == '*':
            num = output
            den = self.right.output if get_left else self.left.output
            if den == 0.: raise PullViolation()
            return (num / den), (output_relopt if den > 0 else output_relopt.neg())

        elif self.operator == '+': return output - self.right.output if get_left else output - self.left.output, output_relopt
        elif self.operator == '-': return output + self.right.output if get_left else self.left.output - output, output_relopt
        elif self.operator == '^': return output ** (1/self.right.output) if get_left else math.log(output, self.left.output), output_relopt
        
        raise RuntimeError(f"Inverse operation not defined for operator {self.operator}.")
    
    def __str__(self) -> str:
        return f"({str(self.left)} {self.operator} {str(self.right)})"
    
    def __eq__(self, other) -> bool:
        if type(other) is not BinaryOperatorSyntaxTree: return False
        return \
            self.operator == other.operator and \
            self.left == other.left and self.right == other.right
    
    def diff(self) -> SyntaxTree:
        f = self.left
        g = self.right

        if self.operator == '/':
            return BinaryOperatorSyntaxTree( '/',
                BinaryOperatorSyntaxTree('-',
                    BinaryOperatorSyntaxTree('*', f.diff(), g.clone()),
                    BinaryOperatorSyntaxTree('*', f.clone(), g.diff())
                ),
                BinaryOperatorSyntaxTree('^', g.clone(), ConstantSyntaxTree(2))
            )
        
        if self.operator == '*':
            return BinaryOperatorSyntaxTree( '+',
                BinaryOperatorSyntaxTree('*', f.diff(), g.clone()),
                BinaryOperatorSyntaxTree('*', f.clone(), g.diff())
            )
        
        elif self.operator == '+' or self.operator == '-':
            return BinaryOperatorSyntaxTree( self.operator, f.diff(), g.diff() )
        
        elif self.operator == '^':
            if type(g) is not ConstantSyntaxTree:
                raise RuntimeError(f"Differentiation not defined for operator {self.operator} and non-constant exponent.")
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
                f.diff()
            )
        
        raise RuntimeError(f"Differentiation not defined for operator {self.operator}.")
    
    def backprop_sign(self, symbmapper:SymbolMapper, lb:float=-numbs.INFTY, ub:float=numbs.INFTY, sign:str='+'):
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

    def pull_output(self, target_output:float, relopt:Relopt=Relopt('='), child=None): # -> float, Relopt
        pulled_output, pulled_relopt = super().pull_output(target_output, relopt, child)
        if child is None or pulled_output is None: return pulled_output, pulled_relopt
        if id(child) == id(self.left):  return self.__operate_inv(pulled_output, pulled_relopt, get_left=True)
        if id(child) == id(self.right): return self.__operate_inv(pulled_output, pulled_relopt, get_left=False)
        raise RuntimeError('Invalid child.')
    
    def get_unknown_stree(self, unknown_stree_label:str):
        stree = self.left.get_unknown_stree(unknown_stree_label)
        if stree is not None: return stree
        return self.right.get_unknown_stree(unknown_stree_label)

    def set_unknown_model(self, model_label:str, model:callable, coeffs_mask:list[float]=None, constrs:dict=None):
        self.left .set_unknown_model(model_label, model, coeffs_mask, constrs)
        self.right.set_unknown_model(model_label, model, coeffs_mask, constrs)
    
    def count_unknown_model(self, model_label:str) -> int:
        return self.left.count_unknown_model(model_label) + \
               self.right.count_unknown_model(model_label)
    
    def accept(self, visitor):
        visitor.visitBinaryOperator(self)
        self.left.accept(visitor)
        self.right.accept(visitor)


class UnaryOperatorSyntaxTree(SyntaxTree):
    def __init__(self, operator:str, inner:SyntaxTree):
        super().__init__()
        self.operator = operator
        self.inner = inner
    
    def clone(self):
        return UnaryOperatorSyntaxTree(self.operator, self.inner.clone())
    
    def compute_output(self, x) -> float:
        self.output  = None
        inner_output  = self.inner.compute_output(x)
        if inner_output is None: return None
        self.output = self.__operate(inner_output)
        return self.output
    
    def set_parent(self, parent=None):
        super().set_parent(parent)
        self.inner.set_parent(self)

    def simplify(self):
        self.inner = self.inner.simplify()

        if type(self.inner) is ConstantSyntaxTree:
            return ConstantSyntaxTree( self.__operate(self.inner.val) )
        
        if self.operator == 'exp' and type(self.inner) is UnaryOperatorSyntaxTree and self.inner.operator == 'log':
            return self.self.inner.inner
        
        if self.operator == 'log' and type(self.inner) is UnaryOperatorSyntaxTree and self.inner.operator == 'exp':
            return self.self.inner.inner
        
        if self.operator == 'sqrt' and type(self.inner) is BinaryOperatorSyntaxTree and \
           self.inner.operator == '^' and type(self.inner.right) is ConstantSyntaxTree and self.inner.right.val == 2:
            return self.inner.left
        
        return self
    
    def __operate(self, inner:float) -> float:
        if   self.operator == 'exp' : return math.exp (inner)
        elif self.operator == 'log' : return math.log (inner)
        elif self.operator == 'sqrt': return math.sqrt(inner)
        raise RuntimeError(f"Operation not defined for operator {self.operator}.")
    
    def __operate_inv(self, output:float, output_relopt:Relopt): # -> float, Relopt
        if self.operator == 'exp' : return math.log(output), output_relopt
        if self.operator == 'log' : return math.exp(output), output_relopt
        if self.operator == 'sqrt': return output ** 2, output_relopt
        
        raise RuntimeError(f"Inverse operation not defined for operator {self.operator}.")
    
    def __str__(self) -> str:
        return f"{self.operator}({str(self.inner)})"
    
    def __eq__(self, other) -> bool:
        if type(other) is not UnaryOperatorSyntaxTree: return False
        return \
            self.operator == other.operator and \
            self.inner == other.inner
    
    def diff(self) -> SyntaxTree:
        g = self.inner

        if self.operator == 'exp':
            return BinaryOperatorSyntaxTree( '*',
                UnaryOperatorSyntaxTree('exp', g.clone() ),
                g.diff()
                )
        
        if self.operator == 'log':
            return BinaryOperatorSyntaxTree( '/',
                g.diff(),
                g.clone()
            )
        
        if self.operator == 'sqrt':
            return BinaryOperatorSyntaxTree( '/',
                g.diff(),
                BinaryOperatorSyntaxTree( '*',
                    ConstantSyntaxTree(2),
                    UnaryOperatorSyntaxTree('sqrt', g.clone())
                )
            )
        
        raise RuntimeError(f"Differentiation not defined for operator {self.operator}.")
    
    def backprop_sign(self, symbmapper:SymbolMapper, lb:float=-numbs.INFTY, ub:float=numbs.INFTY, sign:str='+'):
        raise RuntimeError(f"Sign backprop not defined for operator {self.operator}.")  # TODO: not utilized for now.

    def pull_output(self, target_output:float, relopt:Relopt=Relopt('='), child=None): # -> float, Relopt
        pulled_output, pulled_relopt = super().pull_output(target_output, relopt, child)
        if child is None or pulled_output is None: return pulled_output, pulled_relopt
        if id(child) == id(self.inner):  return self.__operate_inv(pulled_output, pulled_relopt)
        raise RuntimeError('Invalid child.')
    
    def get_unknown_stree(self, unknown_stree_label:str):
        return self.inner.get_unknown_stree(unknown_stree_label)

    def set_unknown_model(self, model_label:str, model:callable, coeffs_mask:list[float]=None, constrs:dict=None):
        self.inner.set_unknown_model(model_label, model, coeffs_mask, constrs)
    
    def count_unknown_model(self, model_label:str) -> int:
        return self.inner.count_unknown_model(model_label)
    
    def accept(self, visitor):
        visitor.visitBinaryOperator(self)
        self.inner.accept(visitor)


class ConstantSyntaxTree(SyntaxTree):
    def __init__(self, val:float):
        super().__init__()
        self.val = val
    
    def clone(self):
        return ConstantSyntaxTree(self.val)
    
    def compute_output(self, x) -> float:
        self.output = self.val
        return self.output
    
    def __str__(self) -> str:
        return str(self.val)
    
    def __eq__(self, other) -> bool:
        if type(other) is not ConstantSyntaxTree: return False
        return self.val == other.val
    
    def diff(self) -> SyntaxTree:
        return ConstantSyntaxTree(0)
    
    def backprop_sign(self, symbmapper:SymbolMapper, lb:float=-numbs.INFTY, ub:float=numbs.INFTY, sign:str='+'):
        return self.val > 0 if sign == '+' else self.val < 0
    
    def accept(self, visitor):
        visitor.visitConstant(self)
        

class UnknownSyntaxTree(SyntaxTree):
    def __init__(self, label:str='A', model=None, coeffs_mask=None, constrs=None):
        super().__init__()
        self.label = label
        self.model = None
        self.coeffs_mask = None
        self.constrs = None
    
    def clone(self):
        return UnknownSyntaxTree(self.label)
    
    def compute_output(self, x) -> float:
        self.output = None
        if self.model is None:
            print("None model!!")
            return None
        self.output = self.model(x)
        return self.output
    
    def __str__(self) -> str:
        return f"{self.label}(x)"
    
    def __eq__(self, other) -> bool:
        if type(other) is not UnknownSyntaxTree: return False
        return self.label == other.label
    
    def diff(self) -> SyntaxTree:
        return UnknownSyntaxTree(f"{self.label}'")
    
    def backprop_sign(self, symbmapper:SymbolMapper, lb:float=-numbs.INFTY, ub:float=numbs.INFTY, sign:str='+'):
        constr = PropositionalConstraint( self, ConstantSyntaxTree(0), '>' if sign == '+' else '<', lb, ub )
        return symbmapper.map_as_symbol(constr)
    
    def get_unknown_stree(self, unknown_stree_label:str):
        if self.label == unknown_stree_label: return self
        return None

    def set_unknown_model(self, model_label:str, model:callable, coeffs_mask:list[float]=None, constrs:dict=None):
        if self.label == model_label:
            self.model = model
            self.coeffs_mask = coeffs_mask
            self.constrs = constrs
    
    def count_unknown_model(self, model_label:str) -> int:
        return 1 if self.label == model_label else 0
    
    def accept(self, visitor):
        visitor.visitUnknown(self)


class SyntaxTreeVisitor:
    def visitBinaryOperator(self, stree:BinaryOperatorSyntaxTree): pass
    def visitConstant      (self, stree:ConstantSyntaxTree):       pass
    def visitUnknown       (self, stree:UnknownSyntaxTree):        pass


class UnknownSyntaxTreeCollector(SyntaxTreeVisitor):
    def __init__(self):
        self.unknown_labels = set()
    
    def visitBinaryOperator(self, stree:BinaryOperatorSyntaxTree): pass
    def visitConstant      (self, stree:ConstantSyntaxTree):       pass
    def visitUnknown       (self, stree:UnknownSyntaxTree):
        self.unknown_labels.add(stree.label)
