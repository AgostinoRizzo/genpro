import numpy as np
from symbols.syntax_tree import SyntaxTree
from symbols.const import ConstantSyntaxTree


class BinaryOperatorSyntaxTree(SyntaxTree):
    OPERATORS = ['+', '-', '*', '/'] #, '^']
    INVERTIBLE_OPERATORS = ['+', '-', '*', '/']

    def __init__(self, operator:str, left:SyntaxTree, right:SyntaxTree):
        super().__init__()
        self.operator = operator
        self.left = left
        self.right = right
    
    def clone(self):
        c = BinaryOperatorSyntaxTree(self.operator, self.left.clone(), self.right.clone())
        c.copy_output_from(self)
        return c
    
    def __call__(self, x):
        if self.output is None:
            self.output = self.__operate(self.left(x), self.right(x))
        return self.output
    
    def __getitem__(self, x_d):
        x, d = x_d
        if d not in self.y_know:
            if d == ():
                self.y_know[d] = self.__operate(self.left[x_d], self.right[x_d])
            elif len(d) == 1:
                self.y_know[d] = self.__operate_deriv(self.left[(x,())], self.left[x_d], self.right[(x,())], self.right[x_d])
            else:
                raise RuntimeError(f"Derivative {d} not supported.")
        return self.y_know[d]
    
    def clear_output(self):
        super().clear_output()
        self.left.clear_output()
        self.right.clear_output()
    
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
        return self
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
            if self.left == self.right:
                return UnaryOperatorSyntaxTree('square', self.left)
        
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
    
    def __operate_inv(self, output:np.array, get_left:bool=True) -> np.array:
        if   self.operator == '/':
            self.right.output[self.right.output == 0.0] = np.nan
            
            # get the numerator.
            if get_left:
                return  output * self.right.output
            
            # get the denominator.
            return self.left.output / output
        
        elif self.operator == '*':
            num = output
            den = self.right.output if get_left else self.left.output
            return  num / den

        elif self.operator == '+': return (output - self.right.output) if get_left else (output - self.left.output)
        elif self.operator == '-': return (output + self.right.output) if get_left else (self.left.output - output)
        elif self.operator == '^':
            if get_left:
                if np.all(self.right.output == 2.): return np.sqrt(output)
                if np.all(self.right.output == 3.): return np.cbrt(output)
                raise RuntimeError(f"Inverse power not defined for exponent {self.right.output}")
            else:
                np.log(output, self.left.output)            
        
        raise RuntimeError(f"Inverse operation not defined for operator {self.operator}.")
    
    def __operate_deriv(self, left:np.array, left_deriv:np.array, right:np.array, right_deriv:np.array) -> np.array:
        if   self.operator == '/': return ((right*left_deriv) - (left*right_deriv)) / (left**2)
        elif self.operator == '*': return (right*left_deriv) + (left*right_deriv)
        elif self.operator == '+': return left_deriv + right_deriv
        elif self.operator == '-': return left_deriv - right_deriv
        raise RuntimeError(f"Derivative not defined for operator {self.operator}.")
    
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

    def pull_output(self, target_output:np.array, child=None, flatten:bool=False) -> np.array:
        pulled_output = super().pull_output(target_output, child, flatten)
        if child is None or pulled_output is None:
            return pulled_output
        if   id(child) == id(self.left):
            pulled_output = self.__operate_inv(pulled_output, get_left=True)
        elif id(child) == id(self.right):
            pulled_output = self.__operate_inv(pulled_output, get_left=False)
        else:
            raise RuntimeError('Invalid child.')
        if flatten:
            pulled_output = utils.flatten(pulled_output)
        return pulled_output
    
    def pull_know(self, k_target:np.array, noroot_target:bool=False, child=None, track:dict={}) -> tuple[np.array,bool]:
               
        k_pulled, noroot_pulled = super().pull_know(k_target, noroot_target, track=track)
        if child is None:
            return k_pulled, noroot_pulled
        
        A = None
        B = None
        pull_left = True
        child_id = id(child)
        if child_id == id(self.left):
            A = self.left
            B = self.right
        elif child_id == id(self.right):
            A = self.right
            B = self.left
            pull_left = False
        else:
            raise RuntimeError('Invalid child.')
        k_A = np.sign(A.y_know[()])  # TODO: never used!
        k_B = np.sign(B.y_know[()])
        
        k_target = k_pulled
        noroot_target = noroot_pulled
        k_pulled = np.full(k_target.shape, np.nan)
        noroot_pulled = False

        if self.operator == '+':
            k_pulled[(k_target > 0.0) & (k_B < 0.0)] = +1.0
            k_pulled[(k_target < 0.0) & (k_B > 0.0)] = -1.0
        
        elif self.operator == '-':
            if pull_left:
                k_pulled[(k_target > 0.0) & (k_B > 0.0)] = +1.0
                k_pulled[(k_target < 0.0) & (k_B < 0.0)] = -1.0
            else:
                k_pulled[(k_target > 0.0) & (k_B < 0.0)] = -1.0
                k_pulled[(k_target < 0.0) & (k_B > 0.0)] = +1.0
        
        elif self.operator == '*' or self.operator == '/':
            pos_mask = k_target > 0.0
            neg_mask = k_target < 0.0
            k_pulled[pos_mask] =  k_B[pos_mask]
            k_pulled[neg_mask] = -k_B[neg_mask]

            if noroot_target:
                noroot_pulled = True
            elif self.operator == '/' and not pull_left:
                noroot_pulled = True
        
        else:
            raise RuntimeError('Invalid operator.')
        
        track[id(A)] = (k_pulled, noroot_pulled)
        return k_pulled, noroot_pulled
    
    def pull_know_deriv(self, image_track:dict, derividx:int, k_target:np.array, child=None) -> np.array:
        
        k_pulled = super().pull_know_deriv(image_track, derividx, k_target)
        if child is None:
            return k_pulled
        
        A = None
        B = None
        pull_left = True
        child_id = id(child)
        if child_id == id(self.left):
            A = self.left
            B = self.right
        elif child_id == id(self.right):
            A = self.right
            B = self.left
            pull_left = False
        else:
            raise RuntimeError('Invalid child.')
        k_A  = image_track[id(A)][0]
        k_B  = np.sign(B.y_know[()])
        k_dB = np.sign(B.y_know[(derividx,)])
        
        k_target = k_pulled
        k_pulled = np.full(k_target.shape, np.nan)

        if self.operator == '+':
            k_pulled[(k_target > 0.0) & (k_dB < 0.0)] = +1.0
            k_pulled[(k_target < 0.0) & (k_dB > 0.0)] = -1.0
        
        elif self.operator == '-':
            if pull_left:
                k_pulled[(k_target > 0.0) & (k_dB > 0.0)] = +1.0
                k_pulled[(k_target < 0.0) & (k_dB < 0.0)] = -1.0
            else:
                k_pulled[(k_target > 0.0) & (k_dB < 0.0)] = -1.0
                k_pulled[(k_target < 0.0) & (k_dB > 0.0)] = +1.0
        
        elif self.operator == '*':
            k_A_isknown = ~np.isnan(k_A)

            mask = k_A_isknown & (k_target > 0.0) & (k_A != k_dB)
            k_pulled[mask] =  k_B[mask]

            mask = k_A_isknown & (k_target < 0.0) & (k_A == k_dB)
            k_pulled[mask] = -k_B[mask]
        
        elif self.operator == '/':
            k_A_isknown = ~np.isnan(k_A)
            if pull_left:
                mask = k_A_isknown & (k_target > 0.0) & (k_A == k_dB)
                k_pulled[mask] =  k_B[mask]

                mask = k_A_isknown & (k_target < 0.0) & (k_A != k_dB)
                k_pulled[mask] = -k_B[mask]
            else:
                mask = k_A_isknown & (k_target > 0.0) & (k_A != k_dB)
                k_pulled[mask] = -k_B[mask]

                mask = k_A_isknown & (k_target < 0.0) & (k_A == k_dB)
                k_pulled[mask] =  k_B[mask]
        
        else:
            raise RuntimeError('Invalid operator.')
        
        return k_pulled

    def get_coeffs(self, coeffs:list):
        self.left.get_coeffs(coeffs)
        self.right.get_coeffs(coeffs)
    
    def set_coeffs(self, coeffs:list, start:int=0):
        self.left.set_coeffs(coeffs, start)
        self.right.set_coeffs(coeffs, start)
    
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
    
    def get_nodes(self, nodes:list):
        super().get_nodes(nodes)
        self.left.get_nodes(nodes)
        self.right.get_nodes(nodes)
    
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