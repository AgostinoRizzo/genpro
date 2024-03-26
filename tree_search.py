import numpy as np
from scipy.optimize import minimize, Bounds, root
#import tensorflow as tf
import sympy
import math
import random
import time
import dataset

import warnings
warnings.filterwarnings("error")

def __get_slope(dp_i: dataset.DataPoint, dp_j: dataset.DataPoint) -> float:
    return (dp_j.y - dp_i.y) / (dp_j.x - dp_i.x)

"""
requires S.data as mesh
"""
def compute_smoothness(S: dataset):  # TODO: kernel based computation
    n = len(S.data)
    assert(n >= 2)

    slopes = []
    for i in range(1, n):
        slopes.append(__get_slope(S.data[i-1], S.data[i]))
    
    return np.average(slopes), np.std(slopes)


"""
Sum Operator: a+b=k
k: target image
a: alpha value
return: beta value
"""
def __sum_opt_beta(k:float, a:float) -> float:
    return k - a

"""
Product Operator: a*b=k
k: target image
a: alpha value
return: beta value
"""
def __prod_opt_beta(k:float, a:float) -> float:
    return k / a

"""
Power Operator: a^b=k
k: target image
a: alpha value
return: beta value
"""
def __pow_opt_beta(k:float, a:float) -> float:
    return math.log(k, a)

"""
Division Operator: a/b=k
k: target image
a: alpha value
return: beta value
"""
def __div_opt_beta(k:float, a:float) -> float:
    return a / k

"""
requires S.data as mesh
"""
def infer_operator(S:dataset.Dataset, opt:str='sum'):
    __opt_beta = __sum_opt_beta
    if   opt == 'sum' : pass
    elif opt == 'prod': __opt_beta = __prod_opt_beta
    elif opt == 'pow' : __opt_beta = __pow_opt_beta
    elif opt == 'div' : __opt_beta = __div_opt_beta
    else: raise RuntimeError('Invalid operator.')

    n = len(S.data)
    eps = 0.001
    y_radius = 0.001
    i = 0
    alphas = []
    betas  = []

    trial = 0

    while i < n and trial < 100:
        i = 0
        iter = 0
        alpha_origin = random.uniform(S.yl, S.yu)

        while i < n and iter < 10000:
            k = S.data[i].y

            if i == 0:
                step = 0
                if iter > 0:
                    step = (eps * (iter/2)) if iter % 2 == 0 else (-eps * ((iter+1)/2))
                alphas = [alpha_origin + step]
                try: betas  = [__opt_beta(k, alphas[0])]
                except:
                    i = -1
                    iter += 1
            else:
                l = alphas[i-1] - y_radius
                u = alphas[i-1] + y_radius

                a_best = None
                b_best = None
                diff_min = y_radius

                if l <= u:
                    for a in np.linspace(l, u, int((u - l) / eps)):
                        try: b = __opt_beta(k, a)
                        except: continue
                        diff = abs(b - betas[i-1]) #+ abs(a - alphas[i-1])) / 2
                        if diff < diff_min:
                            a_best = a
                            b_best = b
                            diff_min = diff
                
                if diff_min == y_radius:
                    i = -1
                    iter += 1
                else:
                    alphas.append(a_best)
                    betas.append(b_best)

            i += 1
        
        trial += 1
    
    print(f"Stop at: {alphas[0]}")
    print(f"Trials: {trial}, Last iterations: {iter}")

    S_alphas = dataset.Dataset()
    S_betas  = dataset.Dataset()
    
    if len(alphas) == n and len(betas) == n:
        for i in range(n):
            S_alphas.data.append(dataset.DataPoint(S.data[i].x, alphas[i]))
            S_betas .data.append(dataset.DataPoint(S.data[i].x, betas [i]))
    else:
        alphas = []
        betas  = []
        print('No solution found.')

    return alphas, betas#, S_alphas, S_betas


"""
requires S.data as mesh
"""
def infer_operator_optimz(S:dataset.Dataset, opt:str='sum'):
    __opt_beta = __sum_opt_beta
    if   opt == 'sum' : pass
    elif opt == 'prod': __opt_beta = __prod_opt_beta
    elif opt == 'pow' : __opt_beta = __pow_opt_beta
    elif opt == 'div' : __opt_beta = __div_opt_beta
    else: raise RuntimeError('Invalid operator.')

    n = len(S.data)
    alphas0 = np.array([random.uniform(1, 10) for i in range(n)])
    bounds = Bounds(lb=-np.inf) #1e-10

    def target(alphas:np.array) -> float:
        val = 0.
        slopes = []
        for i in range(2, n):
            alpha_slope_prev = alphas[i-1] - alphas[i-2]
            alpha_slope_curr = alphas[i] - alphas[i-1]
            val += (alpha_slope_curr - alpha_slope_prev) ** 2

            beta_slope_prev = __opt_beta(S.data[i-1].y, alphas[i-1]) - __opt_beta(S.data[i-2].y, alphas[i-2])
            beta_slope_curr = __opt_beta(S.data[i].y, alphas[i]) - __opt_beta(S.data[i-1].y, alphas[i-1])
            val += (beta_slope_curr - beta_slope_prev) ** 2

            slopes.append(alpha_slope_curr)
            slopes.append(beta_slope_curr)

        return val
    
    res = minimize(target, alphas0, method='nelder-mead',
                   options={'xatol': 1e-8, 'disp': True, 'maxiter': n*800},
                   bounds=bounds)
    
    alphas = res.x
    betas = []
    for i in range(n): betas.append(__opt_beta(S.data[i].y, alphas[i]))
    betas = np.array(betas)

    print(res)
    return alphas, betas


"""
requires S.data as mesh
"""
def infer_operator_gradoptimz(S:dataset.Dataset, opt:str='sum'):
    __opt_beta = __sum_opt_beta
    if   opt == 'sum' : pass
    elif opt == 'prod': __opt_beta = __prod_opt_beta
    elif opt == 'pow' : __opt_beta = __pow_opt_beta
    elif opt == 'div' : __opt_beta = __div_opt_beta
    else: raise RuntimeError('Invalid operator.')

    n = len(S.data)

    def target(alphas:np.array) -> float:
        val = 0.
        slopes = []
        for i in range(2, n):
            alpha_slope_prev = alphas[i-1] - alphas[i-2]
            alpha_slope_curr = alphas[i] - alphas[i-1]
            val += (alpha_slope_curr - alpha_slope_prev) ** 2

            beta_slope_prev = __opt_beta(S.data[i-1].y, alphas[i-1]) - __opt_beta(S.data[i-2].y, alphas[i-2])
            beta_slope_curr = __opt_beta(S.data[i].y, alphas[i]) - __opt_beta(S.data[i-1].y, alphas[i-1])
            #val += (beta_slope_curr - beta_slope_prev) ** 2

            slopes.append(alpha_slope_curr)
            slopes.append(beta_slope_curr)

        return val
    
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.05,  epsilon=1e-07,)
    alphas_curr = np.array([random.uniform(S.yl, S.yu) for i in range(n)])

    num_steps = 5000
    h = 1e-5
    learning_rate = 0.05
    gradient = None

    for i in range(num_steps):
        
        gradient = []
        for vidx in range(n):
            target_curr = target(alphas_curr)
            alphas_curr[vidx] += h
            target_next = target(alphas_curr)
            alphas_curr[vidx] -= h
            
            gradient.append( (target_next-target_curr) / h )

        gradient = np.array(gradient)
        #print(gradient)
        
        #optimizer.apply_gradients([(gradient, alphas_curr)])
        alphas_curr += - learning_rate * gradient
    
    alphas = alphas_curr
    betas = []
    for i in range(n): betas.append(__opt_beta(S.data[i].y, alphas[i]))
    betas = np.array(betas)

    print(f"Last gradient {gradient}")

    return alphas, betas


"""
requires S.data as mesh
"""
def infer_operator_slq_optimz(S:dataset.Dataset, opt:str='sum'):
    __opt_beta = __sum_opt_beta
    if   opt == 'sum' : pass
    elif opt == 'prod': __opt_beta = __prod_opt_beta
    elif opt == 'pow' : __opt_beta = __pow_opt_beta
    elif opt == 'div' : __opt_beta = __div_opt_beta
    else: raise RuntimeError('Invalid operator.')

    n = len(S.data)

    def get_target(alphas):
        target_expr = 0.
        slopes = []
        slopes_avg = 0.

        for i in range(2, n):
            alpha_slope_prev = alphas[i-1] - alphas[i-2]
            alpha_slope_curr = alphas[i] - alphas[i-1]
            #target_expr += 0.5 * (alpha_slope_curr**2 - alpha_slope_prev**2) ** 2
            target_expr += (alpha_slope_curr - alpha_slope_prev) ** 2

            beta_slope_prev = __opt_beta(S.data[i-1].y, alphas[i-1]) - __opt_beta(S.data[i-2].y, alphas[i-2])
            beta_slope_curr = __opt_beta(S.data[i].y, alphas[i]) - __opt_beta(S.data[i-1].y, alphas[i-1])
            #target_expr += 0.5 * (beta_slope_curr**2 - beta_slope_prev**2) ** 2
            #target_expr += beta_slope_curr ** 2

            #slopes.append(alpha_slope_curr)
            slopes.append(beta_slope_curr)
            slopes_avg += alpha_slope_curr
            slopes_avg += beta_slope_curr
        
        #for s in slopes:
        #    target_expr += (s - slopes_avg) ** 2
        #target_expr /= n-1

        #target_expr = sympy.simplify(target_expr)
        return target_expr
    
    alphas = sympy.symbols(f"a0:{n}")
    target_expr = get_target(alphas)
    print(f"TargetExp: {target_expr}")

    gradient = []
    gradient_lamb = []
    for i in range(n):
        g_i = sympy.diff(target_expr, alphas[i])
        gradient.append(g_i)
        gradient_lamb.append(sympy.lambdify(list(g_i.free_symbols), g_i))
    print(f"Gradient: {gradient}")

    ## solve/nsolve
    #nres = sympy.solvers.solvers.nsolve(gradient, alphas, [1. for _ in range(n)], maxsteps=10)
    #res = {}
    #for i in range(n): res[alphas[i]] = nres[i]
    ##res = sympy.solve(gradient)
    #print(f"Result: {res}")

    ## gradient descent
    n_steps = 2000
    learning_rate = 0.01
    curr_alphas = [random.uniform(S.yl, S.yu) for _ in range(n)]
    curr_gradient = [1. for _ in range(n)]
    alphas_idx_map = {}
    for i in range(n): alphas_idx_map[alphas[i]] = i
    res = {}

    for curr_iter in range(n_steps):

        for i in range(n):
            alphas_subs = []
            for s in gradient[i].free_symbols:
                sidx = alphas_idx_map[s]
                alphas_subs.append(curr_alphas[sidx])
            #print( *alphas_subs )
            curr_gradient[i] = float(gradient_lamb[i]( *alphas_subs ))
        
        #print(f"[{curr_iter}] Current gradient: {np.linalg.norm(np.array(curr_gradient))}")

        for i in range(n):
            curr_alphas[i] += -learning_rate * curr_gradient[i]
        
    for i in range(n):
        res[alphas[i]] = curr_alphas[i]
    print(f"Result: {res}")
    print(f"Final gradient: {np.linalg.norm(np.array(curr_gradient))}")


    alphas_sol = []
    betas_sol = []
    for i in range(n):
        
        ai_sol = res[alphas[i]] if alphas[i] in res.keys() else 0.
        free_ai:set = set() if type(ai_sol) == float else ai_sol.free_symbols
        #print("Free symbs: " + str(free_ai))
        for fai in free_ai: ai_sol = ai_sol.subs(fai, 0)
        #print("ai_sol: " + str(ai_sol))

        alphas_sol.append( ai_sol )
        betas_sol.append( __opt_beta(S.data[i].y, alphas_sol[i]) )
    
    return np.array(alphas_sol), np.array(betas_sol)


"""
infer_syntaxtree + infer_poly
requires S.data as mesh
"""
class OperationDomainError(RuntimeError):
    pass


class OperatorFactory:
    def get_div (self, x1, x2):
        try: return x1 / x2
        except RuntimeWarning: raise OperationDomainError()
    def get_sqrt(self, x):      raise RuntimeError('Operation not defined.')
    def get_log (self, x):      raise RuntimeError('Operation not defined.')
    def get_exp (self, x):      raise RuntimeError('Operation not defined.')
    def get_sin (self, x):      raise RuntimeError('Operation not defined.')
    def get_cos (self, x):      raise RuntimeError('Operation not defined.')

class MathOperatorFactory(OperatorFactory):
    def get_sqrt(self, x): return math.sqrt(x)
    def get_log (self, x): return math.log(x)
    def get_exp (self, x): return math.exp(x)
    def get_sin (self, x): return math.sin(x)
    def get_cos (self, x): return math.cos(x)

class NumpyOperatorFactory(OperatorFactory):
    def get_sqrt(self, x):
        try: return np.sqrt(x)
        except RuntimeWarning: raise OperationDomainError()
    def get_log (self, x):
        try: return np.log(x)
        except RuntimeWarning: raise OperationDomainError()
    def get_exp (self, x):
        try: return np.exp(x)
        except RuntimeWarning: raise OperationDomainError()
    def get_sin (self, x):
        try: return np.sin(x)
        except RuntimeWarning: raise OperationDomainError()
    def get_cos (self, x):
        try: return np.cos(x)
        except RuntimeWarning: raise OperationDomainError()

class SympyOperatorFactory(OperatorFactory):
    def get_sqrt(self, x): return sympy.sqrt(x)
    def get_log (self, x): return sympy.log (x)
    def get_exp (self, x): return sympy.exp (x)
    def get_sin (self, x): return sympy.sin (x)
    def get_cos (self, x): return sympy.cos (x)


class SyntaxTree:
    OPERATORS = ['*', '/', 'sqrt', 'log', 'exp', 'sin', 'cos']

    def __init__(self) -> None:
        self.children = []
    
    def append(self, subtree):
        self.children.append(subtree)
    
    def lambdify(self) -> str:
        raise RuntimeError('Operation not defined.')
    
    def lambdify_deriv(self) -> str:
        raise RuntimeError('Operation not defined.')
    
    def set_coeffs(self, coeffs, offset:int=0) -> int:  # returns updated offset
        raise RuntimeError('Operation not defined.')
    
    def evaluate(self, x) -> float:
        raise RuntimeError('Operation not defined.')
    
    def evaluate_deriv(self, x) -> float:
        raise RuntimeError('Operation not defined.')

    def tostring(self, id:str=''):
        ans = ''
        for c in self.children:
            ans += f"\t{c.tostring()}\n"
        return ans
    
    def get_depth(self) -> int:
        max_ch_depth = 0
        for c in self.children:
            c_depth = c.get_depth()
            if c_depth > max_ch_depth: max_ch_depth = c_depth
        return max_ch_depth + 1

    def get_ncoeffs(self) -> int:
        raise RuntimeError('Operation not supported.')
    
    @staticmethod
    def get_evaluate_operator_lambda(operator:str, optfact:OperatorFactory):
        if   operator == '*':    return lambda f, g, x: f.evaluate(x) * g.evaluate(x), \
                                        lambda f, g, x: f"({f.lambdify(x)}) * ({g.lambdify(x)})", \
                                        2
        elif operator == '/':    return lambda f, g, x: optfact.get_div(f.evaluate(x), g.evaluate(x)), \
                                        lambda f, g, x: f"({f.lambdify(x)}) / ({g.lambdify(x)})", \
                                        2
        elif operator == 'sqrt': return lambda f, x:    optfact.get_sqrt(f.evaluate(x)), \
                                        lambda f, x:    f"np.sqrt({f.lambdify(x)}, out=None)", \
                                        1
        elif operator == 'log':  return lambda f, x:    optfact.get_log(f.evaluate(x)), \
                                        lambda f, x:    f"np.log({f.lambdify(x)}, out=None)", \
                                        1
        elif operator == 'exp':  return lambda f, x:    optfact.get_exp(f.evaluate(x)), \
                                        lambda f, x:    f"np.exp({f.lambdify(x)}, out=None)", \
                                        1
        elif operator == 'sin':  return lambda f, x:    optfact.get_sin (f.evaluate(x)), \
                                        lambda f, x:    f"np.sin({f.lambdify(x)}, out=None)", \
                                        1
        elif operator == 'cos':  return lambda f, x:    optfact.get_cos(f.evaluate(x)), \
                                        lambda f, x:    f"np.cos({f.lambdify(x)}, out=None)", \
                                        1
        raise RuntimeError(f"Operator {operator} not supported. Please choose from {SyntaxTree.OPERATORS}.")
    
    @staticmethod
    def get_deriv_operator_lambda(operator:str, optfact:OperatorFactory):
        if   operator == '*': return \
            lambda f, g, x: (f.evaluate_deriv(x) * g.evaluate(x)) + (f.evaluate(x) * g.evaluate_deriv(x)), \
            lambda f, g:    f"({f.lambdify_deriv()} * {g.lambdify()}) + ({f.lambdify()} * {g.lambdify_deriv()})", \
            2
        
        elif operator == '/': return \
            lambda f, g, x: optfact.get_div( ((f.evaluate_deriv(x) * g.evaluate(x)) - (f.evaluate(x) * g.evaluate_deriv(x))), (g.evaluate(x)**2) ), \
            lambda f, g:    f"((({f.lambdify_deriv()}) * ({g.lambdify()})) - (({f.lambdify()}) * ({g.lambdify_deriv()}))) / (({g.lambdify()})**2)", \
            2
        
        elif operator == 'sqrt': return \
            lambda f, x: optfact.get_div( f.evaluate_deriv(x), (2 * optfact.get_sqrt(f.evaluate(x))) ), \
            lambda f:    f"({f.lambdify_deriv()}) / (2 * np.sqrt({f.lambdify()}, out=None))", \
            1
        
        elif operator == 'log': return \
            lambda f, x: optfact.get_div( f.evaluate_deriv(x), f.evaluate(x) ), \
            lambda f:    f"({f.lambdify_deriv()}) / ({f.lambdify()})", \
            1
        
        elif operator == 'exp': return \
            lambda f, x: optfact.get_exp(f.evaluate(x)) * f.evaluate_deriv(x), \
            lambda f:    f"np.exp({f.lambdify()}, out=None) * ({f.lambdify_deriv()})", \
            1
        
        elif operator == 'sin': return \
            lambda f, x: optfact.get_cos(f.evaluate(x)) * f.evaluate_deriv(x), \
            lambda f:    f"np.cos({f.lambdify()}, out=None) * ({f.lambdify_deriv()})", \
            1
        elif operator == 'cos': return \
            lambda f, x: -optfact.get_sin(f.evaluate(x)) * f.evaluate_deriv(x), \
            lambda f:    f"-np.sin({f.lambdify()}, out=None) * ({f.lambdify_deriv()})", \
            1

        raise RuntimeError(f"Operator {operator} not supported. Please choose from {SyntaxTree.OPERATORS}.")

    @staticmethod
    def create_random(curr_depth:int, n_coeffs:int, n_coeffs_inner:int, max_depth:int=2, optfact:OperatorFactory=NumpyOperatorFactory()):
        if curr_depth >= max_depth:
            return PolySyntaxTree(n_coeffs)
        
        operator = random.choice(SyntaxTree.OPERATORS + ['inner_poly'])
        
        stree = None
        if operator == 'inner_poly':
            stree = InnerPolySyntaxTree(n_coeffs_inner)
            stree.append( SyntaxTree.create_random(curr_depth+1, n_coeffs, n_coeffs_inner, max_depth, optfact) )

        else:
            stree = OperatorSyntaxTree(operator, optfact)
            for _ in range(stree.arity):
                stree.append( SyntaxTree.create_random(curr_depth+1, n_coeffs, n_coeffs_inner, max_depth, optfact) )
        
        return stree
    
    @staticmethod
    def lambdify(stree, x='x'):
        return eval(f"lambda coeffs, x, out1, out2: {stree.lambdify(x)}")
    
    @staticmethod
    def lambdify_deriv(stree):
        return eval(f"lambda coeffs, x, out1, out2: {stree.lambdify_deriv()}")
        #return eval(f"lambda coeffs, x, out1, out2: {stree.lambdify('(x+0.000001)')} - {stree.lambdify()}")
        
    
class OperatorSyntaxTree(SyntaxTree):
    def __init__(self, operator:str, optfact:OperatorFactory) -> None:
        super().__init__()
        self.operator_str = operator
        self.operator, self.lamb_operator, self.arity = SyntaxTree.get_evaluate_operator_lambda(operator, optfact)
        self.deriv_operator, self.lamb_deriv_operator, self.deriv_arity = SyntaxTree.get_deriv_operator_lambda(operator, optfact)
    
    def append(self, subtree):
        self.children.append(subtree)
    
    def set_coeffs(self, coeffs, offset:int=0) -> int:
        for c in self.children:
            offset = c.set_coeffs(coeffs, offset)
        return offset
    
    def lambdify(self, x='x') -> str:
        if len(self.children) != self.arity: raise RuntimeError('Mismatch with arity.')
        return self.lamb_operator(*self.children, x)
    
    def lambdify_deriv(self) -> str:
        if len(self.children) != self.arity: raise RuntimeError('Mismatch with arity.')
        return self.lamb_deriv_operator(*self.children)
    
    def evaluate(self, x) -> float:
        if len(self.children) != self.arity: raise RuntimeError('Mismatch with arity.')
        #try:
        return self.operator(*self.children, x)
        #except Exception: raise OperationDomainError
    
    def evaluate_deriv(self, x) -> float:
        if len(self.children) != self.deriv_arity: raise RuntimeError('Mismatch with arity.')
        #try:
        return self.deriv_operator(*self.children, x)
        #except Exception: raise OperationDomainError

    def get_ncoeffs(self) -> int:
        n = 0
        for c in self.children: n += c.get_ncoeffs()
        return n
    
    def tostring(self, id:str=''):
        if   self.operator_str == '*': return f"{self.children[0].tostring(id='a')}*{self.children[1].tostring(id='b')}"
        elif self.operator_str == '/': return f"{self.children[0].tostring(id='a')}/{self.children[1].tostring(id='b')}"
        return f"{self.operator_str}({self.children[0].tostring()})"


class PolySyntaxTree(SyntaxTree):
    def __init__(self, n_coeffs:int) -> None:
        super().__init__()
        self.n_coeffs = n_coeffs
        self.coeffs_startidx = None
        self.coeffs_endidx = None
        self.coeffs = np.zeros(n_coeffs)  # in decreasing order of power!
        self.coeffs_deriv = np.zeros(n_coeffs-1)
    
    def append(self, subtree):
        raise RuntimeError('Append not supported on PolySyntaxTree')

    def set_coeffs(self, coeffs, offset:int=0) -> int:
        new_offset = offset + self.n_coeffs
        self.coeffs_startidx = offset
        self.coeffs_endidx = new_offset  # not included
        self.coeffs = coeffs[offset:new_offset]
        self.coeffs_deriv = self.coeffs[:-1] * np.arange(self.n_coeffs-1, 0, -1)
        return new_offset
    
    def lambdify(self, x='x') -> str:
        return f"np.polyval(coeffs[{self.coeffs_startidx}:{self.coeffs_endidx}], {x})"

    def lambdify_deriv(self) -> str:
        return f"np.polyval(coeffs[{self.coeffs_startidx}:{self.coeffs_endidx-1}] * np.arange({self.n_coeffs-1}, 0, -1), x)"
    
    def evaluate(self, x) -> float:
        return np.polyval(self.coeffs, x)

    def evaluate_deriv(self, x) -> float:
        return np.polyval(self.coeffs_deriv, x)
    
    def get_ncoeffs(self) -> int:
        return self.n_coeffs
    
    def tostring(self, id:str=''):
        if self.n_coeffs == 0 or True:
            return 'P' + ('' if id == '' else '_' + id) + '(x)'
        epsilon = 1e-8
        ans = ''
        for deg in range(self.n_coeffs):
            c = self.curr_coeffs[deg]
            if abs(c) < epsilon: continue
            if deg > 0: ans += '+' + f"{c}*x{'**' + str(deg) if deg > 1 else ''}" if abs(1-c) >= epsilon else f"x**{deg}"
            else: ans += f"{c}"
        return ans


class InnerPolySyntaxTree(PolySyntaxTree):
    def __init__(self, n_coeffs:int) -> None:
        super().__init__(n_coeffs)
        self.arity = 1
    
    def append(self, subtree):
        self.children.append(subtree)
    
    def set_coeffs(self, coeffs, offset:int=0) -> int:
        for c in self.children:
            offset = c.set_coeffs(coeffs, offset)
        return super().set_coeffs(coeffs, offset)
    
    def lambdify(self, x='x') -> str:
        if len(self.children) != self.arity: raise RuntimeError('Mismatch with arity.')
        ch_lamb = self.children[0].lambdify(x)
        return f"np.polyval(coeffs[{self.coeffs_startidx}:{self.coeffs_endidx}], {ch_lamb})"

    def lambdify_deriv(self) -> str:
        if len(self.children) != self.arity: raise RuntimeError('Mismatch with arity.')
        ch_lamb = self.children[0].lambdify()
        ch_lamb_deriv = self.children[0].lambdify_deriv()
        return f"np.polyval(coeffs[{self.coeffs_startidx}:{self.coeffs_endidx-1}] * np.arange({self.n_coeffs-1}, 0, -1), {ch_lamb}) * {ch_lamb_deriv}"
    
    def evaluate(self, x) -> float:
        if len(self.children) != self.arity: raise RuntimeError('Mismatch with arity.')
        ch_eval = self.children[0].evaluate(x)
        return np.polyval(self.coeffs, ch_eval)
    
    def evaluate_deriv(self, x) -> float:
        if len(self.children) != self.arity: raise RuntimeError('Mismatch with arity.')
        ch_eval = self.children[0].evaluate(x)
        ch_eval_deriv = self.children[0].evaluate_deriv(x)
        return np.polyval(self.coeffs_deriv, ch_eval) * ch_eval_deriv
    
    def get_ncoeffs(self) -> int:
        n = self.n_coeffs
        for c in self.children: n += c.get_ncoeffs()
        return n
    
    def tostring(self, id:str=''):
        child_str = self.children[0].tostring()

        if self.n_coeffs == 0 or True:
            return 'P' + ('' if id == '' else '_' + id) + f"({child_str})"
        
        epsilon = 1e-8
        ans = ''
        for deg in range(self.n_coeffs):
            c = self.curr_coeffs[deg]
            if abs(c) < epsilon: continue
            if deg > 0: ans += '+' + f"{c}*{child_str}{'**' + str(deg) if deg > 1 else ''}" if abs(1-c) >= epsilon else f"{child_str}**{deg}"
            else: ans += f"{c}"
        return ans
    

def get_ineq_activation(f_x:np.array, th_activ:np.array, sign:np.array, sigma:float=1e12) -> np.array:
    #assert threshold == 0.
    # sigmoid activation
    #if f_x < 0: return np.exp(sigma*f_x) / (1 + np.exp(sigma*f_x))  # two implementations to avoid overflow.
    #return 1 / (1 + np.exp(-sigma*f_x))

    #return np.tanh(sigma * f_x * 0.5) * 0.5 + 0.5
    #return -np.log(1 + np.exp(-f_x -10)) + 1
    #return np.where(f_x * y_activ < 0, f_x + y_activ, y_activ)
    
    #activ = np.empty(f_x.size)
    #for i in range(f_x.size):
    #    if f_x[i] * y_activ[i] < 0: activ[i] = f_x[i] * y_activ[i] + y_activ[i]
    #    else: activ[i] = y_activ[i]
    #return activ #np.where(f_x * y_activ < 0, f_x, y_activ)

    activ = np.empty(f_x.size)
    for i in range(f_x.size):
        if sign[i] > 0:  # > case
            activ[i] = min(th_activ[i], f_x[i])
        else:            # < case
            activ[i] = max(th_activ[i], f_x[i])
    return activ


get_system_tot_time = 0.
get_system_time_count = 0
get_system_setcoeffs_tottime = 0.
get_system_evaluate_tottime = 0.
get_system_evalderiv_tottime = 0.

def get_system(coeffs, stree:SyntaxTree, stree_lamb, stree_lamb_deriv, out1, out2,
               interc_dpx:np.array, interc_dpy:np.array,
               image_activ_sign, deriv_activ_sign,
               image_range, image_activ_range, deriv_range, deriv_activ_range,
               interc_weights:np.array=None):
    global get_system_tot_time
    global get_system_time_count
    global get_system_setcoeffs_tottime
    global get_system_evaluate_tottime
    global get_system_evalderiv_tottime

    start_tottime = time.time()

    start_setcoeffs = time.time()
    get_system_setcoeffs_tottime += time.time() - start_setcoeffs

    is_interc       = image_range[1] - image_range[0]  > 0
    is_deriv_interc = deriv_range[1] - deriv_range[0]  > 0
    is_activ        = image_activ_range[1] - image_activ_range[0]  > 0
    is_deriv_activ  = deriv_activ_range[1] - deriv_activ_range[0]  > 0

    eqs = np.empty(interc_dpx.size)
    offset = 0

    try:

        image_size = image_activ_range[1] - image_range[0]
        if image_size > 0:
            image = stree_lamb(coeffs, interc_dpx[image_range[0]:image_activ_range[1]], out1[:image_size], out2[:image_size])
            image_activ_size = image_activ_range[1] - image_activ_range[0]
            if image_activ_size > 0:
                image[image_activ_range[0]:image_activ_range[1]] = get_ineq_activation(image[image_activ_range[0]:image_activ_range[1]], interc_dpy[image_activ_range[0]:image_activ_range[1]], image_activ_sign)
            np.subtract( image, interc_dpy[image_range[0]:image_activ_range[1]], out=eqs[image_range[0]:image_activ_range[1]] )

        deriv_size = deriv_activ_range[1] - deriv_range[0]
        if deriv_size > 0:
            deriv = stree_lamb_deriv(coeffs, interc_dpx[deriv_range[0]:deriv_activ_range[1]], out1[:deriv_size], out2[:deriv_size])
            deriv_activ_size = deriv_activ_range[1] - deriv_activ_range[0]
            if deriv_activ_size > 0:
                deriv[deriv_activ_range[0]:deriv_activ_range[1]] = get_ineq_activation(deriv[deriv_activ_range[0]:deriv_activ_range[1]], interc_dpy[deriv_activ_range[0]:deriv_activ_range[1]], deriv_activ_sign)
            np.subtract( deriv, interc_dpy[deriv_range[0]:deriv_activ_range[1]], out=eqs[deriv_range[0]:deriv_activ_range[1]] )

        if interc_weights is not None:
                eqs *= interc_weights
        
    except RuntimeWarning: raise OperationDomainError()
    
    """for (deg, sign, dp) in interc_dp[:max(len(coeffs), 1)]:  #random.choices(interc_dp, k=max(len(coeffs), 100)):

        eq = None
        if   deg == 0:
            start_evaluate = time.time()
            eq = stree.evaluate(dp.x)
            get_system_evaluate_tottime += time.time() - start_evaluate
        
        elif deg == 1:
            start_evalderiv = time.time()
            eq = stree.evaluate_deriv(dp.x)
            get_system_evalderiv_tottime += time.time() - start_evalderiv
        
        else: raise RuntimeError(f"Degree {deg} not supported.")

        y_target = dp.y
        if   sign == '>':
            eq = get_ineq_activation(eq, y_target)
            y_target = 1.
        elif sign == '<':
            eq = get_ineq_activation(eq, y_target)
            y_target = 0.
        
        eq -= y_target
        eqs.append(eq)"""
    
    get_system_tot_time += time.time() - start_tottime
    get_system_time_count += 1

    return eqs


get_func_total_time = 0.
get_func_total_calls = 0

def get_func(coeffs, stree:SyntaxTree, interc_dpx:np.array, interc_dpy:np.array):
    global get_func_total_time
    global get_func_total_calls
    start_time = time.time()

    stree.set_coeffs(coeffs)
    y = np.sum( (stree.evaluate(interc_dpx) - interc_dpy) ** 2 )

    get_func_total_time += time.time() - start_time
    get_func_total_calls += 1
    return y

def tune_syntax_tree(S:dataset.Dataset, stree:SyntaxTree, stree_lamb, stree_lamb_deriv, out1, out2,
                     interc_dpx:np.array, interc_dpy:np.array,
                     image_activ_sign, deriv_activ_sign,
                     image_range, image_activ_range, deriv_range, deriv_activ_range,
                     interc_weights:np.array=None,
                     coeffs_0:np.array=None,
                     verbose:bool=True, maxiter:int=0) -> dict:

    start_time = time.time()
    tot_coeffs = stree.get_ncoeffs()
    
    if coeffs_0 is None: coeffs_0 = np.array( [random.uniform(1., 5.) for _ in range(tot_coeffs)] )
    #res, _, _, msg = fsolve( get_system, coeffs_0, args=(stree, interc_dp), full_output=True )
    res = root(
        get_system, coeffs_0,
        args=(stree, stree_lamb, stree_lamb_deriv, out1, out2, interc_dpx, interc_dpy, image_activ_sign, deriv_activ_sign, image_range, image_activ_range, deriv_range, deriv_activ_range, interc_weights),
        method='lm', options={'maxiter':maxiter} )
    #res = minimize( get_func, coeffs_0, args=(stree, interc_dpx, interc_dpy), method='BFGS', options={'maxiter':maxiter} )
    if verbose: print(res)
    #print(res.fun)
    sol = res.x
    #close_res = np.isclose(get_system(res, stree, stree_lamb, stree_lamb_deriv, interc_dp), [0. for _ in range(tot_coeffs)])
    close_sol = False #np.isclose(get_system(sol, stree, stree_lamb, stree_lamb_deriv, interc_dp), [0. for _ in range(len(interc_dp))])
    root_found = np.all(close_sol == True)
    
    if verbose:
        #print(f"Message: {msg}")
        print(f"Solution: {sol}")
        print(f"Is close: {close_sol}")

    is_interc       = image_range[1] - image_range[0]  > 0
    is_deriv_interc = deriv_range[1] - deriv_range[0]  > 0
    is_activ        = image_activ_range[1] - image_activ_range[0]  > 0
    is_deriv_activ  = deriv_activ_range[1] - deriv_activ_range[0]  > 0

    coeffs = sol
    sse = 0
    sse_size = interc_dpx.size
    try:
        
        eqs = get_system(coeffs, stree, stree_lamb, stree_lamb_deriv, out1, out2,
                         interc_dpx, interc_dpy, image_activ_sign, deriv_activ_sign,
                         image_range, image_activ_range, deriv_range, deriv_activ_range,
                         interc_weights)

        sse = np.sum( eqs ** 2 )
        
        """image_size = image_activ_range[1] - image_range[0]
        if image_size > 0:
            image = stree_lamb(coeffs, interc_dpx[image_range[0]:image_activ_range[1]], out1[:image_size], out2[:image_size])
            image_activ_size = image_activ_range[1] - image_activ_range[0]
            if image_activ_size > 0:
                image[image_activ_range[0]:image_activ_range[1]] = get_ineq_activation(image[image_activ_range[0]:image_activ_range[1]], interc_dpy[image_activ_range[0]:image_activ_range[1]], image_activ_sign)
            sse += np.sum( np.subtract( image, interc_dpy[image_range[0]:image_activ_range[1]] ) ** 2 )

        deriv_size = deriv_activ_range[1] - deriv_range[0]
        if deriv_size > 0:
            deriv = stree_lamb_deriv(coeffs, interc_dpx[deriv_range[0]:deriv_activ_range[1]], out1[:deriv_size], out2[:deriv_size])
            deriv_activ_size = deriv_activ_range[1] - deriv_activ_range[0]
            if deriv_activ_size > 0:
                deriv[deriv_activ_range[0]:deriv_activ_range[1]] = get_ineq_activation(deriv[deriv_activ_range[0]:deriv_activ_range[1]], interc_dpy[deriv_activ_range[0]:deriv_activ_range[1]], deriv_activ_sign)
            sse += np.sum( np.subtract( deriv, interc_dpy[deriv_range[0]:deriv_activ_range[1]] ) ** 2 )"""
        
        
        """if is_interc:
            sse = np.sum( (stree_lamb(coeffs, interc_dpx, out1[:interc_dpx.size], out2[:interc_dpx.size]) - interc_dpy) ** 2 )
            sse_size = interc_dpx.size

        if is_deriv_interc:
            sse += np.sum( (stree_lamb_deriv(coeffs, deriv_interc_dpx, out1[:deriv_interc_dpx.size], out2[:deriv_interc_dpx.size]) - deriv_interc_dpy) ** 2 )
            sse_size += deriv_interc_dpx.size

        if is_activ:
            sse += np.sum( (get_ineq_activation(stree_lamb(coeffs, activ_dpx, out1[:activ_dpx.size], out2[:activ_dpx.size])) - activ_dpy) ** 2 )
            sse_size += activ_dpx.size
        
        if is_deriv_activ:
            sse += np.sum( (get_ineq_activation(stree_lamb_deriv(coeffs, deriv_activ_dpx, out1[:deriv_activ_dpx.size], out2[:deriv_activ_dpx.size])) - deriv_activ_dpy) ** 2 )
            sse_size += deriv_activ_dpx.size"""
        
    except RuntimeWarning: raise OperationDomainError()

    """for (deg, sign, dp) in interc_dp:
        stree.set_coeffs(sol)
        
        y = None
        if   deg == 0: y = stree.evaluate(dp.x)
        elif deg == 1: y = stree.evaluate_deriv(dp.x)
        else: raise RuntimeError(f"Degree {deg} not supported.")

        Y.append(y)
        error = y - dp.y
        if   sign == '>': error = min(0, error)
        elif sign == '<': error = max(0, error)
        sse += error ** 2"""
    mse = sse / sse_size
    
    elapsed_time = time.time() - start_time
    return { 'sol': sol, 'sse': sse, 'mse': mse, 'root_found': root_found, 'elapsed_time': elapsed_time }


def get_data_interc_points(S:dataset.Dataset, sample_size:int=0):
    data = S.data if sample_size == 0 else random.choices(S.data, k=sample_size)
    data_interc_dpx = np.array( [dp.x for dp in data] )
    data_interc_dpy = np.array( [dp.y for dp in data] )
    return data_interc_dpx, data_interc_dpy


def get_knowledge_interc_points(S:dataset.Dataset, bin_bounds:dict=None, sample_size:int=20):
    interc_dpx = []
    interc_dpy = []
    image_activ_sign = []
    deriv_activ_sign = []
    image_range       = [0, 0]
    image_activ_range = [0, 0]
    deriv_range       = [0, 0]
    deriv_activ_range = [0, 0]

    offset = 0
    knowledge:dataset.DataKnowledge = S.knowledge
    if bin_bounds is not None:
        knowledge = dataset.DataKnowledge(None)
        for x in bin_bounds.keys():
            knowledge.add_sign(0, x, x, '+', bin_bounds[x]['lower'])
            knowledge.add_sign(0, x, x, '-', bin_bounds[x]['upper'])

    if 0 in knowledge.derivs.keys():
        interc_dpx += [dp.x for dp in knowledge.derivs[0]] #* sample_size
        interc_dpy += [dp.y for dp in knowledge.derivs[0]] #* sample_size
        image_range[0] = offset
        image_range[1] = offset + len(knowledge.derivs[0]) #* sample_size
        offset = image_range[1]
    else:
        image_range[0] = offset
        image_range[1] = offset
    
    if 0 in knowledge.sign.keys() and len(knowledge.sign[0]) > 0:
        image_activ_range[0] = offset
        for (l,u,sign,th) in knowledge.sign[0]:
            sign_val = 1. if sign == '+' else -1.
            actual_sample_size = sample_size if l < u else 1
            interc_dpx += [x for x in np.linspace(l, u, actual_sample_size)]
            interc_dpy += [th for _ in range(actual_sample_size)]
            image_activ_sign += [sign_val for _ in range(actual_sample_size)]
            image_activ_range[1] = offset + actual_sample_size
            offset = image_activ_range[1]
    else:
        image_activ_range[0] = offset
        image_activ_range[1] = offset
    

    if 1 in knowledge.derivs.keys():
        interc_dpx += [dp.x for dp in knowledge.derivs[1]] #* sample_size
        interc_dpy += [dp.y for dp in knowledge.derivs[1]] #* sample_size
        deriv_range[0] = offset
        deriv_range[1] = offset + len(knowledge.derivs[1]) #* sample_size
        offset = deriv_range[1]
    else:
        deriv_range[0] = offset
        deriv_range[1] = offset
    
    if 1 in knowledge.sign.keys() and len(knowledge.sign[1]) > 0:
        deriv_activ_range[0] = offset
        for (l,u,sign,th) in knowledge.sign[1]:
            sign_val = 1. if sign == '+' else -1.
            actual_sample_size = sample_size if l < u else 1
            interc_dpx += [x for x in np.linspace(l, u, actual_sample_size)]
            interc_dpy += [th for _ in range(actual_sample_size)]
            deriv_activ_sign += [sign_val for _ in range(actual_sample_size)]
            deriv_activ_range[1] = offset + actual_sample_size
            offset = deriv_activ_range[1]
    else:
        deriv_activ_range[0] = offset
        deriv_activ_range[1] = offset

  
    print(f"{image_range}, {image_activ_range}, {deriv_range}, {deriv_activ_range}")
    return np.array( interc_dpx ), np.array( interc_dpy ), \
           np.array( image_activ_sign ), np.array( deriv_activ_sign ), \
           image_range, image_activ_range, deriv_range, deriv_activ_range


def __print_header(header:str):
    print('='*10 + ' ' + header + ' ' + '='*10)

def infer_syntaxtree(S:dataset.Dataset, bin_bounds:dict=None, max_degree:int=2, max_degree_inner:int=1, max_depth:int=2, trials:int=10, pk_pressure:float=0.8):
    global get_system_tot_time
    global get_system_time_count
    global get_system_setcoeffs_tottime
    global get_system_evaluate_tottime
    global get_system_evalderiv_tottime
    
    global get_func_total_time
    global get_func_total_calls

    get_system_tot_time = 0.
    get_system_time_count = 0
    get_system_setcoeffs_tottime = 0.
    get_system_evaluate_tottime = 0.
    get_system_evalderiv_tottime = 0.

    get_func_total_time = 0.
    get_func_total_calls = 0

    n_coeffs = max_degree + 1
    n_coeffs_inner = max_degree_inner + 1

    data_interc_dpx, data_interc_dpy = get_data_interc_points(S)
    knowledge_interc_dpx, knowledge_interc_dpy, \
    image_activ_sign, deriv_activ_sign, \
    image_range, image_activ_range, deriv_range, deriv_activ_range = get_knowledge_interc_points(S, bin_bounds=bin_bounds)

    print(f"Total data constraints:      {data_interc_dpx.size}")
    print(f"Total knowledge constraints: {knowledge_interc_dpx.size}")

    data_out1      = np.empty(data_interc_dpx.size)  # TODO: can be obtimized (just use max)
    data_out2      = np.empty(data_interc_dpx.size)
    knowledge_out1 = np.empty(knowledge_interc_dpx.size)
    knowledge_out2 = np.empty(knowledge_interc_dpx.size)

    best_stree = None
    best_data_tuning_report = None
    best_knowledge_image_tuning_report = None
    best_knowledge_deriv_tuning_report = None
    best_fitness = None

    data_elapsed_time = 0.
    knowledge_elapsed_time = 0.

    n_restarts = 5#5
    n_actual_restarts = 0

    __print_header('Syntax Tree Inference')

    for _ in range(trials):
        try:
            depth = random.randint(1, max_depth)
            stree = SyntaxTree.create_random(1, n_coeffs, n_coeffs_inner, depth)

            stree.set_coeffs(np.zeros(stree.get_ncoeffs()))
            stree_lamb = SyntaxTree.lambdify(stree)
            stree_lamb_deriv = SyntaxTree.lambdify_deriv(stree)

            tree_found = False
            #if type(stree) is OperatorSyntaxTree and stree.operator_str == '/' and stree.arity == 2 and type(stree.children[0]) is PolySyntaxTree and type(stree.children[1]) is PolySyntaxTree:
            #    print("TREE FOUND")
            #    tree_found = True
            if type(stree) is InnerPolySyntaxTree and type(stree.children[0]) is OperatorSyntaxTree and stree.children[0].operator_str == 'sin' and \
                type(stree.children[0].children[0]) is PolySyntaxTree:
                print("TREE FOUND " + stree.tostring())
                tree_found = True

            for _ in range(n_restarts):
                
                ## data tuning
                data_tuning_report      = tune_syntax_tree(
                        S, stree, stree_lamb, stree_lamb_deriv, data_out1, data_out2,
                        data_interc_dpx, data_interc_dpy,
                        None, None,
                        [0, data_interc_dpx.size], [data_interc_dpx.size, data_interc_dpx.size], [data_interc_dpx.size, data_interc_dpx.size], [data_interc_dpx.size, data_interc_dpx.size],
                        verbose=False, maxiter=200)
                
                # image tuning (PK)
                lastidx = image_activ_range[1]
                knowledge_image_tuning_report = tune_syntax_tree(
                        S, stree, stree_lamb, stree_lamb_deriv, knowledge_out1, knowledge_out2,
                        #knowledge_interc_dpx, knowledge_interc_dpy,
                        #image_range, image_activ_range, deriv_range, deriv_activ_range,
                        knowledge_interc_dpx[:lastidx], knowledge_interc_dpy[:lastidx],
                        image_activ_sign, None,
                        image_range, image_activ_range, [lastidx,lastidx], [lastidx,lastidx],
                        verbose=False, maxiter=150) if pk_pressure > 0 else None

                # deriv tuning (PK)
                firstidx = deriv_range[0]
                knowledge_deriv_tuning_report = None
                """tune_syntax_tree(
                        S, stree, stree_lamb, stree_lamb_deriv, knowledge_out1, knowledge_out2,
                        knowledge_interc_dpx[firstidx:], knowledge_interc_dpy[firstidx:],
                        None, deriv_activ_sign,
                        [0,0], [0,0], [deriv_range[0]-firstidx,deriv_range[1]-firstidx], [deriv_activ_range[0]-firstidx, deriv_activ_range[1]-firstidx],
                        coeffs_0=knowledge_image_tuning_report['sol'],
                        verbose=False, maxiter=50) if pk_pressure > 0 else None"""
                
                data_fitness = data_tuning_report['mse']
                knowledge_fitness = max( 0 if knowledge_image_tuning_report is None else knowledge_image_tuning_report['mse'],
                                         0 if knowledge_deriv_tuning_report is None else knowledge_deriv_tuning_report['mse'] )
                
                fitness = ((1-pk_pressure) * data_fitness) + (pk_pressure * knowledge_fitness)

                if tree_found:
                    pass#print(f"Tree found fitness: {data_tuning_report['mse'] } [data], {knowledge_tuning_report['mse']} [knowledge], {fitness} [fitness]")

                if best_stree is None or fitness < best_fitness:
                    #print(f"Update best fitness from {best_fitness} to {fitness}")
                    best_stree = stree
                    best_data_tuning_report = data_tuning_report
                    best_knowledge_image_tuning_report = knowledge_image_tuning_report
                    best_knowledge_deriv_tuning_report = knowledge_deriv_tuning_report
                    best_fitness = fitness
                
                data_elapsed_time += data_tuning_report['elapsed_time']
                if knowledge_image_tuning_report is not None: knowledge_elapsed_time += knowledge_image_tuning_report['elapsed_time']
                if knowledge_deriv_tuning_report is not None: knowledge_elapsed_time += knowledge_deriv_tuning_report['elapsed_time']

                n_actual_restarts += 1
                #if fitness <= 0.01: break

        except OperationDomainError:  # domain error
            pass
    
    print(f"\nData tuning (avg time):      {int((data_elapsed_time / (trials*n_actual_restarts)) * 1e3)} ms")
    print(f"Knowledge tuning (avg time):   {int((knowledge_elapsed_time / (trials*n_actual_restarts)) * 1e3)} ms")
    print(f"Data tuning (total time):      {int(data_elapsed_time * 1e3)} ms")
    print(f"Knowledge tuning (total time): {int(knowledge_elapsed_time * 1e3)} ms")

    """print(f"\nGet func (total time): {int((get_func_total_time) * 1e3)} ms")
    print(f"Get func (total calls):  {get_func_total_calls}")"""

    print(f"\nGet system (avg time): {int((get_system_tot_time / get_system_time_count) * 1e3)} ms")
    print(f"Get system (total time): {int((get_system_tot_time) * 1e3)} ms")
    print(f"Get system (total calls): {get_system_time_count}")

    print(f"\nSetcoeffs (total time): {int((get_system_setcoeffs_tottime) * 1e3)} ms")
    """print(f"Evaluate (total time): {int((get_system_evaluate_tottime) * 1e3)} ms")
    print(f"Evalderiv (total time): {int((get_system_evalderiv_tottime) * 1e3)} ms")"""

    return best_stree, best_data_tuning_report, best_knowledge_image_tuning_report, best_knowledge_deriv_tuning_report


"""def infer_poly(S:dataset.Dataset, max_degree:int=2, comb_func=lambda a,b : a+b):
    n_coeffs = max_degree+1
    coeffs_a = sympy.symbols(f"a0:{n_coeffs}")
    coeffs_b = sympy.symbols(f"b0:{n_coeffs}")
    coeffs = coeffs_a + coeffs_b

    eqs = []
    inter_points = []
    for _ in range(n_coeffs*2):
        
        dp = random.choice(S.data[30:70])
        inter_points.append(dp)
        poly_a = 0.
        poly_b = 0.

        for deg in range(n_coeffs):
            poly_a += coeffs_a[deg] * (dp.x ** deg)
            poly_b += coeffs_b[deg] * (dp.x ** deg)
        
        eq = comb_func(poly_a, poly_b)
        eq -= dp.y
        eqs.append(eq)
    
    for _ in range(10):
        coeffs_0 = tuple( [random.uniform(1., 5.) for _ in range(n_coeffs*2)] )
        print(f"EQS: {eqs}\nCOEFFS: {coeffs}\nCOEFFS0: {coeffs_0}")
        #res = sympy.nsolve(eqs, coeffs_a + coeffs_b, coeffs_0, verify=True)
        res = sympy.solve(eqs, *(coeffs))
        print(f"Result: {res}")
        if len(res) > 0: break

    def poly(coeffs, x:float) -> float:
        y = 0.
        for deg in range(len(coeffs)):
            y += coeffs[deg] * (x**deg)
        return y

    #res_coeffs = [c for c in res]
    res_coeffs = []
    for c in coeffs:
        c_val = 1.

        if c in res.keys():
            fcs_val = [(fcs, 1.) for fcs in res[c].free_symbols]
            c_val = res[c].subs(fcs_val)

        res_coeffs.append(c_val)
    
    alphas = []
    betas  = []
    for dp in S.data:
        alphas.append( poly(res_coeffs[0:n_coeffs], dp.x) )
        betas .append( poly(res_coeffs[n_coeffs:n_coeffs*2], dp.x) )
    
    return alphas, betas, inter_points
"""


def enhance_syntax_tree(stree:SyntaxTree, S:dataset.Dataset, sample_size:int, bin_bounds:dict=None, n_restarts:int=10,
                        data_weight:float=1, knowledge_weight:float=1):
    
    __print_header('Syntax Tree Enhancement')

    data_interc_dpx, data_interc_dpy = get_data_interc_points(S, sample_size=0)
    knowledge_interc_dpx, knowledge_interc_dpy, \
    image_activ_sign, deriv_activ_sign, \
    image_range, image_activ_range, deriv_range, deriv_activ_range = get_knowledge_interc_points(S, bin_bounds=bin_bounds, sample_size=sample_size)

    #knowledge_interc_dpx = np.repeat( knowledge_interc_dpx, sample_size )
    #knowledge_interc_dpy = np.repeat( knowledge_interc_dpy, sample_size )
    #knowledge_deriv_interc_dpx = np.repeat( knowledge_deriv_interc_dpx, sample_size )
    #knowledge_deriv_interc_dpy = np.repeat( knowledge_deriv_interc_dpy, sample_size )

    interc_dpx = np.concatenate( (data_interc_dpx, knowledge_interc_dpx) )
    interc_dpy = np.concatenate( (data_interc_dpy, knowledge_interc_dpy) )

    image_range[0] = 0
    image_range[1] = interc_dpx.size
    image_activ_range[0] += data_interc_dpx.size
    image_activ_range[1] += data_interc_dpx.size
    deriv_range[0] += data_interc_dpx.size
    deriv_range[1] += data_interc_dpx.size
    deriv_activ_range[0] += data_interc_dpx.size
    deriv_activ_range[1] += data_interc_dpx.size

    interc_weights = np.array( ([data_weight] * data_interc_dpx.size) + ([knowledge_weight] * knowledge_interc_dpx.size) )

    stree.set_coeffs(np.zeros(stree.get_ncoeffs()))
    stree_lamb = SyntaxTree.lambdify(stree)
    stree_lamb_deriv = SyntaxTree.lambdify_deriv(stree)

    out1 = np.empty(interc_dpx.size)  # TODO: can be obtimized (just use max)
    out2 = np.empty(interc_dpx.size)

    #interc_dpx = data_interc_dpx
    #interc_dpy = data_interc_dpy

    best_tuning_report = None

    for i_restart in range(n_restarts):
        tuning_report = tune_syntax_tree(S, stree, stree_lamb, stree_lamb_deriv, out1, out2,
                                         interc_dpx, interc_dpy,  # merge data+knowledge (0th deriv)
                                         image_activ_sign, deriv_activ_sign,
                                         image_range, image_activ_range, deriv_range, deriv_activ_range,
                                         interc_weights,
                                         verbose=False, maxiter=500)  # no limit for maxiter
        if best_tuning_report is None or tuning_report['mse'] < best_tuning_report['mse']:
            print(f"[Restart #{i_restart+1}] MSE improvement from {None if best_tuning_report is None else best_tuning_report['mse']} to {tuning_report['mse']}")
            best_tuning_report = tuning_report
        else:
            print(f"[Restart #{i_restart+1}] No improvement.")
        
        #if best_tuning_report['mse'] < 0.01: break

    print(f"Training MSE: {best_tuning_report['mse']}")

    return best_tuning_report


def test_syntax_tree(stree:SyntaxTree, S:dataset.Dataset, sample_size:int, pk_epsilon:float=1e-8):

    __print_header('Syntax Tree Testing')

    test_sse = 0.
    test_r2  = 0.
    for dp in S.test:
        test_sse += (stree.evaluate(dp.x) - dp.y) ** 2
    test_mse = test_sse / len(S.test)
    test_r2 = 1 - (test_sse / S.test_sst)

    pk_sat_count = 0
    pk_sat_size  = 0
    for deg in S.knowledge.derivs:
        if deg >= 2: continue
        for dp in S.knowledge.derivs[deg]:
            resid = abs((stree.evaluate(dp.x) if deg == 0 else stree.evaluate_deriv(dp.x)) - dp.y)
            if resid < pk_epsilon:
                pk_sat_count += 1
            pk_sat_size += 1
    for deg in S.knowledge.sign:
        if deg >= 2: continue
        for (l,u,sign,th) in S.knowledge.sign[deg]:
            for x in np.linspace(l, u, sample_size):
                tree_eval = (stree.evaluate(x) if deg == 0 else stree.evaluate_deriv(x))
                if (sign == '+' and tree_eval >= th) or (sign == '-' and tree_eval <= th): pk_sat_count += 1
                pk_sat_size += 1

    return test_mse, test_r2, pk_sat_count/pk_sat_size, pk_sat_count, pk_sat_size


