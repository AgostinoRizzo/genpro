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
    def get_sqrt(self, x): raise RuntimeError('Operation not defined.')
    def get_log (self, x): raise RuntimeError('Operation not defined.')
    def get_exp (self, x): raise RuntimeError('Operation not defined.')
    def get_sin (self, x): raise RuntimeError('Operation not defined.')
    def get_cos (self, x): raise RuntimeError('Operation not defined.')

class MathOperatorFactory:
    def get_sqrt(self, x): return math.sqrt(x)
    def get_log (self, x): return math.log(x)
    def get_exp (self, x): return math.exp(x)
    def get_sin (self, x): return math.sin(x)
    def get_cos (self, x): return math.cos(x)

class NumpyOperatorFactory:
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

class SympyOperatorFactory:
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
        if   operator == '*':    return lambda f, g, x: f.evaluate(x) * g.evaluate(x), 2
        elif operator == '/':    return lambda f, g, x: f.evaluate(x) / g.evaluate(x), 2
        elif operator == 'sqrt': return lambda f, x: optfact.get_sqrt(f.evaluate(x)),  1
        elif operator == 'log':  return lambda f, x: optfact.get_log (f.evaluate(x)),  1
        elif operator == 'exp':  return lambda f, x: optfact.get_exp (f.evaluate(x)),  1
        elif operator == 'sin':  return lambda f, x: optfact.get_sin (f.evaluate(x)),  1
        elif operator == 'cos':  return lambda f, x: optfact.get_cos (f.evaluate(x)),  1
        raise RuntimeError(f"Operator {operator} not supported. Please choose from {SyntaxTree.OPERATORS}.")
    
    @staticmethod
    def get_deriv_operator_lambda(operator:str, optfact:OperatorFactory):
        if   operator == '*':    return lambda f, g, x: \
            (f.evaluate_deriv(x) * g.evaluate(x)) + (f.evaluate(x) * g.evaluate_deriv(x)), 2
        
        elif operator == '/':    return lambda f, g, x: \
            ((f.evaluate_deriv(x) * g.evaluate(x)) - (f.evaluate(x) * g.evaluate_deriv(x))) / (g.evaluate(x)**2), 2
        
        elif operator == 'sqrt': return lambda f, x: \
            f.evaluate_deriv(x) / (2 * optfact.get_sqrt(f.evaluate(x))), 1
        
        elif operator == 'log':  return lambda f, x: \
            f.evaluate_deriv(x) / f.evaluate(x), 1
        
        elif operator == 'exp':  return lambda f, x: \
            optfact.get_exp(f.evaluate(x)) * f.evaluate_deriv(x), 1
        
        elif operator == 'sin':  return lambda f, x: optfact.get_cos(f.evaluate(x)) * f.evaluate_deriv(x), 1
        elif operator == 'cos':  return lambda f, x: -optfact.get_sin(f.evaluate(x)) * f.evaluate_deriv(x), 1

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
        
    
class OperatorSyntaxTree(SyntaxTree):
    def __init__(self, operator:str, optfact:OperatorFactory) -> None:
        super().__init__()
        self.operator_str = operator
        self.operator, self.arity = SyntaxTree.get_evaluate_operator_lambda(operator, optfact)
        self.deriv_operator, self.deriv_arity = SyntaxTree.get_deriv_operator_lambda(operator, optfact)
    
    def append(self, subtree):
        self.children.append(subtree)
    
    def set_coeffs(self, coeffs, offset:int=0) -> int:
        for c in self.children:
            offset = c.set_coeffs(coeffs, offset)
        return offset
    
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
        self.curr_coeffs = []
    
    def append(self, subtree):
        raise RuntimeError('Append not supported on PolySystaxTree')

    def set_coeffs(self, coeffs, offset:int=0) -> int:
        new_offset = offset + self.n_coeffs
        self.curr_coeffs = coeffs[offset:new_offset]
        return new_offset
    
    def evaluate(self, x) -> float:
        y = 0.
        for deg in range(self.n_coeffs):
            y += self.curr_coeffs[deg] * (x ** deg)
        return y

    def evaluate_deriv(self, x) -> float:
        y = 0.
        for deg in range(1, self.n_coeffs):
            y += self.curr_coeffs[deg] * deg * (x ** (deg-1))
        return y
    
    def get_ncoeffs(self) -> int:
        return self.n_coeffs
    
    def tostring(self, id:str=''):
        if len(self.curr_coeffs) == 0 or True:
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
        new_offset = offset + self.n_coeffs
        self.curr_coeffs = coeffs[offset:new_offset]
        return new_offset
    
    def evaluate(self, x) -> float:
        if len(self.children) != self.arity: raise RuntimeError('Mismatch with arity.')
        y = 0.
        ch_eval = self.children[0].evaluate(x)
        for deg in range(self.n_coeffs):
            y += self.curr_coeffs[deg] * (ch_eval ** deg)
        return y
    
    def evaluate_deriv(self, x) -> float:
        if len(self.children) != self.arity: raise RuntimeError('Mismatch with arity.')
        y = 0.
        ch_eval = self.children[0].evaluate(x)
        ch_eval_deriv = self.children[0].evaluate_deriv(x)
        for deg in range(1, self.n_coeffs):
            y += self.curr_coeffs[deg] * deg * (ch_eval ** (deg-1)) * ch_eval_deriv
        return y
    
    def get_ncoeffs(self) -> int:
        n = self.n_coeffs
        for c in self.children: n += c.get_ncoeffs()
        return n
    
    def tostring(self, id:str=''):
        child_str = self.children[0].tostring()

        if len(self.curr_coeffs) == 0 or True:
            return 'P' + ('' if id == '' else '_' + id) + f"({child_str})"
        
        epsilon = 1e-8
        ans = ''
        for deg in range(self.n_coeffs):
            c = self.curr_coeffs[deg]
            if abs(c) < epsilon: continue
            if deg > 0: ans += '+' + f"{c}*{child_str}{'**' + str(deg) if deg > 1 else ''}" if abs(1-c) >= epsilon else f"{child_str}**{deg}"
            else: ans += f"{c}"
        return ans
    

def get_ineq_activation(f_x:float, threshold:float=0., sigma:float=1e3) -> float:
    assert threshold == 0.
    # sigmoid activation
    #if f_x < 0: return np.exp(sigma*f_x) / (1 + np.exp(sigma*f_x))  # two implementations to avoid overflow.
    #return 1 / (1 + np.exp(-sigma*f_x))
    return np.tanh(sigma * f_x * 0.5) * 0.5 + 0.5


get_system_tot_time = 0.
get_system_time_count = 0
get_system_setcoeffs_tottime = 0.
get_system_evaluate_tottime = 0.
get_system_evalderiv_tottime = 0.

def get_system(coeffs, stree:SyntaxTree,
               interc_dpx:np.array, interc_dpy:np.array, activ_dpx:np.array=None, activ_dpy:np.array=None,):
    global get_system_tot_time
    global get_system_time_count
    global get_system_setcoeffs_tottime
    global get_system_evaluate_tottime
    global get_system_evalderiv_tottime

    start_tottime = time.time()
    eqs = []

    start_setcoeffs = time.time()
    stree.set_coeffs(coeffs)
    get_system_setcoeffs_tottime += time.time() - start_setcoeffs

    eqs = stree.evaluate(interc_dpx) - interc_dpy

    if activ_dpx is not None and activ_dpy is not None:
       eqs = np.concatenate( (eqs, get_ineq_activation(stree.evaluate(activ_dpx)) - activ_dpy) )
    
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

def tune_syntax_tree(S:dataset.Dataset, stree:SyntaxTree,
                     interc_dpx:np.array, interc_dpy:np.array, activ_dpx:np.array=None, activ_dpy:np.array=None,
                     verbose:bool=True, maxiter:int=0) -> dict:

    start_time = time.time()
    tot_coeffs = stree.get_ncoeffs()
    
    coeffs_0 = np.array( [random.uniform(1., 5.) for _ in range(tot_coeffs)] )
    #res, _, _, msg = fsolve( get_system, coeffs_0, args=(stree, interc_dp), full_output=True )
    res = root(
        get_system, coeffs_0,
        args=(stree, interc_dpx, interc_dpy, activ_dpx, activ_dpy),
        method='lm', options={'maxiter':maxiter} )
    #res = minimize( get_func, coeffs_0, args=(stree, interc_dpx, interc_dpy), method='BFGS', options={'maxiter':maxiter} )
    #print(res)
    #print(res.fun)
    sol = res.x
    #close_res = np.isclose(get_system(res, stree, interc_dp), [0. for _ in range(tot_coeffs)])
    close_sol = False #np.isclose(get_system(sol, stree, interc_dp), [0. for _ in range(len(interc_dp))])
    root_found = np.all(close_sol == True)
    
    if verbose:
        #print(f"Message: {msg}")
        print(f"Solution: {sol}")
        print(f"Is close: {close_sol}")

    stree.set_coeffs(sol)
    sse_size = 0
    try:
        sse = np.sum( (stree.evaluate(interc_dpx) - interc_dpy) ** 2 )
        sse_size = len(interc_dpx)
        if activ_dpx is not None and activ_dpy is not None:
            sse += np.sum( (get_ineq_activation(stree.evaluate(activ_dpx)) - activ_dpy) ** 2 )
            sse_size += len(activ_dpx)
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


def get_knowledge_interc_points(S:dataset.Dataset) -> list:
    interc_dpx = []
    interc_dpy = []
    activ_dpx  = []
    activ_dpy  = []

    for deg in S.knowledge.derivs.keys():
        interc_dpx += [dp.x for dp in S.knowledge.derivs[deg]]
        interc_dpy += [dp.y for dp in S.knowledge.derivs[deg]]
        #for dp in S.knowledge.derivs[deg]:
            #interc_dp.append( (deg, '=', dp) )

    for deg in S.knowledge.sign.keys():
        for (l,u,sign) in S.knowledge.sign[deg]:
            dpy = 1. if sign == '+' else 0.
            activ_dpx += [x for x in np.linspace(l, u, 20)]
            activ_dpy += [dpy for _ in range(20)]
            #for x in np.linspace(l, u, 10):
            #    interc_dp.append( (deg, '>' if sign == '+' else '<', dataset.DataPoint(x, 0)) )"""
    
    return np.array( interc_dpx ), np.array( interc_dpy ), np.array( activ_dpx ), np.array( activ_dpy )


def infer_syntaxtree(S:dataset.Dataset, max_degree:int=2, max_degree_inner:int=1, max_depth:int=2, trials:int=10):
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

    data_interc_dp  = S.data
    data_interc_dpx = np.array( [dp.x for dp in data_interc_dp] )
    data_interc_dpy = np.array( [dp.y for dp in data_interc_dp] )
    knowledge_interc_dpx, knowledge_interc_dpy, knowledge_activ_dpx, knowledge_activ_dpy = get_knowledge_interc_points(S)

    best_stree = None
    best_data_tuning_report = None
    best_knowledge_tuning_report = None
    best_fitness = None

    data_elapsed_time = 0.
    knowledge_elapsed_time = 0.

    n_restarts = 5
    n_actual_restarts = 0

    for _ in range(trials):
        try:
            depth = random.randint(1, max_depth)
            stree = SyntaxTree.create_random(1, n_coeffs, n_coeffs_inner, depth)

            tree_found = False
            if type(stree) is OperatorSyntaxTree and stree.operator_str == '/' and stree.arity == 2 and type(stree.children[0]) is PolySyntaxTree and type(stree.children[1]) is PolySyntaxTree:
                print("TREE FOUND")
                tree_found = True

            for _ in range(n_restarts):
                
                data_tuning_report      = tune_syntax_tree(S, stree, data_interc_dpx,      data_interc_dpy,      verbose=False, maxiter=50)
                knowledge_tuning_report = tune_syntax_tree(S, stree,
                                                           knowledge_interc_dpx, knowledge_interc_dpy, knowledge_activ_dpx, knowledge_activ_dpy,
                                                           verbose=False, maxiter=50)
                fitness = fitness = .2 * data_tuning_report['mse'] + .8 * knowledge_tuning_report['mse']

                if tree_found:
                    pass#print(f"Tree found fitness: {data_tuning_report['mse'] } [data], {knowledge_tuning_report['mse']} [knowledge], {fitness} [fitness]")

                if best_stree is None or fitness < best_fitness:
                    #print(f"Update best fitness from {best_fitness} to {fitness}")
                    best_stree = stree
                    best_data_tuning_report = data_tuning_report
                    best_knowledge_tuning_report = knowledge_tuning_report
                    best_fitness = fitness
                
                data_elapsed_time += data_tuning_report['elapsed_time']
                knowledge_elapsed_time += knowledge_tuning_report['elapsed_time']

                n_actual_restarts += 1
                if fitness <= 0.01: break

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

    return best_stree, best_data_tuning_report, best_knowledge_tuning_report


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
