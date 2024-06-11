import itertools
import random
from scipy.optimize import root_scalar
from scipy import stats
import numpy as np

import dataset
import tree_search


def _find_roots(f, nrestarts:int=10, epsilon=1e-2) -> list:  # roots are returned in ascending order of input x.
    fprime = lambda x: (f(x + epsilon) - f(x)) / epsilon
    roots = []
    search_radius = 0.1

    for _ in range(nrestarts):
        for _ in range(nrestarts):
            x0 = random.uniform(-search_radius, search_radius)
            try: sol = root_scalar(f, fprime=fprime, x0=x0, method='newton')
            except RuntimeWarning: continue
            
            if sol.converged:
                found = False
                for r in roots:
                    if abs(sol.root - r) < 1e-5:
                        found = True
                        break
                if not found:
                    roots.append(sol.root)
        
        search_radius *= 2
    
    return sorted(roots)


def _get_sign(f_x) -> str:
    if f_x == 0: return '0'
    return '+' if f_x > 0 else '-'


class PositivityEvaluator:
    def compute_positivity(self, f, epsilon=1e-10) -> str:
        raise RuntimeError("Method not defined in super class 'PositivityEvaluator'")

class RootPositivityEvaluator(PositivityEvaluator):
    def compute_positivity(self, f, epsilon=1e-10) -> str:
        roots = _find_roots( f )
        if len(roots) == 0: return ''
        positivity = _get_sign( f(roots[0] - epsilon) )
        for root in roots:
            positivity += _get_sign( f(root + epsilon) )
        return positivity

class MeshPositivityEvaluator(PositivityEvaluator):
    def compute_positivity(self, f, epsilon=1e-10) -> str:
        positivity = ''
        curr_sign = None
        X = np.linspace(-10, 10, 500)
        f_X = f(X)
        for i in range(X.size):
            sign = _get_sign(f_X[i])
            if curr_sign is None or sign != curr_sign:
                curr_sign = sign
                positivity += sign
        return positivity


# precondition: dataset centered on origin (0, 0)
class Grounder:
    def ground(self, stree:tree_search.SyntaxTree, epsilon=1e-10):
        raise RuntimeError("Method not defined in super class 'Grounder'")

class BruteForceGrounder(Grounder):
    def ground(self, stree:tree_search.SyntaxTree, epsilon=1e-10):  # returns a list of ground coeffs.
        ncoeffs = stree.get_ncoeffs()
        ground_vals = [0, -1, +1]
        ground_coeffs = []
        positivity_eval = MeshPositivityEvaluator()

        for coeffs in itertools.product(ground_vals, repeat=ncoeffs):
            if coeffs[:3] == (0,0,0) or coeffs[3:] == (0,0,0): continue
            stree.set_coeffs(coeffs)
            
            stree_lamb = tree_search.SyntaxTree.lambdify(stree)
            f = lambda x: stree_lamb(coeffs, x, None, None)
            image_positivity = positivity_eval.compute_positivity(f)
            if image_positivity != '+-': continue

            stree_lambderiv = tree_search.SyntaxTree.lambdify_deriv(stree)
            fprime = lambda x: stree_lambderiv(coeffs, x, None, None)
            deriv_positivity = positivity_eval.compute_positivity(fprime)
            if deriv_positivity != '+-+': continue

            ground_coeffs.append(coeffs)
        
        return ground_coeffs


class DiffGrounder(Grounder):
    def __init__(self, S:dataset.Dataset):
        self.S = S

    def ground(self, stree:tree_search.SyntaxTree, epsilon=1e-10):
        stree.set_coeffs(np.zeros(stree.get_ncoeffs()))
        stree_lamb = tree_search.SyntaxTree.lambdify(stree)
        
        # data points
        interc_dpx = [dp.x for dp in self.S.data]
        interc_dpy = [dp.y for dp in self.S.data]
        image_range = [0, len(interc_dpx)]
        
        # positivity contraints
        image_activ_sign = []
        """if 0 in self.S.knowledge.sign.keys():
            for (l,u,sign,th) in self.S.knowledge.sign[0]:
                sign_val = 1. if sign == '+' else -1.
                sample_size = 500
                interc_dpx += [x for x in np.linspace(l*4, u*4, sample_size)]
                interc_dpy += [th for _ in range(sample_size)]
                image_activ_sign += [sign_val for _ in range(sample_size)]"""
        image_activ_range = [image_range[1], len(interc_dpx)]

        interc_dpx = np.array( interc_dpx )
        interc_dpy = np.array( interc_dpy )

        out1 = np.empty(interc_dpx.size)
        out2 = np.empty(interc_dpx.size)

        # tune syntax tree coeffs
        tuning_report = tree_search.tune_syntax_tree(
            self.S,  # dataset
            stree, stree_lamb, None,  # syntax tree
            out1, out2,  # data out 1 and 2
            interc_dpx, interc_dpy,  # image constraints
            image_activ_sign, None,  # image+deriv activation sign
            image_range, image_activ_range, [interc_dpx.size, interc_dpx.size], [interc_dpx.size, interc_dpx.size],  # constraint ranges
            verbose=False, maxiter=2000)
        
        sol = tuning_report['sol']
        stree.set_coeffs(sol.tolist())
        print(f"Solution: {sol}")
        
        polys = []
        stree.get_polys(polys)
        for poly in polys:
            coeffs = poly.coeffs
            coeffs_abs = np.absolute(coeffs)
            mu  = np.max(coeffs_abs) / 2
            sigma = (np.max(coeffs_abs) - mu) / stats.norm.ppf(0.7)  # (max - mu) / qnorm(98%)

            print("\nPolynomial grounding:")

            ground_coeffs = np.where( coeffs_abs < mu, 0, 1 ) * (coeffs / coeffs_abs)
            print(f"\tGround coeffs: {ground_coeffs}")

            flip_probs = stats.norm.pdf(coeffs_abs, mu, sigma)  #1 - ( np.exp(coeffs) / np.sum(np.exp(coeffs)) )  # apply softmax
            print(f"\tFlip probs: {flip_probs}")
        