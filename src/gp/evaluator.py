import numpy as np
from symbols.syntax_tree import SyntaxTree
from symbols.deriv import Derivative
from gp import gp, evaluation


class Evaluator:
    def evaluate(self, stree:SyntaxTree): return None
    def create_stats(self): return GPStats()


class R2Evaluator(Evaluator):
    def __init__(self, dataset, minimize:bool=True):
        self.dataset = dataset
        self.minimize = False
    
    def evaluate(self, stree:SyntaxTree):
        return RealEvaluation(max(0., self.dataset.evaluate(stree)['r2']), self.minimize)


# different from srgp.KnowledgeEvaluator
class KnowledgeEvaluator(Evaluator):
    def __init__(self, knowledge):
        self.K = knowledge
    
    def _compute_stree_derivs(self, stree, derivs):
        return SyntaxTree.diff_all(stree, derivs, include_zeroth=True)
    
    def evaluate(self, stree:SyntaxTree):
        K_derivs = self.K.get_derivs()
        stree_derivs = self._compute_stree_derivs(stree, K_derivs)
        K_eval = self.K.evaluate(stree_derivs)
        K_eval = (K_eval['mse0'] + K_eval['mse1'] + K_eval['mse2']) / 3  # TODO: separate?! or a weighted mean?
        if np.isnan(K_eval): K_eval = 1e12
        return RealEvaluation(K_eval, minimize=True)


class NumericalKnowledgeEvaluator(KnowledgeEvaluator):
    def __init__(self, knowledge):
        super().__init__(knowledge)
    
    def _compute_stree_derivs(self, stree, derivs):
        return Derivative.create_all(stree, derivs, self.K.nvars, self.K.numlims)
    
    def evaluate(self, stree:SyntaxTree):
        K_derivs = self.K.get_derivs()
        stree_derivs = self._compute_stree_derivs(stree, K_derivs)
        K_eval = self.K.evaluate(stree_derivs, eval_deriv=True)
        K_eval = (K_eval['mse0'] + K_eval['mse1'] + K_eval['mse2']) / 3  # TODO: separate?! or a weighted mean?
        if np.isnan(K_eval): K_eval = 1e12
        return RealEvaluation(K_eval, minimize=True)


class FUEvaluator(Evaluator):
    def __init__(self, dataset, knowledge):
        self.data = dataset
        self.know = knowledge
    
    def _compute_stree_derivs(self, stree, derivs):
        return SyntaxTree.diff_all(stree, derivs, include_zeroth=True)

    def evaluate(self, stree:SyntaxTree, eval_deriv=False):
        know_derivs = self.know.get_derivs()
        stree_derivs = self._compute_stree_derivs(stree, know_derivs)
        
        know_eval = self.know.evaluate(stree_derivs, eval_deriv)
        know_mse  = (know_eval['mse0'] + know_eval['mse1'] + know_eval['mse2']) / 3  # TODO: separate?! or a weighted mean?
        know_nv   =  know_eval['nv0' ] + know_eval['nv1' ] + know_eval['nv2' ]
        know_n    =  know_eval['n0'  ] + know_eval['n1'  ] + know_eval['n2'  ]
        know_ls   =  know_eval['ls0' ] and know_eval['ls1' ] and know_eval['ls2' ]
        know_sat  = stree.sat
        if np.isnan(know_mse): know_mse = 1e12

        data_r2 = max(0., self.data.evaluate(stree)['r2']) # TODO: put this into data.evaluate(...).

        """optCollector = backprop.SyntaxTreeOperatorCollector()
        stree.accept(optCollector)
        ispoly = True
        for o in ['/', 'log', 'exp', 'sqrt']:
            if o in optCollector.opts:
                ispoly = False
                break
        if ispoly:
            know_nv = 1e10
            know_mse = 1e10"""

        return FUEvaluation(know_mse, know_nv, know_n, know_ls, know_sat, data_r2, stree.get_nnodes())
    
    def create_stats(self): return FUGPStats()


class NumericalFUEvaluator(FUEvaluator):
    def __init__(self, dataset, knowledge):
        super().__init__(dataset, knowledge)
    
    def _compute_stree_derivs(self, stree, derivs):
        return Derivative.create_all(stree, derivs, self.know.nvars, self.know.numlims)


"""
Fast Evaluator.
"""

class FastR2Evaluator(Evaluator):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def evaluate(self, stree:SyntaxTree):
        ssr   = np.sum( (stree(self.dataset.X) - self.dataset.y) ** 2 )
        r2    = max( 0., 1 - ((ssr / self.dataset.sst) if self.dataset.sst > 0. else 1.) )
        return r2
        #return RealEvaluation(r2, minimize=False)


class FastKnowledgeEvaluator(Evaluator):
    def __init__(self, know, npoints:int=100):
        self.know = know
        data = know.dataset
        
        self.meshspace = {}
        self.derivs = know.get_derivs()
        meshspace_0 = know.spsampler.meshspace(data.xl, data.xu, npoints)
        nvars = data.nvars
        for d in self.derivs:
            derivdeg = len(d)
            if derivdeg == 0: self.meshspace[d] = meshspace_0
            if derivdeg != 1: continue  # TODO: only up to first derivative (*).
            h = np.zeros(data.nvars)
            h[d[0]] = self.know.numlims.STEPSIZE
            self.meshspace[d] = meshspace_0 + h
        
        self.meshspace_map = {}
        self.__init_meshspace_map()
    
    def evaluate(self, stree:SyntaxTree):

        #from symbols.parsing import parse_syntax_tree
        #stree = parse_syntax_tree('((square(x0) * (sqrt(0.64) / x0)) / square((square(x0) - (-0.18 / 1.94))))')

        n   = {0: 0, 1: 0, 2: 0}
        nv  = {0: 0, 1: 0, 2: 0}
        ssr = {0: 0, 1: 0, 2: 0}
        
        meshspace_y = {}
        y0 = stree[(self.meshspace[()], ())]
        for d in self.derivs:
            derivdeg = len(d)
            if derivdeg == 0: meshspace_y[()] = y0
            if derivdeg != 1: continue  # TODO: only up to first derivative (*).
            meshspace_y[d] = (stree[(self.meshspace[d], d)] - y0) / self.know.numlims.STEPSIZE

        # positivity constraints.
        for deriv, constrs in self.know.sign.items():
            derivdeg = len(deriv)
            if derivdeg > 0: continue

            for (l,u,sign,th) in constrs:
                meshspace_idx = self.meshspace_map[(deriv, l, u, sign, th)]
                n[derivdeg] += meshspace_idx.size

                y = meshspace_y[deriv][meshspace_idx]

                #sr = ( np.minimum(0, y - th) if sign == '+' else np.maximum(0, y - th) ) ** 2

                #if derivdeg == 0:
                nv [derivdeg] += np.sum( (( y < th ) if sign == '+' else ( y > th )) | np.isnan(y) )
                #else:
                #    nv [derivdeg] += np.sum( (( y < th - 1e-2 ) if sign == '+' else ( y > th + 1e-2 )) | np.isnan(y) )
                
                #nv [derivdeg] += np.sum(np.isnan(sr))
                #ssr[derivdeg] += np.sum(sr)
        
        return {'mse0': (ssr[0]/n[0]) if n[0] > 0. else 0.,
                'mse1': (ssr[1]/n[1]) if n[1] > 0. else 0.,
                'mse2': (ssr[2]/n[2]) if n[2] > 0. else 0.,
                'nv0' : nv[0], 'nv1' : nv[1], 'nv2' : nv[2],
                'n0'  : n [0], 'n1'  : n [1], 'n2'  : n [2]}

    
    def __init_meshspace_map(self):

        # positivity constraints.
        for deriv, constrs in self.know.sign.items():
            for (l,u,sign,th) in constrs:
                
                meshspace_idx = []
                for i in range(self.meshspace[()].shape[0]):
                    pt = self.meshspace[()][i]

                    if (pt >= l).all() and (pt <= u).all():
                        meshspace_idx.append(i)

                self.meshspace_map[(deriv, l, u, sign, th)] = np.array(meshspace_idx)


class FastFUEvaluator(Evaluator):
    def __init__(self, dataset, knowledge):
        self.data_evaluator = FastR2Evaluator(dataset)
        self.know_evaluator = FastKnowledgeEvaluator(knowledge)

    def evaluate(self, stree:SyntaxTree, eval_deriv=False):
        
        know_eval = self.know_evaluator.evaluate(stree)
        know_mse  = (know_eval['mse0'] + know_eval['mse1'] + know_eval['mse2']) / 3  # TODO: separate?! or a weighted mean?
        know_nv   =  know_eval['nv0' ] + know_eval['nv1' ] + know_eval['nv2' ]
        know_n    =  know_eval['n0'  ] + know_eval['n1'  ] + know_eval['n2'  ]
        know_ls   =  0  # TODO: remove it!
        know_sat  = stree.sat
        if np.isnan(know_mse): know_mse = 1e12

        data_r2 = self.data_evaluator.evaluate(stree)

        return evaluation.FUEvaluation(know_mse, know_nv, know_n, know_ls, know_sat, data_r2, stree.cache.nnodes)
    
    def create_stats(self):
        return gp.FUGPStats()