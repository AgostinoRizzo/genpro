import numpy as np
from backprop import gp, backprop


class FastR2Evaluator(gp.Evaluator):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def evaluate(self, stree:backprop.SyntaxTree):
        ssr   = np.sum( (stree(self.dataset.X) - self.dataset.y) ** 2 )
        r2    = max( 0., 1 - ((ssr / self.dataset.sst) if self.dataset.sst > 0. else 1.) )
        return r2
        #return RealEvaluation(r2, minimize=False)


class FastKnowledgeEvaluator(gp.Evaluator):
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
    
    def evaluate(self, stree:backprop.SyntaxTree):

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


class FastFUEvaluator(gp.Evaluator):
    def __init__(self, dataset, knowledge):
        self.data_evaluator = FastR2Evaluator(dataset)
        self.know_evaluator = FastKnowledgeEvaluator(knowledge)

    def evaluate(self, stree:backprop.SyntaxTree, eval_deriv=False):
        
        know_eval = self.know_evaluator.evaluate(stree)
        know_mse  = (know_eval['mse0'] + know_eval['mse1'] + know_eval['mse2']) / 3  # TODO: separate?! or a weighted mean?
        know_nv   =  know_eval['nv0' ] + know_eval['nv1' ] + know_eval['nv2' ]
        know_n    =  know_eval['n0'  ] + know_eval['n1'  ] + know_eval['n2'  ]
        know_ls   =  0  # TODO: remove it!
        know_sat  = stree.sat
        if np.isnan(know_mse): know_mse = 1e12

        data_r2 = self.data_evaluator.evaluate(stree)

        return gp.FUEvaluation(know_mse, know_nv, know_n, know_ls, know_sat, data_r2, stree.cache.nnodes)
    
    def create_stats(self, nbests:int=1):
        return gp.FUGPStats(nbests)