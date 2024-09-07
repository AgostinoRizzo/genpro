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
        self.meshspace = know.spsampler.meshspace(data.xl, data.xu, npoints)
        self.meshspace_map = {}
        self.__init_meshspace_map()
    
    def evaluate(self, stree:backprop.SyntaxTree):
        n   = {0: 0, 1: 0, 2: 0}
        nv  = {0: 0, 1: 0, 2: 0}
        ssr = {0: 0, 1: 0, 2: 0}

        y0 = stree[self.meshspace]
        meshspace_y = {(): y0} #, (0,): (stree(self.meshspace + self.know.numlims.STEPSIZE) - y0) / self.know.numlims.STEPSIZE}

        # positivity constraints.
        for deriv, constrs in self.know.sign.items():
            derivdeg = len(deriv)
            if derivdeg > 0: continue

            for (l,u,sign,th) in constrs:
                meshspace_idx = self.meshspace_map[(deriv, l, u, sign, th)]
                n[derivdeg] += meshspace_idx.shape[0]

                y = meshspace_y[deriv][meshspace_idx]

                #sr = ( np.minimum(0, y - th) if sign == '+' else np.maximum(0, y - th) ) ** 2
                nv [derivdeg] += np.sum( (( y < th ) if sign == '+' else ( y > th )) | np.isnan(y) )
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
                for i in range(self.meshspace.shape[0]):
                    pt = self.meshspace[i]

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