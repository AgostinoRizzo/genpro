import numpy as np
from symbols.syntax_tree import SyntaxTree
from gp.stats import GPStats, LayeredGPStats
from gp.evaluation import LayeredEvaluation


class Evaluator:
    def evaluate(self, stree:SyntaxTree): return None
    def create_stats(self): return GPStats()


class R2Evaluator(Evaluator):
    def __init__(self, data):
        self.data = data
    
    def evaluate(self, stree:SyntaxTree):
        ssr = np.sum( (stree(self.data.X) - self.data.y) ** 2 )
        r2  = max( 0., 1 - ((ssr / self.data.sst) if self.data.sst > 0. else 1.) )
        return r2


class KnowledgeEvaluator(Evaluator):
    def __init__(self, know, X_mesh):
        self.know = know
        self.X_mesh = X_mesh
        self.meshspace_map = {}
        self.__init_meshspace_map()
    
    def evaluate(self, stree:SyntaxTree):
        n  = 0
        nv = 0

        # positivity constraints.
        for deriv, constrs in self.know.sign.items():
            derivdeg = len(deriv)
            if derivdeg > 1:
                raise RuntimeError(f"Evaluation of {derivdeg}th derivative not supported.")

            for (l,u,sign,th) in constrs:
                meshspace_idx = \
                    self.meshspace_map[(deriv, l, u, sign, th)] if self.know.nvars == 1 else \
                    self.meshspace_map[(deriv, tuple(l), tuple(u), sign, th)]
                n += meshspace_idx.size

                y = stree[(self.X_mesh, deriv)][meshspace_idx]
                nv += np.sum( (( y < th ) if sign == '+' else ( y > th )) | (~np.isfinite(y)) )
        
        return n, nv
    
    def __init_meshspace_map(self):

        # positivity constraints.
        for deriv, constrs in self.know.sign.items():
            for (l,u,sign,th) in constrs:
                
                meshspace_idx = []
                for i in range(self.X_mesh.shape[0]):
                    pt = self.X_mesh[i]

                    if (pt >= l).all() and (pt <= u).all() and not self.know.is_undef_at(pt):
                        meshspace_idx.append(i)

                if self.know.nvars == 1:
                    self.meshspace_map[(deriv, l, u, sign, th)] = np.array(meshspace_idx)
                else:
                    self.meshspace_map[(deriv, tuple(l), tuple(u), sign, th)] = np.array(meshspace_idx)


class LayeredEvaluator(Evaluator):
    def __init__(self, know_evaluator, r2_evaluator):
        self.know_evaluator = know_evaluator
        self.r2_evaluator = r2_evaluator

    def evaluate(self, stree:SyntaxTree):
        n, nv = self.know_evaluator.evaluate(stree)
        r2 = self.r2_evaluator.evaluate(stree)
        return LayeredEvaluation(n, nv, r2)
    
    def create_stats(self):
        return LayeredGPStats()
