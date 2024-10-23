import numpy as np
from symbols.syntax_tree import SyntaxTree
from gp.stats import QualitiesGPStats, FeasibilityGPStats
from gp.evaluation import RealEvaluation, LayeredEvaluation, UnconstrainedLayeredEvaluation
from backprop.utils import count_symmetric


class Evaluator:
    def evaluate(self, stree:SyntaxTree): return None
    def create_stats(self): return None


class R2Evaluator(Evaluator):
    def __init__(self, data):
        self.data = data
    
    def evaluate(self, stree:SyntaxTree):
        ssr = np.sum( (stree(self.data.X) - self.data.y) ** 2 )
        r2  = max( 0., 1 - ((ssr / self.data.sst) if self.data.sst > 0. else 1.) )
        return RealEvaluation(r2, minimize=False)
    
    def create_stats(self):
        return QualitiesGPStats(0.0, 1.0, 'R2')


class KnowledgeEvaluator(Evaluator):
    def __init__(self, know, mesh):
        self.know = know
        self.mesh = mesh
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

                y = stree[(self.mesh.X, deriv)][meshspace_idx]
                nv += np.sum( (( y < th ) if sign == '+' else ( y > th )) | (~np.isfinite(y)) )
        
        # symmetry constraints.
        if self.know.has_symmvars():
            __n = len(self.know.symmvars) - 1
            n += __n
            nv += __n - count_symmetric(stree[(self.mesh.X, ())], self.mesh.symm_Y_Ids)

        return n, nv
    
    def __init_meshspace_map(self):

        # positivity constraints.
        for deriv, constrs in self.know.sign.items():
            for (l,u,sign,th) in constrs:
                
                meshspace_idx = []
                for i in range(self.mesh.X.shape[0]):
                    pt = self.mesh.X[i]

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
        r2 = self.r2_evaluator.evaluate(stree).value
        return LayeredEvaluation(n, nv, r2, stree)
    
    def create_stats(self):
        return FeasibilityGPStats(QualitiesGPStats(0.0, 1.0, 'R2'))


class UnconstrainedLayeredEvaluator(LayeredEvaluator):
    def __init__(self, know_evaluator, r2_evaluator):
        super().__init__(know_evaluator, r2_evaluator)

    def evaluate(self, stree:SyntaxTree):
        n, nv = self.know_evaluator.evaluate(stree)
        r2 = self.r2_evaluator.evaluate(stree).value
        return UnconstrainedLayeredEvaluation(n, nv, r2)
