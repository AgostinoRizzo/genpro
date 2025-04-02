import numpy as np
from symbols.syntax_tree import SyntaxTree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.const import ConstantSyntaxTree
from gp.stats import QualitiesGPStats, FeasibilityGPStats
from gp.evaluation import RealEvaluation, LayeredEvaluation, LinearScaling
from backprop.utils import count_symmetric


class LinearScaler:
    def __init__(self, t):
        assert np.isfinite(t).all() and t.size > 0
        self.t = t
        self.t_mean = t.mean()
        self.t_err = t - self.t_mean
        self.t_sst = np.sum(self.t_err ** 2)
        assert self.t_sst >= 0.0
    
    def scale(self, stree, y):
        # return a + by
        a, b = self._compute_coeffs(stree, y)
        return a + b * y, LinearScaling(a, b)
    
    def scale_stree(self, stree, y):
        a, b = self._compute_coeffs(stree, y)
        return LinearScaling(a, b).scale_stree(stree)

    def scale_stree_wrt(self, stree, y, know_evaluator):
        a, b = self._compute_coeffs(stree, y)
        scaling = LinearScaling(a, b)
        if scaling.scaling < 0.0:
            scaling.translation = 0.0
            scaling.scaling = 1.0
        else:
            scaling.translation += know_evaluator.get_image_posit_maxerror(stree, scaling)
        return scaling.scale_stree(stree)
    
    def create_stats(self):
        return QualitiesGPStats(0.0, 1.0, self.name)
    
    def _compute_coeffs(self, stree, y):
        y_mean = y.mean()
        y_err = y - y_mean
        b = np.sum(self.t_err * y_err) / np.sum(y_err ** 2)
        if np.isnan(b): b = 1.0
        a = self.t_mean - b * y_mean
        return a, b

class ConstrainedLinearScaler(LinearScaler):
    def __init__(self, t, know_evaluator):
        super().__init__(t)
        self.know_evaluator = know_evaluator
    
    def _compute_coeffs(self, stree, y):
        y_mean = y.mean()
        y_err = y - y_mean
        
        b = np.sum(self.t_err * y_err) / np.sum(y_err ** 2)
        if np.isnan(b) or b < 0.0: b = 1.0
        
        a = self.t_mean - b * y_mean
        a += self.know_evaluator.get_image_posit_maxerror(stree, LinearScaling(a, b))

        return a, b


class ConstraintsPassLinearScaler(LinearScaler):
    def __init__(self, t):
        super().__init__(t)

    def _compute_coeffs(self, stree, y):
        a = 0.0
        b = np.sum(self.t * y) / np.sum(y**2)
        if b <= 0.0: b = 1.0
        return a, b


class Evaluator:
    def evaluate(self, stree:SyntaxTree): return None
    def create_stats(self): return None


class DataEvaluator(Evaluator):
    def __init__(self, data, linscaler:LinearScaler=None):
        assert np.isfinite(data.y).all()
        self.data = data
        self.linscaler = linscaler
    
    def evaluate(self, stree:SyntaxTree):
        if self.data.sst <= 0: return None
        y = stree(self.data.X)
        if np.isfinite(y).all():
            if self.linscaler is None: return y, LinearScaling()
            else: return self.linscaler.scale(stree, y)
        return None, LinearScaling()
    
class R2Evaluator(DataEvaluator):
    def __init__(self, data, linscaler:LinearScaler=None):
        super().__init__(data, linscaler)
        self.name = 'R²'
        
    def evaluate(self, stree:SyntaxTree):
        r2 = -np.inf
        y, scaling = super().evaluate(stree)
        if y is not None:
            ssr = np.sum( (y - self.data.y) ** 2 )
            r2  = 1 - (ssr / self.data.sst)
            if not np.isfinite(r2):
                r2 = -np.inf
        r2 = max(r2,0.0)
        return RealEvaluation(r2, minimize=False, name=self.name, isfeasible=(r2>0.0), scaling=scaling)
    
    def create_stats(self):
        return QualitiesGPStats(1.0, 0.0, self.name)


class PearsonR2Evaluator(DataEvaluator):
    def __init__(self, data, linscaler:LinearScaler=None):
        super().__init__(data, linscaler)
        data_y_mean = data.y.mean()
        self.data_y_err = data.y - data_y_mean
        self.data_y_sst = np.sum(self.data_y_err ** 2)
        self.name = 'Pearson R²'
    
    def evaluate(self, stree:SyntaxTree):
        r2 = 0.0
        y, scaling = super().evaluate(stree)

        if self.data_y_sst > 0.0 and y is not None:
            y_mean = y.mean()
            y_err = y - y_mean
            y_sst = np.sum(y_err ** 2)
            if y_sst > 0.0:
                r2 = np.sum(y_err * self.data_y_err) / np.sqrt(y_sst * self.data_y_sst)
                r2 = r2 ** 2
        
        return RealEvaluation(r2, minimize=False, name=self.name, scaling=scaling)
    
    def create_stats(self):
        return QualitiesGPStats(0.0, 1.0, self.name)


class MSEEvaluator(DataEvaluator):
    def __init__(self, data, linscaler:LinearScaler=None):
        super().__init__(data, linscaler)
        self.n = data.y.size
        self.name = 'MSE'
        self.const_mse = np.sum((self.data.y.mean() - self.data.y) ** 2) / self.n
    
    def evaluate(self, stree:SyntaxTree):
        mse = np.inf
        y, scaling = super().evaluate(stree)

        if y is not None:
            mse = np.sum((y - self.data.y) ** 2) / self.n
        
        return RealEvaluation(mse, minimize=True, name=self.name, isfeasible=(mse<self.const_mse), scaling=scaling)
    
    def create_stats(self):
        return QualitiesGPStats(0.0, np.inf, self.name)


class NMSEEvaluator(DataEvaluator):
    def __init__(self, data, linscaler:LinearScaler=None):
        super().__init__(data, linscaler)
        self.n = data.y.size
        self.data_y_var = data.y.var()
        self.name = 'NMSE'
    
    def evaluate(self, stree:SyntaxTree):
        nmse = np.inf
        y, scaling = super().evaluate(stree)

        if y is not None and self.n > 0 and self.data_y_var > 0.0:
            nmse = np.sum((y - self.data.y) ** 2) / (self.n * self.data_y_var)
            if not np.isfinite(nmse):
                nmse = np.inf
        nmse = min(nmse, 1.0)
        return RealEvaluation(nmse, minimize=True, name=self.name, isfeasible=(nmse<1.0), scaling=scaling)
    
    def create_stats(self):
        return QualitiesGPStats(0.0, 1.0, self.name)


class KnowledgeEvaluator(Evaluator):
    def __init__(self, know, mesh):
        self.know = know
        self.mesh = mesh
        self.meshspace_map = {}
        self.__init_meshspace_map()
    
    def evaluate(self, stree:SyntaxTree, scaling):
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
                if derivdeg == 0:
                    y = scaling.translation + scaling.scaling * y
                elif derivdeg == 1:
                    y = scaling.scaling * y

                nv += np.sum( (( y < th ) if sign == '+' else ( y > th )) | (~np.isfinite(y)) )
        
        # symmetry constraints.
        if self.know.has_symmvars():
            __n = len(self.know.symmvars) - 1
            n += __n
            nv += __n - count_symmetric(stree[(self.mesh.X, ())], self.mesh.symm_Y_Ids)

        return n, nv
    
    def get_image_posit_maxerror(self, stree, scaling):
        max_errors = []
        
        # positivity constraints.
        for deriv, constrs in self.know.sign.items():
            derivdeg = len(deriv)
            if derivdeg != 0: continue

            for (l,u,sign,th) in constrs:
                meshspace_idx = \
                    self.meshspace_map[(deriv, l, u, sign, th)] if self.know.nvars == 1 else \
                    self.meshspace_map[(deriv, tuple(l), tuple(u), sign, th)]

                y = stree[(self.mesh.X, deriv)][meshspace_idx]
                y_scaled = scaling.translation + scaling.scaling * y

                th_err = np.maximum(0, th - y_scaled) if sign == '+' else np.maximum(0, y_scaled - th)
                max_th_err = th_err.max()
                if max_th_err > 0.0:
                    max_errors.append(max_th_err if sign == '+' else (-max_th_err))
        
        return np.mean(max_errors)
                
    
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
    def __init__(self, know_evaluator, data_evaluator, know_pressure:float=1.0):
        self.know_evaluator = know_evaluator
        self.data_evaluator = data_evaluator
        self.know_pressure  = know_pressure

    def evaluate(self, stree:SyntaxTree):
        data_eval = self.data_evaluator.evaluate(stree)
        n, nv = self.know_evaluator.evaluate(stree, data_eval.scaling)
        return LayeredEvaluation(n, nv, data_eval, stree, self.know_pressure)
    
    def create_stats(self):
        return FeasibilityGPStats(self.data_evaluator.create_stats())


class UnconstrainedLayeredEvaluator(LayeredEvaluator):
    def __init__(self, know_evaluator, data_evaluator):
        super().__init__(know_evaluator, data_evaluator, 0.0)
