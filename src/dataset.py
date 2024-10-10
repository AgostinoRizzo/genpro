import random
import math
import numpy as np
import sympy
import csv
from sklearn.model_selection import train_test_split
import numlims
import space
import plotting


class DataPoint:
    def __init__(self, x, y:float) -> None:
        self.x = np.array(x, dtype=float) if type(x) is list else np.float64(x)
        self.y = np.float64(y)
    
    def distance(self, other) -> float:
        return np.sqrt((other.x-self.x)**2 + (other.y-self.y)**2)


class Evaluation:
    def __init__(self):
        self.training  = {}
        self.testing   = {}
        self.knowledge = {}
    
    def __init__(self, training:dict, testing:dict, knowledge:dict=None):
        self.training  = training
        self.testing   = testing
        self.knowledge = knowledge
    
    def better_than(self, other) -> bool:
        for derivdeg in range(3):
            mse = 'mse' + str(derivdeg)
            if self .knowledge[mse] < other.knowledge[mse] and self .testing['r2'] >= 0.1: return True
            if other.knowledge[mse] < self .knowledge[mse] and other.testing['r2'] >= 0.1: return False
        if self .testing['r2'] > other.testing['r2']: return True
        if other.testing['r2'] > self .testing['r2']: return False
        return self.training['r2'] > other.training['r2']
    
    def __str__(self) -> str:
        eval_str = ''
        for eval_name, measure_map in [('Training', self.training), ('Testing', self.testing), ('Knowledge', self.knowledge)]:
            eval_str += f"{eval_name}\n"
            for measure, value in measure_map.items(): eval_str += f"\t{measure}: {value}\n"
        return eval_str


class DataKnowledge:
    def __init__(self,
                 dataset=None,
                 limits:numlims.NumericLimits=None,
                 spsampler:space.SpaceSampler=None) -> None:
        assert dataset is None or (limits is None and spsampler is None)
        self.dataset = dataset
        self.nvars = None if dataset is None else dataset.nvars
        self.derivs = {}
        self.sign = {}
        self.symm = {}
        self.noroot = set()
        self.zero = {}
        self.undef = []
        self.numlims = limits if dataset is None else dataset.numlims
        self.spsampler = spsampler if dataset is None else dataset.spsampler
    
    def add_deriv(self, d, xy:DataPoint):
        if type(d) is int: d = (0,)*d
        assert type(d) is tuple
        if d not in self.derivs:
            self.derivs[d] = []
        self.derivs[d].append(xy)
    
    def add_sign(self, d, l, u, sign:str='+', th:float=0):
        if type(d) is int: d = (0,)*d
        if type(l) is list: l = np.array(l, dtype=float)
        if type(u) is list: u = np.array(u, dtype=float)
        assert type(d) is tuple
        if d not in self.sign:
            self.sign[d] = []
        self.sign[d].append((l,u,sign,th))
    
    def add_symm(self, d, x, iseven:bool=True):
        if type(d) is int: d = (0,)*d
        assert type(d) is tuple
        assert type(x) is float or type(x) is int  # TODO: manage symmetry constraints in multivar (*).
        self.symm[d] = (x, iseven)
    
    def add_noroot(self, d):
        if type(d) is int: d = (0,)*d
        assert type(d) is tuple
        self.noroot.add(d)
    
    def add_zero(self, d, l, u):
        if type(d) is int: d = (0,)*d
        assert type(d) is tuple
        self.zero[d] = (l,u)
    
    def add_undef(self, x):
        self.undef.append(x)
    
    def is_undef_at(self, x):
        for undef_x in self.undef:
            if np.array_equal(x, undef_x):
                return True
        return False
    
    """
    def get_mesh(self, sample_size:int=20) -> np.array:
        deriv_points = []
        for dp in self.derivs.values():
            deriv_points.append(dp.x)
        return np.concatenate((
            np.linspace(self.dataset.xl, self.dataset.xu, max(0, sample_size - len(deriv_points))),
            np.array(deriv_points))).sort()
    """
    
    def evaluate(self, model_map:dict, eval_deriv=False) -> dict:
        n   = {0: 0, 1: 0, 2: 0}
        nv  = {0: 0, 1: 0, 2: 0}
        ssr = {0: 0, 1: 0, 2: 0}
        ls  = {0: True, 1: True, 2: True}

        # intersection points.
        for deriv, dps in self.derivs.items():
            derivdeg = len(deriv)
            #if derivdeg != 0 and not eval_deriv: continue
            n[derivdeg] += len(dps)
            X = np.array( [dp.x for dp in dps] )
            y = np.array( [dp.y for dp in dps] )

            sr = (model_map[deriv](X) - y) ** 2
            nv [derivdeg] += np.count_nonzero(sr)
            nv [derivdeg] += np.sum(np.isnan(sr))
            ssr[derivdeg] += np.sum(sr)
        
        # positivity constraints.
        for deriv, constrs in self.sign.items():
            derivdeg = len(deriv)
            #if derivdeg != 0 and not eval_deriv: continue
            for (_l,_u,sign,th) in constrs:
                l = _l + self.numlims.EPSILON
                u = _u - self.numlims.EPSILON
                if np.any(l > u): continue
                
                X = self.spsampler.meshspace(l, u, 20)  # TODO: factorize sample size.
                n[derivdeg] += X.shape[0]
                model_y = model_map[deriv](X)

                sr = ( np.minimum(0, model_y - th) if sign == '+' else np.maximum(0, model_y - th) ) ** 2
                nv [derivdeg] += np.sum(( model_y < th ) if sign == '+' else ( model_y > th ))  #np.count_nonzero(sr)
                nv [derivdeg] += np.sum(np.isnan(sr))
                ssr[derivdeg] += np.sum(sr)
        
        # positivity landscape.
        """X = self.spsampler.meshspace(self.dataset.xl, self.dataset.xu, 20)
        model_y0 = np.sign( model_map[()](X) )
        model_y1 = np.sign( model_map[(0,)](X) )
        
        from itertools import groupby
        model_y0 = [k for k,g in groupby(model_y0)]
        model_y1 = [k for k,g in groupby(model_y1)]
        ls[0] = model_y0 == [1, -1]
        ls[1] = model_y1 == [1, -1, 1]"""

        
        # symmetry constraints.
        """for deriv, (x0, iseven) in self.symm.items():
            assert type(x0) is float or type(x0) is int  # TODO: manage symmetry constraints in multivar (*).
            derivdeg = len(deriv)
            X = self.spsampler.meshspace(x0 + self.numlims.EPSILON, self.numlims.INFTY, 20)  # TODO: factorize sample size.
            n[derivdeg] += X.shape[0]
            model_y1 = model_map[deriv](X)
            model_y2 = model_map[deriv](x0-(X-x0))
            
            sr = ( (model_y1 - model_y2) if iseven else (model_y1 + model_y2) ) ** 2
            nv [derivdeg] += np.count_nonzero(sr)
            nv [derivdeg] += np.sum(np.isnan(sr))
            ssr[derivdeg] += np.sum(sr)"""
        
        return {'mse0': (ssr[0]/n[0]) if n[0] > 0. else 0.,
                'mse1': (ssr[1]/n[1]) if n[1] > 0. else 0.,
                'mse2': (ssr[2]/n[2]) if n[2] > 0. else 0.,
                'nv0' : nv[0], 'nv1' : nv[1], 'nv2' : nv[2],
                'n0'  : n [0], 'n1'  : n [1], 'n2'  : n [2],
                'ls0' : ls[0], 'ls1' : ls[1], 'ls2' : ls[2]}
    
    def synthesize(self, refmod, X) -> dict:
        #X = np.append(X, self.spsampler.meshspace(self.dataset.xl, self.dataset.xu, 20))
        K_derivs = self.get_derivs()
        from backprop import backprop
        model_derivs = backprop.SyntaxTree.diff_all(refmod, K_derivs, include_zeroth=True)
        valid = np.full(X.shape[0], True)

        # TODO: intersection points.
        
        # positivity constraints.
        for deriv, constrs in self.sign.items():
            derivdeg = len(deriv)
            for (_l,_u,sign,th) in constrs:
                l = _l + self.numlims.EPSILON
                u = _u - self.numlims.EPSILON
                if np.any(l > u): continue
                
                y_valid = model_derivs[deriv](X) > th if sign == '+' else model_derivs[deriv](X) < th
                valid &= (X < l).ravel() | (X > u).ravel() | y_valid
        
        # TODO: symmetry constraints.
        
        X_valid = X[valid]
        S = NumpyDataset(nvars=self.nvars)
        S.xl = self.dataset.xl
        S.xu = self.dataset.xu
        S.yl = self.dataset.yl
        S.yu = self.dataset.yu
        S.X = X_valid
        S.y = model_derivs[()](X_valid)
        return S
    
    def get_derivs(self) -> set[tuple[int]]:
        derivs = set()
        derivs.update(self.derivs.keys())
        derivs.update(self.sign.keys())
        derivs.update(self.symm.keys())
        derivs.update(self.noroot)
        return derivs
    
    def plot(self):
        if self.nvars != 1:
            raise RuntimeError(f"Plotting not supported for {self.nvars} dimensions.")
        
        if () in self.derivs.keys():
            for xy in self.derivs[()]:
                plt.plot(xy.x, xy.y, 'rx', markersize=10)
        
        if () in self.sign.keys():
            for (l,u,s,_) in self.sign[()]:
                plt.axvspan(l, u, alpha=0.05, color='g' if s == '+' else 'r')
    
    def __str__(self) -> str:
        out_str = ''

        out_str += "===== Intersection Points =====\n"
        for deriv, dps in self.derivs.items():
            out_str += f"Deriv: {deriv}\n"
            for dp in dps:
                out_str += f"\tx={dp.x}, y={dp.y}\n"
        
        out_str += "\n===== Positivity Constraints =====\n"
        for deriv, constrs in self.sign.items():
            out_str += f"Deriv: {deriv}\n"
            for (_l,_u,sign,th) in constrs:
                sign_str = '>' if sign == '+' else '<'
                out_str += f"{sign_str}{th} [{_l}, {_u}]\n"
        
        out_str += "\n===== Symmetry Constraints =====\n"
        for deriv, (x0, iseven) in self.symm.items():
            iseven_str = 'even' if iseven else 'odd'
            out_str += f"Deriv: {deriv}, x0={x0}, {iseven_str}\n"
        
        return out_str
    
    def synth_dataset(self, X, deriv:tuple[int]=()):
        S = Dataset(self.dataset.nvars, self.dataset.xl, self.dataset.xu, self.spsampler)
        
        y = np.full(X.shape[0], np.nan)
        
        # positivity constraints.
        for (l, u,sign,th) in self.sign[deriv]:
            assert th == 0.0

            if X.ndim == 1:
                y[(X >= l) & (X <= u)] = 1.0 if sign == '+' else -1.0
            else:
                y[(X >= l).all(axis=1) & (X <= u).all(axis=1)] = 1.0 if sign == '+' else -1.0
            
            for i, y_i in enumerate(y):
                S.data.append( DataPoint(X[i], y_i) )
        
        S = NumpyDataset(S)
        S.X = X
        S.y = y
        return S
            

def compute_sst(dpoints:list) -> float:
    if len(dpoints) == 0: return 0.
    y_mean = 0.
    for dp in dpoints: y_mean += dp.y
    y_mean /= len(dpoints)

    sst = 0.
    for dp in dpoints: sst += (dp.y - y_mean) ** 2
    return sst


class Dataset:
    def __init__(self,
                 nvars:int,
                 xl, xu,
                 spsampler:space.SpaceSampler,
                 plotter:plotting.DatasetPlotter=None) -> None:
        self.data = []
        self.test = []
        self.nvars = nvars
        self.xl = xl
        self.xu = xu
        self.yl = -1.
        self.yu =  1.
        self._xl = self.xl  # used for scaling.
        self._xu = self.xu
        self._yl = self.yl
        self._yu = self.yu
        self.spsampler = spsampler
        self.plotter = plotter
        self.numlims = numlims.NumericLimits()
        self.numlims.set_bounds(self.xl, self.xu)
        self.knowledge = DataKnowledge(self)
        self.data_sst = 0.
        self.test_sst = 0.
        """self.__sorted_data__ = None  # TODO: remove index?!
        self.__x_map__ = None"""
    
    def sample(self, size:int=100, noise:float=0., mesh:bool=False):
        X = \
            self.spsampler.meshspace(self.xl, self.xu, self.spsampler.get_meshsize(size)) if mesh else \
            self.spsampler.randspace(self.xl, self.xu, size)
        
        y = self.func(X)
        if noise > 0.:
            y_std = y.std()
            y += np.random.normal(scale=math.sqrt(noise*y_std), size=X.shape[0])

        for i in range(y.size):
            self.data.append(DataPoint(X[i], y[i]))
        
        self._on_data_changed()
    
    def load(self, filename:str):
        csvfile = open(filename)
        csvreader = csv.reader(csvfile)
        for entry in csvreader:
            x = np.array( [float(entry[i]) for i in range(self.nvars)] )
            y = float(entry[self.nvars])
            if self.is_xscaled(): x = self._xmap(x)
            if self.is_yscaled(): y = self._ymap(y)
            self.data.append(DataPoint(x, y))
        csvfile.close()
        self._on_data_changed()
    
    def erase(self, x_from, x_to):
        self.test = []
        new_data = []
        for dp in self.data:
            if (dp.x < x_from).all() or (dp.x > x_to).all(): new_data.append(dp)
        self.data = new_data
        self._on_data_changed()

    def split(self, train_size:float=0.7, randstate:int=0):  # TODO: remove init val of randstate
        if len(self.data) > 1:
            train, test = train_test_split(self.data, train_size=train_size, random_state=randstate)
            self.data = list(train)
            self.test = list(test)
            self._on_data_changed()
    
    def clear(self):
        self.data = []
        self.test = []
        self._on_data_changed()
    
    def minmax_scale_y(self):
        Dataset.__minmax_scale_y(self.data, self.yl, self.yu)
    
    def remove_outliers(self):
        self.data = Dataset.__remove_outliers(self.data)
    
    @staticmethod
    def __minmax_scale_y(data:list, yl:float, yu:float):
        if len(data) == 0: return
        y_min = data[0].y
        y_max = data[0].y
        for dp in data:
            if dp.y < y_min: y_min = dp.y
            if dp.y > y_max: y_max = dp.y
        y_l = y_max - y_min
        for dp in data:
            y_std = (dp.y - y_min) / y_l
            dp.y = (y_std * (yu - yl)) + yl
    
    @staticmethod
    def __remove_outliers(data:list) -> list:
        if len(data) == 0: return list()
        
        Y = [dp.y for dp in data]
        Q1 = np.percentile(Y, 25, method='midpoint')
        Q3 = np.percentile(Y, 75, method='midpoint')
        IQR = Q3 - Q1
        upper = Q3+70.5*IQR
        lower = Q1-70.5*IQR
        
        new_data = []
        for dp in data:
            if dp.y <= upper and dp.y >= lower: new_data.append(dp)
        return new_data
    
    """
    def index(self):
        self.__sorted_data__ = sorted(self.data, key=lambda dp: dp.x)
        self.__x_map__ = {}
        for idx, dp in enumerate(self.__sorted_data__):
            self.__x_map__[dp.x] = idx
    
    def share_index(self, other):
        self.__sorted_data__ = other.__sorted_data__
        self.__x_map__ = other.__x_map__
    
    def compute_point_density(self, dp:DataPoint) -> float:  # TODO: can be optimized using numpy.
        d = 0.
        n = len(self.data)
        for other_dp in self.data:
            if id(other_dp) == id(dp):
                n -= 1
                continue
            d += np.sqrt( ((other_dp.x-dp.x) ** 2) + ((other_dp.y-dp.y) ** 2) )
        max_d = np.sqrt( ((self.xu-self.xl) ** 2) + ((self.yu-self.yl) ** 2) )
        return max_d - (d / n)
    """
    
    def compute_yvar(self, x0, dx_ratio:float=0.05) -> float:
        """assert self.__sorted_data__ is not None and self.__x_map__ is not None

        k = max( int(len(self.data) * k_ratio), 1 )
        if k == 1: return 0.
        k_left = (k-1) // 2
        k_right = k - 1 - k_left

        i0 = self.__x_map__[dp0.x]
        offset = i0 - k_left
        Y = np.empty(k)
        Y[i0-offset] = self.__sorted_data__[i0].y
        for i in range(i0-1, max(i0-k_left-1, -1),     -1): Y[i-offset] = self.__sorted_data__[i].y
        for i in range(i0+1, min(i0+k_left+1, len(Y)), +1): Y[i-offset] = self.__sorted_data__[i].y

        if dp0.x < -1.5 or (dp0.x > -0.5 and dp0.x < 0.5): print(f"-----> VAR({dp0.x}) = {np.var(Y)}")
        return np.var(Y)"""

        h = (self.xu - self.xl) * 0.05 * 0.5
        l = x0 - h
        u = x0 + h
        Y = []
        for dp in self.data:
            if (dp.x > l).all() and (dp.x < u).all(): Y.append(dp.y)
        if len(Y) <= 1: return self.numlims.EPSILON  # TODO: manage this situation (otherwise in qp.qp_solve we have a division by zero).
        return np.var( np.array(Y) )
    
    def _on_data_changed(self):
        self.data_sst = compute_sst(self.data)
        self.test_sst = compute_sst(self.test)
    
    def func(self, x:float) -> float:
        pass

    def get_sympy(self, evaluated:bool=False):
        return None
    
    def is_xscaled(self) -> bool:
        return False
    
    def is_yscaled(self) -> bool:
        return False

    """
    def inrange(self, dp:DataPoint, scale:float=1.) -> bool:
        x_padd = (self.xu - self.xl) * scale
        y_padd = (self.yu - self.yl) * scale
        
        return dp.x >= self.xl - x_padd and dp.x <= self.xu + x_padd and \
                dp.y >= self.yl - y_padd and dp.y <= self.yu + y_padd
    
    def inrange_xy(self, x:float, y:float, scale:float=1.5) -> bool:
        return self.inrange(DataPoint(x, y), scale)
    """

    def _xmap(self, x, toorigin:bool=False) -> float:
        if toorigin: return self._xl + (((x - self.xl) / (self.xu - self.xl)) * (self._xu - self._xl))
        return self.xl + (((x - self._xl) / (self._xu - self._xl)) * (self.xu - self.xl))
    
    def _ymap(self, y:float) -> float:
        return self.yl + (((y - self._yl) / (self._yu - self._yl)) * (self.yu - self.yl))
    
    def evaluate(self, model) -> Evaluation:
        def compute_measures(data:list[DataPoint], data_sst:float) -> dict:
            ssr = 0.
            for dp in data: ssr += (model(dp.x) - dp.y) ** 2  # TODO: can be done efficiently using numpy.
            mse   = ssr / len(data) if len(data) > 0. else 0.
            r2    = (1 - (ssr / data_sst)) if data_sst > 0. else 1.
            return {'mse': mse, 'rmse': math.sqrt(mse), 'r2': r2}
        
        return Evaluation(
            compute_measures(self.data, self.data_sst),
            compute_measures(self.test, self.test_sst) # TODO(take derivatives of model): self.knowledge.evaluate(model)
        )
    
    def evaluate_extra(self, model) -> dict:
        dx = (self.xu - self.xl) / 2
        X = self.spsampler.meshspace(self.xl - dx, self.xu + dx, 500)  # TODO: factorize sample size.
        Y = self.func(X)
        X = X[~np.isnan(Y)]  # remove nan values (where self.func is not defined).
        Y = Y[~np.isnan(Y)]
        Y_pred = model(X)

        ssr = np.sum((Y_pred - Y) ** 2)
        sst = np.sum((Y - Y.mean()) ** 2)

        mse  = ssr / Y.size
        rmse = math.sqrt(mse)
        r2   = 1 - (ssr / sst)

        return {'mse': mse, 'rmse': rmse, 'r2': r2}
    
    def get_plotter(self) -> plotting.DatasetPlotter:
        if self.plotter is None:
            raise RuntimeError(f"Plotting not supported for {self.nvars} dimensions.")
        return self.plotter
    
    def get_name(self) -> str:
        return ''
    
    def get_xlabel(self, xidx:int=0) -> str:
        if self.nvars == 1: return 'x'
        return f"x{xidx}"

    def get_ylabal(self) -> str:
        return 'y'
    
    def get_varnames(self) -> dict[int,str]:
        if self.nvars == 1: return {0: 'x'}
        varnames = {}
        for idx in range(self.nvars):
            varnames[idx] = f"x{idx}"
        return varnames


class NumpyDataset:
    def __init__(self,
                 S:Dataset=None,
                 test:bool=False,
                 nvars:int=None):
        
        assert (S is None) != (nvars is None)
        self.nvars = nvars if S is None else S.nvars
        
        if S is None:
            self.X = np.empty((0,0))
            self.y = np.empty(0)
            self.xl = np.array([-1.]*nvars); self.xu = np.array([1.]*nvars)
            self.yl =  0.; self.yu = 1.
            self.knowledge = None
            self.numlims = numlims.NumericLimits()
            self.numlims.set_bounds(self.xl, self.xu)
            self.spsampler = \
                space.UnidimSpaceSampler(randstate=0) if self.nvars == 1 else \
                space.MultidimSpaceSampler(randstate=0)  # TODO: factorize randstate.
        else:
            XY = S.test if test else S.data
            self.X = np.array( [dp.x for dp in XY] )
            self.y = np.array( [dp.y for dp in XY] )
            self.xl = S.xl; self.xu = S.xu
            self.yl = S.yl; self.yu = S.yu
            self.knowledge = S.knowledge
            self.numlims = S.numlims
            self.spsampler = S.spsampler
        
        if self.X.ndim == 1:
            self.X = np.expand_dims(self.X, axis=1)  # returns a view!
        
        self.plotter = None
        if self.nvars == 1:
            self.plotter = plotting.NumpyDatasetPlotter(self, plotting.Dataset1dPlotterImpl(self))
        elif self.nvars == 2:
            self.plotter = plotting.NumpyDatasetPlotter(self, plotting.Dataset2dPlotterImpl(self))

        self.sst = None
        self.on_y_changed()
    
    def get_size(self) -> int:
        return self.y.size
    
    def is_empty(self) -> bool:
        return self.y.size == 0
    
    def X_from(self, X):
        if self.X.shape == X.shape: self.X[:,:] = X
        else: self.X = np.copy(X)
    
    def y_from(self, y):
        if self.y.size == Y: self.y[:] = y
        else: self.y = np.copy(y)
        self.on_y_changed()
    
    def clear(self):  # remove nan and inf values from X and y.
        # true are kept.
        clear_mask =   np.isfinite(self.X).all(axis=1)  &   np.isfinite(self.y) & \
                     (~np.isnan   (self.X).all(axis=1)) & (~np.isnan   (self.y))
        self.X = self.X[clear_mask,:]
        self.y = self.y[clear_mask]
        self.on_y_changed()
    
    def synchronize_X_limits(self):
        if self.X.size > 0:
            self.xl = self.X.min(axis=0)
            self.xu = self.X.max(axis=0)
    
    def synchronize_y_limits(self):
        if self.y.size > 0:
            self.yl = self.y.min()
            self.yu = self.y.max()
    
    def remove_outliers(self):
        if self.is_empty(): return

        Q1 = np.percentile(self.y, 25, method='midpoint')
        Q3 = np.percentile(self.y, 75, method='midpoint')
        IQR = Q3 - Q1
        upper = Q3+1.5*IQR  # TODO: fix coeffs.
        lower = Q1-1.5*IQR
        
        mask = (self.y >= lower) & (self.y <= upper)  # true are kept.
        self.X = self.X[mask,:]
        self.y = self.y[mask]

        self.on_y_changed()
    
    def compute_yvar(self, x0, dx_ratio:float=0.05) -> float:
        h = (self.xu - self.xl) * dx_ratio * 0.5
        l = x0 - h
        u = x0 + h
        y = self.y[ (self.X > l).all(axis=1) & (self.X < u).all(axis=1) ]
        if y.size <= 1: return self.numlims.EPSILON  # TODO: manage this situation (otherwise in qp.qp_solve we have a division by zero).
        return np.var(y)
    
    def get_y_iqr(self) -> float:
        y25, y75 = np.percentile(self.y, [75 ,25])
        return abs(y75 - y25)
    
    def on_y_changed(self):
        # update sst.
        if self.is_empty():
            self.sst = 0.
            return
        y_mean = self.y.mean()
        self.sst = np.sum( (self.y - y_mean) ** 2 )
    
    def evaluate(self, model) -> Evaluation:
        n     = self.get_size()
        ssr   = np.sum( (model(self.X) - self.y) ** 2 )
        mse   = 0. if n == 0 else ssr / n
        r2    = 1 - (ssr / self.sst) if self.sst > 0. else 1.
        return {'mse': mse, 'rmse': math.sqrt(mse), 'r2': r2}
    
    def get_plotter(self) -> plotting.DatasetPlotter:
        if self.plotter is None:
            raise RuntimeError(f"Plotting not supported for {self.nvars} dimensions.")
        return self.plotter


class Dataset1d(Dataset):
    def __init__(self, xl, xu):
        assert type(xl) is float and type(xu) is float
        super().__init__(
            nvars=1, xl=xl, xu=xu,
            spsampler=space.UnidimSpaceSampler(),
            plotter=plotting.DatasetPlotter(self, plotting.Dataset1dPlotterImpl(self)))


class Datasetnd(Dataset):
    def __init__(self, xl, xu):
        if type(xl) is list: xl = np.array(xl, dtype=float)
        if type(xu) is list: xu = np.array(xu, dtype=float)
        assert not np.isscalar(xl) and not np.isscalar(xu) and \
               xl.ndim == 1 and xu.ndim == 1 and xl.size == xu.size and xl.size > 1
        
        nvars = xl.size
        super().__init__(
            nvars=nvars, xl=xl, xu=xu,
            spsampler=space.MultidimSpaceSampler(),
            plotter=plotting.DatasetPlotter(self, plotting.Dataset2dPlotterImpl(self)) if nvars == 2 else None)