import random
import math
import numpy as np
import sympy
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numbs


class DataPoint:
    def __init__(self, x:float, y:float) -> None:
        self.x = x
        self.y = y
    
    def distance(self, other) -> float:
        return math.sqrt((other.x-self.x)**2 + (other.y-self.y)**2)


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
    def __init__(self, dataset=None) -> None:
        self.dataset = dataset
        self.derivs = {}
        self.sign = {}
        self.symm = {}
        self.noroot = set()
    
    def add_deriv(self, d:int, xy:DataPoint):
        if d not in self.derivs.keys():
            self.derivs[d] = []
        self.derivs[d].append(xy)
    
    def add_sign(self, d:int, l:float, u:float, sign:str='+', th:float=0):
        if d not in self.sign.keys():
            self.sign[d] = []
        self.sign[d].append((l,u,sign,th))
    
    def add_symm(self, d:int, x:float, iseven:bool=True):
        self.symm[d] = (x, iseven)
    
    def add_noroot(self, d:int):
        self.noroot.add(d)
    
    def get_mesh(self, sample_size:int=20) -> np.array:
        deriv_points = []
        for dp in self.derivs.values():
            deriv_points.append(dp.x)
        return np.concatenate((
            np.linspace(self.dataset.xl, self.dataset.xu, max(0, sample_size - len(deriv_points))),
            np.array(deriv_points))).sort()
    
    def evaluate(self, model:tuple[callable,callable,callable]) -> dict:
        n   = {0: 0, 1: 0, 2: 0}
        ssr = {0: 0, 1: 0, 2: 0}

        # intersection points.
        for derivdeg, dps in self.derivs.items():
            n[derivdeg] += len(dps)
            X = np.array( [dp.x for dp in dps] )
            Y = np.array( [dp.y for dp in dps] )
            ssr[derivdeg] += np.sum( (model[derivdeg](X) - Y) ** 2 )
        
        # positivity constraints.
        for derivdeg, constrs in self.sign.items():
            for (_l,_u,sign,th) in constrs:
                l = _l + numbs.EPSILON
                u = _u - numbs.EPSILON
                if l > u: continue
                X = np.linspace(l, u, 1 if l == u else 20)  # TODO: factorize sample size.
                n[derivdeg] += X.size
                model_Y = model[derivdeg](X)
                ssr[derivdeg] += np.sum( ( np.minimum(0, model_Y - th) if sign == '+' else np.maximum(0, model_Y - th) ) ** 2 )
        
        # symmetry constraints.
        for derivdeg, (x0, iseven) in self.symm.items():
            X = np.linspace(x0 + numbs.EPSILON, numbs.INFTY, 20)  # TODO: factorize sample size.
            n[derivdeg] += X.size
            model_Y1 = model[derivdeg](X)
            model_Y2 = model[derivdeg](x0-(X-x0))
            ssr[derivdeg] += np.sum( ( (model_Y1 - model_Y2) if iseven else (model_Y1 + model_Y2) ) ** 2 )
        
        return {'mse0': (ssr[0]/n[0]) if n[0] > 0. else 0.,
                'mse1': (ssr[1]/n[1]) if n[1] > 0. else 0.,
                'mse2': (ssr[2]/n[2]) if n[2] > 0. else 0.}
    
    def plot(self):
        if 0 in self.derivs.keys():
            for xy in self.derivs[0]:
                plt.plot(xy.x, xy.y, 'rx', markersize=10)
        
        if 0 in self.sign.keys():
            for (l,u,s,_) in self.sign[0]:
                plt.axvspan(l, u, alpha=0.05, color='g' if s == '+' else 'r')
            

def compute_sst(dpoints:list) -> float:
    if len(dpoints) == 0: return 0.
    y_mean = 0.
    for dp in dpoints: y_mean += dp.y
    y_mean /= len(dpoints)

    sst = 0.
    for dp in dpoints: sst += (dp.y - y_mean) ** 2
    return sst


class Dataset:
    def __init__(self) -> None:
        self.data = []
        self.test = []
        self.xl = -1.
        self.xu = 1.
        self.yl = 0.
        self.yu = 1.
        self._xl = self.xl  # used for scaling.
        self._xu = self.xu
        self._yl = self.yl
        self._yu = self.yu
        self.knowledge = DataKnowledge(self)
        self.data_sst = 0.
        self.test_sst = 0.
        self.__sorted_data__ = None
        self.__x_map__ = None
    
    def sample(self, size:int=100, noise:float=0., mesh:bool=False):
        y_noise = (self.yu - self.yl) * noise * 0.5
        
        X = \
            np.linspace(self.xl, self.xu, size).tolist() if mesh else \
            [random.uniform(self.xl, self.xu) for _ in range(size)]
        
        for x in X:
            y = self.func(x) + (0. if noise == 0. else random.gauss(sigma=y_noise))
            self.data.append(DataPoint(x, y))
        
        self._on_data_changed()
    
    def load(self, filename:str):
        csvfile = open(filename)
        csvreader = csv.reader(csvfile)
        for entry in csvreader:
            x = float(entry[0])
            y = float(entry[1])
            if self.is_xscaled(): x = self._xmap(x)
            if self.is_yscaled(): y = self._ymap(y)
            self.data.append(DataPoint(x, y))
        csvfile.close()
        self._on_data_changed()
    
    def erase(self, x_from, x_to):
        self.test = []
        new_data = []
        for dp in self.data:
            if dp.x < x_from or dp.x > x_to: new_data.append(dp)
        self.data = new_data
        self._on_data_changed()

    def split(self, train_size:float=0.7, seed:int=0):
        if len(self.data) > 1:
            train, test = train_test_split(self.data, train_size=train_size, random_state=seed)
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
            d += math.sqrt( ((other_dp.x-dp.x) ** 2) + ((other_dp.y-dp.y) ** 2) )
        max_d = math.sqrt( ((self.xu-self.xl) ** 2) + ((self.yu-self.yl) ** 2) )
        return max_d - (d / n)
    
    def compute_yvar(self, x0:float, dx_ratio:float=0.05) -> float:
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
            if dp.x > l and dp.x < u: Y.append(dp.y)
        if len(Y) <= 1: return numbs.EPSILON  # TODO: manage this situation (otherwise in qp.qp_solve we have a division by zero).
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

    def inrange(self, dp:DataPoint, scale:float=1.) -> bool:
        x_padd = (self.xu - self.xl) * scale
        y_padd = (self.yu - self.yl) * scale
        
        return dp.x >= self.xl - x_padd and dp.x <= self.xu + x_padd and \
                dp.y >= self.yl - y_padd and dp.y <= self.yu + y_padd 
    
    def inrange_xy(self, x:float, y:float, scale:float=1.5) -> bool:
        return self.inrange(DataPoint(x, y), scale)
    
    def _xmap(self, x:float, toorigin:bool=False) -> float:
        if toorigin: return self._xl + (((x - self.xl) / (self.xu - self.xl)) * (self._xu - self._xl))
        return self.xl + (((x - self._xl) / (self._xu - self._xl)) * (self.xu - self.xl))
    
    def _ymap(self, y:float) -> float:
        return self.yl + (((y - self._yl) / (self._yu - self._yl)) * (self.yu - self.yl))
    
    def evaluate(self, model:callable) -> Evaluation:
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
    
    def evaluate_extra(self, model:callable) -> dict:
        dx = (self.xu - self.xl) / 2
        X = np.linspace(self.xl - dx, self.xu + dx, 500)  # TODO: factorize sample size.
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

    
    def plot(self, plot_data:bool=True, plot_knowldege:bool=False,
             width:int=10, height:int=8,
             plotref:bool=True, model:callable=None,
             zoomout:float=1.,
             savename:str=None):
        
        plt.figure(2, figsize=[width,height])
        plt.clf()

        if plot_data:
            def plot_data_points(data:list[DataPoint], marker:str, label:str):
                data_labeld = False
                for dp in data:
                    if data_labeld: plt.plot(dp.x, dp.y, marker, markersize=2)
                    else:
                        plt.plot(dp.x, dp.y, marker, markersize=2, label=label)
                        data_labeld = True
            plot_data_points(self.data, 'bo', 'Training data')
            plot_data_points(self.test, 'mo', 'Test data')
        
        if plot_knowldege:
            self.knowledge.plot()

        xstep_zoomout = (self.xu - self.xl) * (zoomout - 1) * .5
        ystep_zoomout = (self.yu - self.yl) * (zoomout - 1) * .5
        xl = self.xl - xstep_zoomout
        xu = self.xu + xstep_zoomout
        yl = self.yl - ystep_zoomout
        yu = self.yu + ystep_zoomout
        sample_size = int(500 * zoomout)

        if plotref:
            x = np.linspace(xl, xu, sample_size)
            plt.plot(x, self.func(x), linestyle='dashed', linewidth=2, color='black', label='Reference model')
        
        if model is not None:
            x = np.linspace(xl, xu, sample_size)
            plt.plot(x, model(x), linewidth=2, color='green', label='Model')
        
        plt.xlim(xl, xu)
        plt.ylim(yl, yu)
        plt.grid()
        plt.legend(loc='upper right', fontsize=14)
        plt.xlabel(self.get_xlabel())
        plt.ylabel(self.get_ylabal())

        if savename is not None:
            plt.savefig(savename, bbox_inches='tight')
    
    def get_name(self) -> str:
        return ''
    
    def get_xlabel(self) -> str:
        return 'x'

    def get_ylabal(self) -> str:
        return 'y'
    


class NumpyDataset:
    def __init__(self, S:Dataset=None, test:bool=False):
        if S is None:
            self.X = np.empty(0)
            self.Y = np.empty(0)
            self.xl = -1.; self.xu = 1.
            self.yl =  0.; self.yu = 1.
            self.knowledge = None
        else:
            XY = S.test if test else S.data
            self.X = np.array( [dp.x for dp in XY] )
            self.Y = np.array( [dp.y for dp in XY] )
            self.xl = S.xl; self.xu = S.xu
            self.yl = S.yl; self.yu = S.yu
            self.knowledge = S.knowledge
        self.sst = None
        self.on_Y_changed()
    
    def get_size(self) -> int:
        return self.X.size
    
    def is_empty(self) -> bool:
        return self.X.size == 0
    
    def X_from(self, X):
        if self.X.size == X.size: self.X[:] = X
        else: self.X = np.copy(X)
    
    def Y_from(self, Y):
        if self.Y.size == Y: self.Y[:] = Y
        else: self.Y = np.copy(Y)
        self.on_Y_changed()
    
    def clear(self):  # remove nan and inf values from X and Y.
        # true are kept.
        clear_mask =   np.isfinite(self.X)  &   np.isfinite(self.Y) & \
                     (~np.isnan   (self.X)) & (~np.isnan   (self.Y))
        self.X = self.X[clear_mask]
        self.Y = self.Y[clear_mask]
        self.on_Y_changed()
    
    def synchronize_X_limits(self):
        if self.X.size > 0:
            self.xl = self.X.min()
            self.xu = self.X.max()
    
    def synchronize_Y_limits(self):
        if self.Y.size > 0:
            self.yl = self.Y.min()
            self.yu = self.Y.max()
    
    def remove_outliers(self):
        if self.is_empty(): return

        Q1 = np.percentile(self.Y, 25, method='midpoint')
        Q3 = np.percentile(self.Y, 75, method='midpoint')
        IQR = Q3 - Q1
        upper = Q3+70.5*IQR  # TODO: fix coeffs.
        lower = Q1-70.5*IQR
        
        mask = (self.Y >= lower) & (self.Y <= upper)  # true are kept.
        self.X = self.X[mask]
        self.Y = self.Y[mask]

        self.on_Y_changed()
    
    def compute_yvar(self, x0:float, dx_ratio:float=0.05) -> float:
        h = (self.xu - self.xl) * dx_ratio * 0.5
        l = x0 - h
        u = x0 + h
        Y = self.Y[ (self.X > l) & (self.X < u) ]
        if Y.size <= 1: return numbs.EPSILON  # TODO: manage this situation (otherwise in qp.qp_solve we have a division by zero).
        return np.var(Y)
    
    def on_Y_changed(self):
        # update sst.
        if self.is_empty():
            self.sst = 0.
            return
        y_mean = self.Y.mean()
        self.sst = np.sum( (self.Y - y_mean) ** 2 )
    
    def evaluate(self, model:callable) -> Evaluation:
        n     = self.get_size()
        ssr   = np.sum( (model(self.X) - self.Y) ** 2 )
        mse   = 0. if n == 0 else ssr / n
        r2    = 1 - (ssr / self.sst) if self.sst > 0. else 1.
        return {'mse': mse, 'rmse': math.sqrt(mse), 'r2': r2}
    
    def plot(self, width:int=10, height:int=8, plotref:bool=True):
        plt.figure(2, figsize=[width,height])
        plt.clf()
        plt.plot(self.X, self.Y, 'bo', markersize=2)
        plt.xlim(self.xl, self.xu)
        plt.ylim(self.yl, self.yu)
        plt.grid()


class MockDataset(Dataset):   
    def __init__(self) -> None:
        super().__init__()
        self.xl = 0.#0.8
        self.xu = 1.#10
        self.yl = 0.#-1
        self.yu = 1.#9
     
    def func(self, x: float) -> float:
        return (x**3 -2*x + 1) / (x*3 + x -1) #np.sin(x) + 1  #x / (x**2)#(x+2) / (x**2 + x + 1)
    
    def get_name(self) -> str:
        return 'Mock'


class PolyDataset(Dataset):   
    def __init__(self) -> None:
        super().__init__()
        """self.xl = -2.5
        self.xu = 2.5
        self.yl = 0.
        self.yu = 1.5"""
        self.xl = 0.01
        self.xu =  4
        self.yl =  0.01
        self.yu =  16
     
    def func(self, x: float) -> float:
        return x **2
        #return 0.2*x**4 -1*x**2 + 1.3
    
    def get_name(self) -> str:
        return 'Poly'


class TrigonDataset(Dataset):   
    def __init__(self) -> None:
        super().__init__()
        self.xl = -5.
        self.xu = 5.
        self.yl = -1.
        self.yu = 1.

        """self.knowledge.add_deriv(0, DataPoint( 0., 0.))
        self.knowledge.add_deriv(0, DataPoint(  .5*math.pi,  1.))
        self.knowledge.add_deriv(0, DataPoint( -.5*math.pi, -1.))
        self.knowledge.add_deriv(0, DataPoint( 1.5*math.pi, -1.))
        self.knowledge.add_deriv(0, DataPoint(-1.5*math.pi,  1.))

        self.knowledge.add_deriv(1, DataPoint(  .5*math.pi,  0.))
        self.knowledge.add_deriv(1, DataPoint( -.5*math.pi,  0.))
        self.knowledge.add_deriv(1, DataPoint( 1.5*math.pi,  0.))
        self.knowledge.add_deriv(1, DataPoint(-1.5*math.pi,  0.))"""
     
    def func(self, x: float) -> float:
        return np.sin(x)
    
    def get_name(self) -> str:
        return 'Trigon'


class MagmanDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xl = -0.075
        self.xu =  0.075
        self.yl = -0.25
        self.yu =  0.25
        
        self.c1 = .00032
        self.c2 = .000305
        self.i  = .000004
        peak_x  = 0.00788845
        
        # intersection points
        """self.knowledge.add_deriv(0, DataPoint( 0., 0.))
        self.knowledge.add_deriv(0, DataPoint(-peak_x, self.func(-peak_x)))
        self.knowledge.add_deriv(0, DataPoint( peak_x, self.func( peak_x)))
        self.knowledge.add_deriv(0, DataPoint(self.xl, self.func(self.xl)))
        self.knowledge.add_deriv(0, DataPoint(self.xu, self.func(self.xu)))"""

        # known (first) derivatives
        """self.knowledge.add_deriv(1, DataPoint(-peak_x,  0.))
        self.knowledge.add_deriv(1, DataPoint( peak_x,  0.))"""

        #
        # positivity/negativity contraints
        #
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, -0.00001, '+')
        self.knowledge.add_sign(0, 0.00001, self.xu, '-')
    
        # monotonically increasing/decreasing
        """self.knowledge.add_sign(1, self.xl, -0.01, '+')
        #self.knowledge.add_sign(1, -peak_x+0.1, peak_x-0.1, '-')
        self.knowledge.add_sign(1, -0.01, self.xu, '+')"""

        # concavity/convexity
        """self.knowledge.add_sign(2, self.xl, -0.01, '+')
        self.knowledge.add_sign(2, 0.01, self.xu, '-')"""

    def func(self, x: float) -> float:
        return -self.i*self.c1*x / (x**2 + self.c2)**3
    
    def get_sympy(self, evaluated:bool=False):
        x  = sympy.Symbol('x')
        i  = sympy.Symbol('i')
        c1 = sympy.Symbol('c1')
        c2 = sympy.Symbol('c2')
        expr = -i*c1*x / (x**2 + c2)**3
        if evaluated: return expr.subs( {i:self.i, c1:self.c1, c2:self.c2} )
        return expr
        #return '-\frac{i \cdot c_1 \cdot x}{\left(x^2 + c_2\right)^3}'
    
    def get_name(self) -> str:
        return 'magman'
    
    def get_xlabel(self) -> str:
        return 'distance [m] (x)'

    def get_ylabal(self) -> str:
        return 'force [N] (y)'


class MagmanDatasetScaled(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xl = -2.
        self.xu = 2.
        self.yl = -2.
        self.yu = 2.

        self._xl = -0.075
        self._xu =  0.075
        self._yl = -0.25
        self._yu =  0.25

        #self.c1 = 1.4
        #self.c2 = 1.2
        #self.i = 7.
        #peak_x = 0.5

        self.c1 = .00032
        self.c2 = .000305
        self.i = .000004
        peak_x = self._xmap(0.00781024967)
        infl_x = self._xmap(0.01352774925)
        #peak_x = 0.20827333333333353
        
        # intersection points
        self.knowledge.add_deriv(0, DataPoint( 0., 0.))
        self.knowledge.add_deriv(0, DataPoint(-peak_x, self.func(-peak_x)))
        self.knowledge.add_deriv(0, DataPoint( peak_x, self.func( peak_x)))
        self.knowledge.add_deriv(0, DataPoint(self.xl, self.func(self.xl)))
        self.knowledge.add_deriv(0, DataPoint(self.xu, self.func(self.xu)))
        #infty = 10
        #self.knowledge.add_deriv(0, DataPoint(-infty, 0))
        #self.knowledge.add_deriv(0, DataPoint(+infty, 0))

        # known (first) derivatives
        self.knowledge.add_deriv(1, DataPoint(-peak_x,  0.))
        self.knowledge.add_deriv(1, DataPoint( peak_x,  0.))

        #
        # positivity/negativity contraints
        #
        
        # known positivity/negativity
        #self.knowledge.add_sign(0, self.xl, -0.001, '+')
        #self.knowledge.add_sign(0, 0.001, self.xu, '-')
        self.knowledge.add_sign(0, -numbs.INFTY, 0, '+')
        self.knowledge.add_sign(0, 0, numbs.INFTY, '-')
    
        # monotonically increasing/decreasing
        #self.knowledge.add_sign(1, self.xl, -peak_x, '+')
        #self.knowledge.add_sign(1, -peak_x, peak_x, '-')
        #self.knowledge.add_sign(1, peak_x, self.xu, '+')
        self.knowledge.add_sign(1, -numbs.INFTY, -peak_x, '+')
        self.knowledge.add_sign(1, -peak_x, peak_x, '-')
        self.knowledge.add_sign(1, peak_x, numbs.INFTY, '+')

        # concavity/convexity
        #self.knowledge.add_sign(2, self.xl, -0.4, '+')
        #self.knowledge.add_sign(2, 0.4, self.xu, '-')
        
        self.knowledge.add_sign(2, -numbs.INFTY, -infl_x, '+')
        self.knowledge.add_sign(2, -infl_x, 0, '-')
        self.knowledge.add_sign(2, 0, infl_x, '+')
        self.knowledge.add_sign(2, infl_x, numbs.INFTY, '-')

        # symmetry
        self.knowledge.add_symm(0, 0, iseven=False)
        self.knowledge.add_symm(1, 0, iseven=True )
        self.knowledge.add_symm(2, 0, iseven=False)

    def func(self, x: float) -> float:
        x = self._xmap(x, toorigin=True)
        y = -self.i*self.c1*x / (x**2 + self.c2)**3
        return self._ymap(y)
    
    def get_sympy(self, evaluated:bool=False):
        x  = sympy.Symbol('x')
        i  = sympy.Symbol('i')
        c1 = sympy.Symbol('c1')
        c2 = sympy.Symbol('c2')
        if evaluated:
            x = self._xmap(x, toorigin=True)
        expr = -i*c1*x / (x**2 + c2)**3
        if evaluated:
            expr = self._ymap(expr)
            return expr.subs( {i:self.i, c1:self.c1, c2:self.c2} )
        return expr
        #return '-\frac{i \cdot c_1 \cdot x}{\left(x^2 + c_2\right)^3}'

    def deriv(self, x: float) -> float:
        x = self._xmap(x, toorigin=True)
        y = (6.4e-9 * x**2 - 3.904e-13) / (x**2 + 0.000305) ** 4
        return self._ymap(y)
    
    def get_name(self) -> str:
        return 'magman'
    
    def get_xlabel(self) -> str:
        return 'distance [m] (x)'

    def get_ylabal(self) -> str:
        return 'force [N] (y)'
    
    def is_xscaled(self) -> bool:
        return True
    
    def is_yscaled(self) -> bool:
        return True


"""        
class MagmanDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xl = -10.
        self.xu = 10.
        self.yl = -3.
        self.yu = 3.

        self.c1 = 10
        self.c2 = 0.6
        self.i = 0.2

        # intersection points
        self.knowledge.add_deriv(0, DataPoint( 0., 0.))
        self.knowledge.add_deriv(0, DataPoint(-5, self.func(-5)))
        self.knowledge.add_deriv(0, DataPoint( 5, self.func( 5)))
        self.knowledge.add_deriv(0, DataPoint(self.xl, self.func(self.xl)))
        self.knowledge.add_deriv(0, DataPoint(self.xu, self.func(self.xu)))

        # known (first) derivatives
        self.knowledge.add_deriv(1, DataPoint(-5,  0.))
        self.knowledge.add_deriv(1, DataPoint( 5,  0.))

        #
        # positivity/negativity contraints
        #
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, 0.01, '+')
        self.knowledge.add_sign(0, 0.0001, self.xu, '-')
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, self.xl, -5.5, '+')
        #self.knowledge.add_sign(1, -5, 5, '-')
        self.knowledge.add_sign(1, 5.5, self.xu, '+')

        # concavity/convexity
        self.knowledge.add_sign(2, self.xl, -5.5, '+')
        self.knowledge.add_sign(2, 5.5, self.xu, '-')

    def func(self, x: float) -> float:
        return -self.i*self.c1*x / (x**2 + self.c2)**3
"""

class HEADataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xl = 0.
        self.xu = 10.
        self.yl = -1.
        self.yu = 1.
    
    def func(self, x: float) -> float:
        if type(x) == float: return self.__func(x)
        y = []
        for _x in x: y.append(self.__func(_x))
        return y

    def __func(self, x: float) -> float:
        return math.e**(-x) * x**3 * math.cos(x) * math.sin(x) * (math.cos(x) * math.sin(x)**2 - 1)
    
    def get_name(self) -> str:
        return 'hea'


class ABSDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xl = 0.
        self.xu = 1.
        self.yl = 0.
        self.yu = 0.45

        self.m = 6.67 #407.75
        self.g = 0.15 #9.81
        self.b = 55.56
        self.c = 1.35
        self.d = 0.4
        self.e = 0.52

        #
        # prior knowledge
        #

        peak_x = 0.0618236
        infl_x = 0.10629991
        peak_y = self.func(peak_x)

        # intersection points
        self.knowledge.add_deriv(0, DataPoint(0., 0.))
        self.knowledge.add_deriv(0, DataPoint(peak_x, peak_y))
        self.knowledge.add_deriv(1, DataPoint(peak_x, 0.))
        
        # known positivity/negativity
        #self.knowledge.add_sign(0, self.xl, numbs.INFTY, '+')
        self.knowledge.add_sign(0, self.xl, self.xu, '+')

        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, self.xl, peak_x, '+')
        #self.knowledge.add_sign(1, peak_x, numbs.INFTY, '-')
        self.knowledge.add_sign(1, peak_x, self.xu, '-')

        # concavity
        self.knowledge.add_sign(2, self.xl, infl_x, '-')
        #self.knowledge.add_sign(2, peak_x, numbs.INFTY, '+')
        self.knowledge.add_sign(2, infl_x, self.xu, '+')
    
    def func(self, x: float) -> float:
        return self.m * self.g * self.d * np.sin(self.c * np.arctan(self.b * (1 - self.e) * x + self.e * np.arctan(self.b * x)))
    
    def get_sympy(self, evaluated:bool=False):
        x = sympy.Symbol('kappa')
        b = sympy.Symbol('b')
        c = sympy.Symbol('c')
        d = sympy.Symbol('d')
        e = sympy.Symbol('e')
        g = sympy.Symbol('g')
        m = sympy.Symbol('m')
        expr = m * g * d * sympy.sin(c * sympy.atan(b * (1 - e) * x + e * sympy.atan(b * x)))
        if evaluated: return expr.subs( {b:self.b, c:self.c, d:self.d, e:self.e, g:self.g, m:self.m} )
        return expr
        #return 'm \cdot g \cdot d \cdot \sin\left( c \cdot \arctan\left( b \cdot (1-e)^x + e^{\arctan\left(b \cdot x\right)}\right)\right)'
    
    def get_name(self) -> str:
        return 'abs'
    
    def get_xlabel(self) -> str:
        return 'k'

    def get_ylabal(self) -> str:
        return 'F(k) [N]'


class OneOverXDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xl = 1e-10
        self.xu = 5.
        self.yl = 0.
        self.yu = 5.

        #
        # prior knowledge
        #

        # intersection points
        self.knowledge.add_deriv(0, DataPoint( 1., 1.))
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, numbs.INFTY, '+')

        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, self.xl, numbs.INFTY, '-')

        # concavity
        self.knowledge.add_sign(2, self.xl, numbs.INFTY, '+')
    
    def func(self, x: float) -> float:
        return 1 / x
    
    def get_sympy(self, evaluated:bool=False):
        x = sympy.Symbol('x')
        return 1 / x
        #return '\frac{1}{x}'
    
    def get_name(self) -> str:
        return '1/x'


class ABSDatasetScaled(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xl = 0.
        self.xu = 1.
        self.yl = 0.
        self.yu = 10

        self._xl = 0.
        self._xu = 1.
        self._yl = 0.
        self._yu = 0.45

        self.m = 6.67
        self.g = 0.15
        self.b = 55.56
        self.c = 1.35
        self.d = 0.4
        self.e = 0.52

        #
        # prior knowledge
        #

        peak_x = self._xmap(0.0618236)
        infl_x = self._xmap(0.10629991)
        peak_y = self.func(peak_x)

        # intersection points
        self.knowledge.add_deriv(0, DataPoint( 0., 0.))
        self.knowledge.add_deriv(0, DataPoint(peak_x, peak_y))
        self.knowledge.add_deriv(1, DataPoint(peak_x, 0.))
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, numbs.INFTY, '+')

        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, self.xl, peak_x, '+')
        self.knowledge.add_sign(1, peak_x, numbs.INFTY, '-')

        # concavity
        self.knowledge.add_sign(2, self.xl, infl_x, '-')
        self.knowledge.add_sign(2, infl_x, numbs.INFTY, '+')
    
    def func(self, x: float) -> float:
        x = self._xmap(x, toorigin=True)
        y = self.m * self.g * self.d * np.sin(self.c * np.arctan(self.b * (1 - self.e) * x + self.e * np.arctan(self.b * x)))
        return self._ymap(y)
    
    def get_sympy(self, evaluated:bool=False):
        x = sympy.Symbol('kappa')
        b = sympy.Symbol('b')
        c = sympy.Symbol('c')
        d = sympy.Symbol('d')
        e = sympy.Symbol('e')
        g = sympy.Symbol('g')
        m = sympy.Symbol('m')
        if evaluated:
            x = self._xmap(x, toorigin=True)
        expr = m * g * d * sympy.sin(c * sympy.atan(b * (1 - e) * x + e * sympy.atan(b * x)))
        if evaluated:
            expr = self._ymap(expr)
            return expr.subs( {b:self.b, c:self.c, d:self.d, e:self.e, g:self.g, m:self.m} )
        return expr
        #return 'm \cdot g \cdot d \cdot \sin\left( c \cdot \arctan\left( b \cdot (1-e)^x + e^{\arctan\left(b \cdot x\right)}\right)\right)'
    
    def get_name(self) -> str:
        return 'abs'
    
    def get_xlabel(self) -> str:
        return 'k'

    def get_ylabal(self) -> str:
        return 'F(k) [N]'
    
    def is_yscaled(self) -> bool:
        return True