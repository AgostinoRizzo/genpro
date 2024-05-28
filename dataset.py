import random
import math
import numpy as np
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
    
    def evaluate(self, model:callable) -> float:
        n = 0
        ssr = 0

        def compute_model_output(model:callable, x:float, deriv:int=0) -> float:  # TODO: use derivative stree directly.
            if deriv == 0: return model(x)
            return (compute_model_output(model, x+numbs.STEPSIZE, deriv-1) - compute_model_output(model, x, deriv-1)) / numbs.STEPSIZE
        
        # intersection points.
        for derivdeg, dps in self.derivs.items():
            n += len(dps)
            for dp in dps:
                ssr += (compute_model_output(model, dp.x, derivdeg) - dp.y) ** 2
        
        # positivity constraints.
        for derivdeg, constrs in self.sign.items():
            for (_l,_u,sign,th) in constrs:
                l = _l + numbs.EPSILON
                u = _u - numbs.EPSILON
                if l > u: continue
                X = np.linspace(l, u, 1 if l == u else 20)  # TODO: factorize sample size.
                n += X.size
                for x in X:
                    model_y = compute_model_output(model, x, derivdeg)
                    ssr += ( min(0, model_y - th) if sign == '+' else max(0, model_y - th) ) ** 2
        
        # symmetry constraints.
        for derivdeg, (x0, iseven) in self.symm.items():
            X = np.linspace(x0 + numbs.EPSILON, numbs.INFTY, 20)  # TODO: factorize sample size.
            n += X.size
            for x in X:
                model_y1 = compute_model_output(model, x, derivdeg)
                model_y2 = compute_model_output(model, x0-(x-x0), derivdeg)
                ssr += ( (model_y1 - model_y2) if iseven else (model_y1 + model_y2) ) ** 2
        
        return ssr / n    
    
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
        raise RuntimeError('Load from CSV file not supported.')
    
    def erase(self, x_from, x_to):
        self.test = []
        new_data = []
        for dp in self.data:
            if dp.x < x_from or dp.x > x_to: new_data.append(dp)
        self.data = new_data
        self._on_data_changed()

    def split(self, train_size:float=0.7, seed:int=0):
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

    def inrange(self, dp:DataPoint, scale:float=1.) -> bool:
        x_padd = (self.xu - self.xl) * scale
        y_padd = (self.yu - self.yl) * scale
        
        return dp.x >= self.xl - x_padd and dp.x <= self.xu + x_padd and \
                dp.y >= self.yl - y_padd and dp.y <= self.yu + y_padd 
    
    def inrange_xy(self, x:float, y:float, scale:float=1.5) -> bool:
        return self.inrange(DataPoint(x, y), scale)
    
    def evaluate(self, model:callable) -> tuple[float,float,float]:  # TODO: for now just mse and r2 over training data.
        ssr = 0.
        for dp in self.data: ssr += (model(dp.x) - dp.y) ** 2  # TODO: can be done efficiently using numpy.
        
        mse   = ssr / len(self.data)
        r2    = 1 - (ssr / self.data_sst)
        k_mse = self.knowledge.evaluate(model)

        return mse, r2, k_mse
    
    def plot(self, plot_data: bool=True, width:int=10, height:int=8, plotref:bool=True):
        plt.figure(2, figsize=[width,height])
        plt.clf()

        if plot_data:
            data_labeld = False
            for dp in self.data:
                if data_labeld: plt.plot(dp.x, dp.y, 'bo', markersize=2)
                else:
                    plt.plot(dp.x, dp.y, 'bo', markersize=2, label='Training data')
                    data_labeld = True
        
        self.knowledge.plot()

        if plotref:
            x = np.linspace(self.xl, self.xu, 100)
            plt.plot(x, self.func(x), linestyle='dashed', linewidth=2, color='black', label='Reference model')
        plt.xlim(self.xl, self.xu)
        plt.ylim(self.yl, self.yu)
        plt.grid()
        plt.legend(loc='upper right', fontsize=14)
        plt.xlabel(self.get_xlabel())
        plt.ylabel(self.get_ylabal())
    
    def get_xlabel(self) -> str:
        return ''

    def get_ylabal(self) -> str:
        return ''


class MockDataset(Dataset):   
    def __init__(self) -> None:
        super().__init__()
        self.xl = 0.#0.8
        self.xu = 1.#10
        self.yl = 0.#-1
        self.yu = 1.#9
     
    def func(self, x: float) -> float:
        return (x**3 -2*x + 1) / (x*3 + x -1) #np.sin(x) + 1  #x / (x**2)#(x+2) / (x**2 + x + 1)


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
    
    def load(self, filename:str):
        csvfile = open(filename)
        csvreader = csv.reader(csvfile)
        for entry in csvreader:
            self.data.append(DataPoint(float(entry[0]), float(entry[1])))
        csvfile.close()
        self._on_data_changed()


class MagmanDatasetScaled(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xl = -2.
        self.xu = 2.
        self.yl = -2.
        self.yu = 2.

        self.__xl = -0.075
        self.__xu =  0.075
        self.__yl = -0.25
        self.__yu =  0.25

        #self.c1 = 1.4
        #self.c2 = 1.2
        #self.i = 7.
        #peak_x = 0.5

        self.c1 = .00032
        self.c2 = .000305
        self.i = .000004
        #peak_x = 0.00788845
        peak_x = 0.20827333333333353
        
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
        self.knowledge.add_sign(2, -numbs.INFTY, -0.4, '+')
        self.knowledge.add_sign(2, -0.4, 0, '-')
        self.knowledge.add_sign(2, 0, 0.4, '+')
        self.knowledge.add_sign(2, 0.4, numbs.INFTY, '-')

        # symmetry
        self.knowledge.add_symm(0, 0, iseven=False)
        self.knowledge.add_symm(1, 0, iseven=True )
        self.knowledge.add_symm(2, 0, iseven=False)

    def func(self, x: float) -> float:
        x = self.__xmap(x, toorigin=True)
        y = -self.i*self.c1*x / (x**2 + self.c2)**3
        return self.__ymap(y)

    def deriv(self, x: float) -> float:
        x = self.__xmap(x, toorigin=True)
        y = (6.4e-9 * x**2 - 3.904e-13) / (x**2 + 0.000305) ** 4
        return self.__ymap(y)
    
    def load(self, filename:str):
        csvfile = open(filename)
        csvreader = csv.reader(csvfile)
        for entry in csvreader:
            self.data.append(DataPoint(self.__xmap(float(entry[0])), self.__ymap(float(entry[1]))))
        csvfile.close()
        self._on_data_changed()
    
    def __xmap(self, x:float, toorigin:bool=False) -> float:
        if toorigin: return self.__xl + (((x - self.xl) / (self.xu - self.xl)) * (self.__xu - self.__xl))
        return self.xl + (((x - self.__xl) / (self.__xu - self.__xl)) * (self.xu - self.xl))
    
    def __ymap(self, y:float) -> float:
        return self.yl + (((y - self.__yl) / (self.__yu - self.__yl)) * (self.yu - self.yl)) 
    
    def get_xlabel(self) -> str:
        return 'distance [m] (x)'

    def get_ylabal(self) -> str:
        return 'force [N] (y)'


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


class ABSDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xl = 0.
        self.xu = 1.
        self.yl = 0.
        self.yu = 0.45

        #
        # prior knowledge
        #

        peak_x = 0.06182
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
        self.knowledge.add_sign(2, self.xl, peak_x, '-')
        #self.knowledge.add_sign(2, peak_x, numbs.INFTY, '+')
        self.knowledge.add_sign(2, peak_x, self.xu, '+')
    
    def func(self, x: float) -> float:
        m = 6.67 #407.75
        g = 0.15 #9.81
        b = 55.56
        c = 1.35
        d = 0.4
        e = 0.52
        return m * g * d * np.sin(c * np.arctan(b * (1 - e) * x + e * np.arctan(b * x)))


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


class ABSDatasetScaled(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xl = 0.
        self.xu = 1.
        self.yl = 0.
        self.yu = 10

        self.__xl = 0.
        self.__xu = 1.
        self.__yl = 0.
        self.__yu = 0.45

        #
        # prior knowledge
        #

        peak_x = 0.06182
        peak_y = self.func(peak_x)

        # intersection points
        self.knowledge.add_deriv(0, DataPoint( 0., 0.))
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, numbs.INFTY, '+')

        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, self.xl, peak_x, '+')
        self.knowledge.add_sign(1, peak_x, numbs.INFTY, '-')

        # concavity
        self.knowledge.add_sign(2, self.xl, peak_x, '-')
        self.knowledge.add_sign(2, peak_x, numbs.INFTY, '+')
    
    def func(self, x: float) -> float:
        m = 6.67
        g = 0.15
        b = 55.56
        c = 1.35
        d = 0.4
        e = 0.52
        y = m * g * d * np.sin(c * np.arctan(b * (1 - e) * x + e * np.arctan(b * x)))
        return self.__ymap(y)

    def __ymap(self, y:float) -> float:  # TODO: put in super class.
        return self.yl + (((y - self.__yl) / (self.__yu - self.__yl)) * (self.yu - self.yl)) 