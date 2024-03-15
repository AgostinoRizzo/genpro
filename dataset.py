import random
import math
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DataPoint:
    def __init__(self, x:float, y:float) -> None:
        self.x = x
        self.y = y
    
    def distance(self, other) -> float:
        return math.sqrt((other.x-self.x)**2 + (other.y-self.y)**2)


class DataKnowledge:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.derivs = {}
        self.sign = {}
    
    def add_deriv(self, d:int, xy:DataPoint):
        if d not in self.derivs.keys():
            self.derivs[d] = []
        self.derivs[d].append(xy)
    
    def add_sign(self, d:int, l:float, u:float, sign:str='+'):
        if d not in self.sign.keys():
            self.sign[d] = []
        self.sign[d].append((l,u,sign))
    
    def plot(self):
        if 0 in self.derivs.keys():
            for xy in self.derivs[0]:
                plt.plot(xy.x, xy.y, 'rx', markersize=10)
        
        if 0 in self.sign.keys():
            for (l,u,s) in self.sign[0]:
                plt.axvspan(l, u, alpha=0.05, color='g' if s == '+' else 'r')
            

class Dataset:
    def __init__(self) -> None:
        self.data = []
        self.test = []
        self.xl = -1.
        self.xu = 1.
        self.yl = 0.
        self.yu = 1.
        self.knowledge = DataKnowledge(self)
        self.sst = 0.
    
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
    
    def _on_data_changed(self):
        y_mean = 0.
        for dp in self.data: y_mean += dp.y
        y_mean /= len(self.data)

        self.sst = 0.
        for dp in self.data: self.sst += (dp.y - y_mean) ** 2
    
    def func(self, x:float) -> float:
        pass

    def inrange(self, dp:DataPoint, scale:float=1.) -> bool:
        x_padd = (self.xu - self.xl) * scale
        y_padd = (self.yu - self.yl) * scale
        
        return dp.x >= self.xl - x_padd and dp.x <= self.xu + x_padd and \
                dp.y >= self.yl - y_padd and dp.y <= self.yu + y_padd 
    
    def inrange_xy(self, x:float, y:float, scale:float=1.5) -> bool:
        return self.inrange(DataPoint(x, y), scale)
    
    def plot(self, plot_data: bool=True):
        plt.figure(2, figsize=[10,8])
        plt.clf()

        if plot_data:
            data_labeld = False
            for dp in self.data:
                if data_labeld: plt.plot(dp.x, dp.y, 'bo', markersize=2)
                else:
                    plt.plot(dp.x, dp.y, 'bo', markersize=2, label='Training data')
                    data_labeld = True
        
        self.knowledge.plot()

        x = np.linspace(self.xl, self.xu, 100)
        plt.plot(x, self.func(x), linestyle='dashed', linewidth=2, color='black', label='Reference model')
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
        self.xl = 0.8
        self.xu = 10
        self.yl = -1
        self.yu = 9
     
    def func(self, x: float) -> float:
        return np.sin(x) + 1  #x / (x**2)#(x+2) / (x**2 + x + 1)


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
        self.knowledge.add_deriv(0, DataPoint( 0., 0.))
        self.knowledge.add_deriv(0, DataPoint(-peak_x, self.func(-peak_x)))
        self.knowledge.add_deriv(0, DataPoint( peak_x, self.func( peak_x)))
        self.knowledge.add_deriv(0, DataPoint(self.xl, self.func(self.xl)))
        self.knowledge.add_deriv(0, DataPoint(self.xu, self.func(self.xu)))

        # known (first) derivatives
        self.knowledge.add_deriv(1, DataPoint(-peak_x,  0.))
        self.knowledge.add_deriv(1, DataPoint( peak_x,  0.))

        #
        # positivity/negativity contraints
        #
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, -0.00001, '+')
        self.knowledge.add_sign(0, 0.00001, self.xu, '-')
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, self.xl, -0.01, '+')
        #self.knowledge.add_sign(1, -peak_x+0.1, peak_x-0.1, '-')
        self.knowledge.add_sign(1, -0.01, self.xu, '+')

        # concavity/convexity
        self.knowledge.add_sign(2, self.xl, -0.01, '+')
        self.knowledge.add_sign(2, 0.01, self.xu, '-')

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
        peak_x = 0.208
        
        # intersection points
        self.knowledge.add_deriv(0, DataPoint( 0., 0.))
        self.knowledge.add_deriv(0, DataPoint(-peak_x, self.func(-peak_x)))
        self.knowledge.add_deriv(0, DataPoint( peak_x, self.func( peak_x)))
        self.knowledge.add_deriv(0, DataPoint(self.xl, self.func(self.xl)))
        self.knowledge.add_deriv(0, DataPoint(self.xu, self.func(self.xu)))

        # known (first) derivatives
        self.knowledge.add_deriv(1, DataPoint(-peak_x,  0.))
        self.knowledge.add_deriv(1, DataPoint( peak_x,  0.))

        #
        # positivity/negativity contraints
        #
        
        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, -0.001, '+')
        self.knowledge.add_sign(0, 0.001, self.xu, '-')
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign(1, self.xl, -0.81, '+')
        self.knowledge.add_sign(1, -peak_x+0.1, peak_x-0.1, '-')
        self.knowledge.add_sign(1, 0.81, self.xu, '+')

        # concavity/convexity
        self.knowledge.add_sign(2, self.xl, -0.81, '+')
        self.knowledge.add_sign(2, 0.81, self.xu, '-')

    def func(self, x: float) -> float:
        x = self.__xmap(x, toorigin=True)
        y = -self.i*self.c1*x / (x**2 + self.c2)**3
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
    
    def func(self, x: float) -> float:
        m = 6.67 #407.75
        g = 0.15 #9.81
        b = 55.56
        c = 1.35
        d = 0.4
        e = 0.52
        return m * g * d * np.sin(c * np.arctan(b * (1 - e) * x + e * np.arctan(b * x)))

