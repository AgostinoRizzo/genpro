import random
import math
import numpy as np
import csv
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
        self.xl = -1.
        self.xu = 1.
        self.yl = 0.
        self.yu = 1.
        self.knowledge = DataKnowledge(self)
    
    def sample(self, size:int=100, noise:float=0.):
        y_noise = (self.yu - self.yl) * noise * 0.5
        for _ in range(size):
            x = random.uniform(self.xl, self.xu)
            y = self.func(x) + (0. if noise == 0. else random.gauss(sigma=y_noise))
            self.data.append(DataPoint(x, y))
    
    def load(self, filename:str):
        raise RuntimeError('Load from CSV file not supported.')
    
    def func(self, x:float) -> float:
        pass

    def inrange(self, dp:DataPoint, scale:float=1.) -> bool:
        x_padd = (self.xu - self.xl) * scale
        y_padd = (self.yu - self.yl) * scale
        
        return dp.x >= self.xl - x_padd and dp.x <= self.xu + x_padd and \
                dp.y >= self.yl - y_padd and dp.y <= self.yu + y_padd 
    
    def inrange_xy(self, x:float, y:float, scale:float=1.5) -> bool:
        return self.inrange(DataPoint(x, y), scale)
    
    def plot(self):
        plt.figure(2, figsize=[10,8])
        plt.clf()

        for dp in self.data:
            plt.plot(dp.x, dp.y, 'bo', markersize=1)
        
        self.knowledge.plot()

        x = np.linspace(self.xl, self.xu, 100)
        plt.plot(x, self.func(x), linestyle='dashed', color='black')
        plt.ylim(self.yl, self.yu)
        plt.grid()


class PolyDataset(Dataset):   
    def __init__(self) -> None:
        super().__init__()
        self.xl = -1.
        self.xu = 1.
        self.yl = -1.
        self.yu = 1.
     
    def func(self, x: float) -> float:
        return x**2

class TrigonDataset(Dataset):   
    def __init__(self) -> None:
        super().__init__()
        self.xl = -20.
        self.xu = 20.
        self.yl = -1.
        self.yu = 1.

        self.knowledge.add_deriv(0, DataPoint( 0., 0.))
        self.knowledge.add_deriv(0, DataPoint(  .5*math.pi,  1.))
        self.knowledge.add_deriv(0, DataPoint( -.5*math.pi, -1.))
        self.knowledge.add_deriv(0, DataPoint( 1.5*math.pi, -1.))
        self.knowledge.add_deriv(0, DataPoint(-1.5*math.pi,  1.))

        self.knowledge.add_deriv(1, DataPoint(  .5*math.pi,  0.))
        self.knowledge.add_deriv(1, DataPoint( -.5*math.pi,  0.))
        self.knowledge.add_deriv(1, DataPoint( 1.5*math.pi,  0.))
        self.knowledge.add_deriv(1, DataPoint(-1.5*math.pi,  0.))
     
    def func(self, x: float) -> float:
        return math.sin(x)


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
        #self.knowledge.add_sign(1, -peak_x+0.1, peak_x-0.1, '-')
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
    
    def __xmap(self, x:float, toorigin:bool=False) -> float:
        if toorigin: return self.__xl + (((x - self.xl) / (self.xu - self.xl)) * (self.__xu - self.__xl))
        return self.xl + (((x - self.__xl) / (self.__xu - self.__xl)) * (self.xu - self.xl))
    
    def __ymap(self, y:float) -> float:
        return self.yl + (((y - self.__yl) / (self.__yu - self.__yl)) * (self.yu - self.yl)) 


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