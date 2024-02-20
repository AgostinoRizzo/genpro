import taylor_spline
import dataset
import numpy as np
import random
import matplotlib.pyplot as plt

class Smoother:
    def __init__(self, S: dataset.Dataset, splide_degree: int=2) -> None:
        self.S = S
        self.spline_degree = splide_degree
        self.X = None
        self.Y = None
        self.Y_LB = None
        self.Y_UB = None
    
    def smoth(self):
        self.S.plot()
        self.__smoth(exp_cov=0.3)

        data_tmp = self.S.data
        self.S.data = []
        for i in range(len(self.X)):
            sigma = 0 #(UB[i] - LB[i]) / 2
            dp = dataset.DataPoint(self.X[i], random.gauss(self.Y[i], sigma))
            self.S.data.append(dp)
            
        self.__smoth(exp_cov=0.1, plot=True)
        self.S.data = data_tmp

    def __smoth(self, exp_cov=0.3, plot: bool=False):

        self.X = np.linspace(self.S.xl, self.S.xu, 500)
        self.Y = []
        self.Y_LB = []
        self.Y_UB = []
        for x in self.X:
            tspline_est = taylor_spline.TaylorSplineEstimator()
            tspline = tspline_est.fit(self.S, self.spline_degree, silent=True, x0=x, exp_cov=exp_cov)
            lb, ub = tspline_est.get_bounds(self.S, tspline)

            self.Y.append(tspline.y(x))
            self.Y_LB.append(lb)
            self.Y_UB.append(ub)

        if plot:
            plt.plot(self.X, self.Y,    linestyle='solid',  linewidth=2, color='purple', label='Model')
            plt.plot(self.X, self.Y_LB, linestyle='dashed', linewidth=1, color='gray',   label='Lower Bound')
            plt.plot(self.X, self.Y_UB, linestyle='dashed', linewidth=1, color='gray',   label='Upper Bound')