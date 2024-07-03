import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


"""
Abstraction interface (according to the Bridge pattern).
"""
class Plotter:
    def __init__(self, impl):
        self.impl = impl


class DatasetPlotter(Plotter):
    def __init__(self, dataset, impl):
        super().__init__(impl)
        self.dataset = dataset
    
    def plot(self,
             plot_data:bool=True,
             plot_knowldege:bool=False,
             width:int=10,
             height:int=8,
             plotref:bool=True,
             model=None,
             zoomout:float=1.,
             savename:str=None,
             show:bool=True):
        """
        acts as a template method w.r.t the implementation.
        """
        
        ax = self.impl.init(width, height)
        self.impl.set_model(self.dataset.func)

        if plot_data:
            def plot_data_points(data:list, marker:str, color:str, label:str):
                data_labeld = False
                for dp in data:
                    if data_labeld: self.impl.plot_datapoint(dp, marker, color, markersize=2)
                    else:
                        self.impl.plot_datapoint(dp, marker, color, markersize=2, label=label)
                        data_labeld = True
            plot_data_points(self.dataset.data, 'o', 'b', 'Training data')
            plot_data_points(self.dataset.test, 'o', 'm', 'Test data')
        
        if plot_knowldege:
            self.dataset.knowledge.plot()

        xstep_zoomout = (self.dataset.xu - self.dataset.xl) * (zoomout - 1) * .5
        ystep_zoomout = (self.dataset.yu - self.dataset.yl) * (zoomout - 1) * .5
        xl = self.dataset.xl - xstep_zoomout
        xu = self.dataset.xu + xstep_zoomout
        yl = self.dataset.yl - ystep_zoomout
        yu = self.dataset.yu + ystep_zoomout

        if plotref:
            self.impl.plot_model(self.dataset.func, xl, xu, zoomout=zoomout,
                linestyle='dashed', linewidth=2, color='black', label='Reference model')
        
        if model is not None:
            self.impl.plot_model(model, xl, xu, zoomout=zoomout, linewidth=2, color='green', label='Model')
        
        self.impl.flush_content()

        self.impl.set_limits(xl, xu, yl, yu)
        self.impl.set_grid()
        self.impl.set_tick()

        ax.legend(loc='upper right', fontsize=14)
        self.impl.set_labels()

        if savename is not None:
            plt.savefig(savename, bbox_inches='tight')
        
        if show:
            plt.show()


class NumpyDatasetPlotter(Plotter):
    def __init__(self, dataset, impl):
        super().__init__(impl)
        self.dataset = dataset
    
    def plot(self,
             width:int=10,
             height:int=8,
             model=None):
        """
        acts as a template method w.r.t the implementation.
        """
        
        ax = self.impl.init(width,height)
        
        self.impl.plot_scatter(self.dataset.X, self.dataset.y, 'o', 'b', markersize=2)
        if model is not None:
            self.impl.plot_model(model, self.dataset.xl, self.dataset.xu, 1.0, linewidth=2, color='green', label='Model')
        self.impl.set_limits(self.dataset.xl, self.dataset.xu, self.dataset.yl, self.dataset.yu)
        self.impl.set_grid()
        self.impl.set_tick()
        plt.show()


"""
Implementation interface (according to the Bridge pattern).
"""
class PlotterImpl:
    def __init__(self, dataset):
        self.dataset = dataset
        self.ax:Axes = None
        self.model = None
    
    def init(self, width, height) -> Axes: pass
    def set_model(self, model): self.model = model
    def plot_datapoint(self, dp, marker, color, markersize, label=None): pass
    def plot_scatter(self, x, y, marker, color, markersize, label=None): pass
    def plot_model(self, model, xl, xu, zoomout, linewidth, color, label, linestyle='solid'): pass
    def flush_content(self): pass
    def set_grid(self): pass
    def set_tick(self): pass
    def set_limits(self, xl, xu, yl, yu): pass
    def set_labels(self): pass


class Dataset1dPlotterImpl(PlotterImpl):
    SAMPLE_SIZE = 500

    def __init__(self, dataset):
        super().__init__(dataset)
    
    def init(self, width, height) -> Axes:
        fig = plt.figure(2, figsize=[width,height])
        self.ax = fig.add_subplot()
        return self.ax
    
    def plot_datapoint(self, dp, marker, color, markersize, label=None):
        self.ax.scatter(dp.x, dp.y, marker=marker, c=color, s=markersize**2, label=label)
    
    def plot_scatter(self, x, y, marker, color, markersize, label=None):
        self.ax.scatter(x, y, marker=marker, c=color, s=markersize**2, label=label)
    
    def plot_model(self, model, xl, xu, zoomout, linewidth, color, label, linestyle='solid'):
        sample_size = int(Dataset1dPlotterImpl.SAMPLE_SIZE * zoomout)
        x = np.linspace(xl, xu, sample_size)
        self.ax.plot(x, model(x), linestyle=linestyle, linewidth=linewidth, color=color, label=label)
    
    def set_grid(self):
        self.ax.set_axisbelow(True)
        self.ax.grid(linestyle='dashed', linewidth=0.7)

    def set_tick(self):
        self.ax.tick_params(direction='in', length=5, top=True, right=True)

    def set_limits(self, xl, xu, yl, yu):
        self.ax.set_xlim(xl, xu)
        self.ax.set_ylim(yl, yu)
    
    def set_labels(self):
        self.ax.set_xlabel(self.dataset.get_xlabel())
        self.ax.set_ylabel(self.dataset.get_ylabal())


class Dataset2dPlotterImpl(PlotterImpl):
    SAMPLE_SIZE = 50

    def __init__(self, dataset):
        super().__init__(dataset)
        self.top_scatter = []
    
    def init(self, width, height) -> Axes:
        fig = plt.figure(2, figsize=[width,height])
        self.ax = fig.add_subplot(projection='3d', computed_zorder=False)
        self.ax.view_init(azim=225)
        return self.ax
    
    def plot_datapoint(self, dp, marker, color, markersize, label=None):
        y_model = self.model(np.array([[dp.x[0], dp.x[1]]]))
        if dp.y >= y_model:
            self.top_scatter.append( (dp.x[0], dp.x[1], dp.y, marker, color, markersize**2, label) )
        else:
            self.ax.scatter(dp.x[0], dp.x[1], dp.y, marker=marker, c=color, s=markersize**2, label=label)            
    
    def plot_scatter(self, x, y, marker, color, markersize, label=None):
        y_model = self.model(dp.x)
        top_idx = y >= y_model
        if top_idx.any():
            self.scatter.append( (x[top_idx:,0], x[top_idx:,1], y[top_idx], marker, color, markersize**2, label) )
        else:
            self.ax.scatter(x[~top_idx:,0], x[~top_idx:,1], y[~top_idx], marker=marker, c=color, s=markersize**2, label=label)
            
    
    def plot_model(self, model, xl, xu, zoomout, linewidth, color, label, linestyle='solid'):
        sample_size = int(Dataset1dPlotterImpl.SAMPLE_SIZE * zoomout)
        x = np.linspace(xl[0], xu[0], sample_size)
        y = np.linspace(xl[1], xu[1], sample_size)
        x, y = np.meshgrid(x, y)
        X = np.column_stack( (np.ravel(x), np.ravel(y)) )
        z = np.array(model(X))
        z = z.reshape(x.shape)
        #self.ax.plot_surface(x, y, z, color=color, alpha=0.3, linewidth=0, antialiased=True)
        self.ax.plot_surface(x, y, z, color=color, edgecolor=color, lw=0.05, alpha=0.3)
        #self.ax.contour(x, y, z, zdir='z', offset= 0, cmap='coolwarm')
        #self.ax.contour(x, y, z, zdir='x', offset=20, cmap='coolwarm')
        #self.ax.contour(x, y, z, zdir='y', offset=20, cmap='coolwarm')
        self.model = model
    
    def flush_content(self):
        for x1, x2, y, marker, color, markersize, label in self.top_scatter:
            self.ax.scatter(x1, x2, y, marker=marker, c=color, s=markersize, label=label)

    def set_grid(self):
        for axis in [self.ax.xaxis, self.ax.yaxis, self.ax.zaxis]:
            axis._axinfo["grid"]['linestyle'] = 'dashed'
            axis._axinfo["grid"]['linewidth'] = 0.5
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    def set_tick(self):
        for axis in ['x', 'y', 'z']:
            self.ax.tick_params(axis=axis, pad=-1)

    def set_limits(self, xl, xu, yl, yu):
        self.ax.set_xlim(xl[0], xu[0])
        self.ax.set_ylim(xl[1], xu[1])
        self.ax.set_zlim(yl, yu)
        
    
    def set_labels(self):
        self.ax.set_xlabel(self.dataset.get_xlabel(xidx=0))
        self.ax.set_ylabel(self.dataset.get_xlabel(xidx=1))
        self.ax.set_zlabel(self.dataset.get_ylabal())

