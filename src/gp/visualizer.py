import numpy as np
from sklearn.decomposition import PCA
from dataset import Dataset, Dataset1d, DataPoint
from gp.gp import GP
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class SpaceVisualizerDataset(Dataset1d):
    def __init__(self, optimal_y:list[float]):
        super().__init__(xl=-5., xu=5.)
        assert len(optimal_y) == 2

        self.yl = -5.
        self.yu =  5.

        # known positivity/negativity
        self.knowledge.add_sign(0, self.xl, self.xu, '+')
    
        # monotonically increasing/decreasing
        self.knowledge.add_sign((0,), self.xl, self.xu, '+')

        self.data.append(DataPoint(-2., optimal_y[0]))
        self.data.append(DataPoint( 3., optimal_y[1]))
    
    def func(self, x: float) -> float:
        return np.exp(0.3*x)


class SpaceVisualizer:
    def __init__(self, data):
        self.data = data
        self.spaces = {}
    
    def track(self, stree:list, group:str):
        if group not in self.spaces:
            self.spaces[group] = []

        y = stree(self.data.X)
        assert y.size == 2

        self.spaces[group].append((y[0], y[1]))
    
    def plot(self, savename=None):

        for group, space in self.spaces.items():
            xs = [x for x, _ in space]
            ys = [y for _, y in space]
            
            min_xs, max_xs = -5, 5 #np.nanquantile(xs, 0.05), np.nanquantile(xs, 0.95)
            min_ys, max_ys = -5, 5 #np.nanquantile(ys, 0.05), np.nanquantile(ys, 0.95)
            
            n_visualized = 0
            for i in range(len(xs)):
                x, y = xs[i], ys[i]
                if np.isfinite(x) and np.isfinite(y) and x >= min_xs and x <= max_xs and y >= min_ys and y <= max_ys:
                    n_visualized += 1
            
            plt.figure(figsize=(5,5))
            
            R_x  = np.array([0, 0, max_xs])
            R_y1 = np.array([max_ys, 0, max_ys])
            R_y2 = np.array([max_ys]*3)
            plt.fill_between(R_x, R_y1, R_y2, color='green', alpha=0.1)

            plt.scatter(xs, ys, c='k', s=0.5)
            plt.scatter(self.data.y[0], self.data.y[1], c='r', s=80, marker='*')

            plt.xlim((min_xs,max_xs))
            plt.ylim((min_ys,max_ys))

            plt.gca().set_axisbelow(True)
            plt.gca().grid(linestyle='dashed', linewidth=0.7)
            plt.gca().tick_params(direction='in', length=5, top=True, right=True)

            #plt.title(f"{group} ({n_visualized}/{len(xs)})")

            plt.xlabel('y1')
            plt.ylabel('y2')

            if savename is not None:
                plt.savefig(savename + str(group) + '.pdf', bbox_inches='tight')

            plt.show()


class EvolutionVisualizer:
    def plot(self, gp_status:GP):
        raise NotImplementedError()

    def get_figure(self) -> Figure:
        raise NotImplementedError()
    
    def on_finalize(self):
        plt.close()


class SolutionPlotVisualizer(EvolutionVisualizer):
    def __init__(self, data:Dataset, sols_to_plot:int=10):
        super().__init__()
        self.data = data
        self.sols_to_plot = sols_to_plot
    
    def plot(self, gp_status:GP):
        data_plotter = self.data.get_plotter()
        is_plotter_init = data_plotter.impl.is_init()
        if is_plotter_init:
            data_plotter.impl.ax.clear()
        data_plotter.plot(show=False, init=not is_plotter_init)

        pop_to_plot = gp_status.population[:min(self.sols_to_plot, gp_status.popsize)]
        
        for i, p in enumerate(pop_to_plot):
            # TODO: add support for linearly scaled solutions
            p.clear_output()
            alpha_val = 1 - (i / len(pop_to_plot))
            data_plotter.impl.plot_model(model=p, xl=self.data.xl, xu=self.data.xu,
                                         zoomout=1, linewidth=1, color=(0.2, 0.2, 0.2, alpha_val), label=None)
            p.clear_output()
    
    def get_figure(self) -> Figure:
        return self.data.get_plotter().impl.fig


class SemanticSpaceVisualizer(EvolutionVisualizer):
    def __init__(self, data:Dataset, width:int, height:int):
        super().__init__()
        self.data = data
        self.width = width
        self.height = height
        self.fig = None
        self.ax = None
    
    def plot(self, gp_status:GP):
        if self.ax is None:
            self.fig = plt.figure(2, figsize=[self.width, self.height])
            self.ax = self.fig.add_subplot()
        else:
            self.ax.clear()
        
        Y = [self.data.y]
        for p in gp_status.population:
            y = p(self.data.X)
            if np.isfinite(y).all():
                Y.append(y)

        pca = PCA(n_components=2)
        Y_new = pca.fit_transform(Y)

        target = [Y_new[0][0], Y_new[0][1]]
        
        y_1 = []
        y_2 = []
        delta = 5
        for y in Y_new[1:]:
            if y[0] >= target[0] - delta and y[0] <= target[0] + delta and \
               y[1] >= target[1] - delta and y[1] <= target[1] + delta:
                y_1.append(y[0])
                y_2.append(y[1])

        self.ax.set_xlim(target[0] - delta, target[0] + delta)
        self.ax.set_ylim(target[1] - delta, target[1] + delta)
        self.ax.scatter(y_1, y_2, c='black')
        self.ax.scatter(target[0], target[1], c='red')
    
    def get_figure(self) -> Figure:
        return self.fig
