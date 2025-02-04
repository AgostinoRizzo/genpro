import numpy as np
from dataset import Dataset1d, DataPoint
import matplotlib.pyplot as plt


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
