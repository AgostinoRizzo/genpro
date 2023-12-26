import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

class Point:
    def __init__(self) -> None:
        self.x = 0.
        self.y = 0.
    
    def get_coord(self, coord:str) -> float:
        if coord == 'x': return self.x
        return self.y
    
    def set_coord(self, coord:str, value:float):
        if coord == 'x': self.x = value
        else: self.y = value


#
# auxiliary function for plotting
#
def B(i, N, t):
    val = comb(N,i) * t**i * (1.-t)**(N-i)
    return val

def P(t, X):
    '''
     xx = P(t, X)
     
     Evaluates a Bezier curve for the points in X.
     
     Inputs:
      X is a list (or array) or 2D coords
      t is a number (or list of numbers) in [0,1] where you want to
        evaluate the Bezier curve
      
     Output:
      xx is the set of 2D points along the Bezier curve
    '''
    X = np.array(X)
    N,d = np.shape(X)   # Number of points, Dimension of points
    N = N - 1
    xx = np.zeros((len(t), d))
    
    for i in range(N+1):
        xx += np.outer(B(i, N, t), X[i])
    
    return xx

class BezierCurve:
    def __init__(self) -> None:
        self.nodes = [Point() for _ in range(4)]
        self.fixed_nodes = {}
        self.binded_nodes = {}
        
    def X(self, t:float) -> float:
        k1_x = self.nodes[0].x
        k2_x = self.nodes[1].x
        k3_x = self.nodes[2].x
        k4_x = self.nodes[3].x
        return ((1-t)**3)*k1_x + 3*t*((1-t)**2)*k2_x + 3*(t**2)*(1-t)*k3_x + (t**3)*k4_x

    def Y(self, t:float) -> float:
        k1_y = self.nodes[0].y
        k2_y = self.nodes[1].y
        k3_y = self.nodes[2].y
        k4_y = self.nodes[3].y
        return ((1-t)**3)*k1_y + 3*t*((1-t)**2)*k2_y + 3*(t**2)*(1-t)*k3_y + (t**3)*k4_y
    
    def X_inv(self, x:float) -> float:
        # by construction, Bx is monotonically increasing!
        l = 0.
        u = 1.
        intv_eps = 0.0001

        while (u - l) > intv_eps:
            c = (u + l) / 2
            Bx_c = self.X(c)
            
            if x == Bx_c: return c
            if x < Bx_c: u = c
            else: l = c

        return (u + l) / 2
    
    def fixnode(self, node_idx:int, coord:str, value:float):
        key = str(node_idx) + '_' + coord
        self.fixed_nodes[key] = value
        if key in self.binded_nodes.keys(): del self.binded_nodes[key]
        if coord == 'x': self.nodes[node_idx-1].x = value
        else: self.nodes[node_idx-1].y = value
    
    def bindnode(self, node_idx:int, pivot_node_idx:int, coord:str):
        key = str(node_idx) + '_' + coord
        self.binded_nodes[key] = pivot_node_idx
        if key in self.fixed_nodes.keys(): del self.fixed_nodes[key]
        
        if coord == 'x': self.nodes[node_idx-1].x = self.nodes[pivot_node_idx-1].x
        else: self.nodes[node_idx-1].y = self.nodes[pivot_node_idx-1].y
    
    def get_chromo_length(self) -> int:
        return 8 - len(self.fixed_nodes.keys()) - len(self.binded_nodes.keys())
    
    def set_chromo(self, chromo:list):
        chromo_i = 0
        for node_idx in range(4):
            for coord in ['x', 'y']:
                key = str(node_idx+1) + '_' + coord

                if key in self.binded_nodes.keys():
                    self.nodes[node_idx].set_coord(coord, self.nodes[self.binded_nodes[key]-1].get_coord(coord))

                elif key not in self.fixed_nodes.keys():
                    self.nodes[node_idx].set_coord(coord, chromo[chromo_i])
                    chromo_i += 1
    
    def get(self, node_idx:int, coord:str) -> float:
        return self.nodes[node_idx-1].get_coord(coord)

    def getx(self, node_idx:int) -> float:
        return self.nodes[node_idx-1].x
    
    def gety(self, node_idx:int) -> float:
        return self.nodes[node_idx-1].y
    
    def plot(self):
        c = [(k.x, k.y) for k in self.nodes]
        X = np.array(c)

        tt = np.linspace(0, 1, 200)
        xx = P(tt, X)

        plt.plot(xx[:,0], xx[:,1])

        # plot tangent lines
        plt.plot([self.getx(1), self.getx(2)], [self.gety(1), self.gety(2)], 'ro', linestyle="--")
        plt.plot([self.getx(3), self.getx(4)], [self.gety(3), self.gety(4)], 'ro', linestyle="--")