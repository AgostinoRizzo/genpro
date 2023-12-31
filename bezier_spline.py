import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.special import comb

from dataset import Dataset

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
    
    def minus(self):
        p = Point()
        p.x = -self.x
        p.y = -self.y
        return p
    
    def add(self, other):
        p = Point()
        p.x = self.x + other.x
        p.y = self.y + other.y
        return p

    def dist(self, other) -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)
    
    def norm(self):
        l = self.length()
        p = Point()
        p.x = self.x / l
        p.y = self.y / l
        return p
    
    def scale(self, s:float):
        p = Point()
        p.x = self.x * s
        p.y = self.y * s
        return p



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
    
    def get_chromo_coordmap(self) -> list:
        cmap = []
        for node_idx in range(4):
            for coord in ['x', 'y']:
                key = str(node_idx+1) + '_' + coord
                
                if key not in self.binded_nodes.keys() and key not in self.fixed_nodes.keys():
                    cmap.append(coord)
        return cmap
    
    def get_chromo(self) -> list:
        chromo = []
        for node_idx in range(4):
            for coord in ['x', 'y']:
                key = str(node_idx+1) + '_' + coord
                
                if key not in self.binded_nodes.keys() and key not in self.fixed_nodes.keys():
                    chromo.append(self.nodes[node_idx].get_coord(coord))
        return chromo
    
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
    
    def getnode(self, node_idx:int) -> Point:
        return self.nodes[node_idx-1]

    def getx(self, node_idx:int) -> float:
        return self.nodes[node_idx-1].x
    
    def gety(self, node_idx:int) -> float:
        return self.nodes[node_idx-1].y
    
    def randomize(self, S:Dataset):
        last_x = S.xl  # TODO: manage with binded_nodes and fixed_nodes
        for _ in range(100):
            for node_idx in range(4):
                for coord in ['x', 'y']:
                    key = str(node_idx+1) + '_' + coord
                    
                    if key not in self.binded_nodes.keys() and key not in self.fixed_nodes.keys():
                        self.nodes[node_idx].set_coord(coord, random.uniform(last_x, S.xu) if coord == 'x' else random.uniform(S.yl, S.yu))
                        if coord == 'x': last_x = self.nodes[node_idx].get_coord('x')
            if self.isvalid(S):
                return
        raise RuntimeError('Invalid spline. Randomize limit exceeded.')
    
    def isvalid(self, S:Dataset) -> bool:
        inrange = S.inrange_xy(self.getx(1), self.gety(1)) and \
                    S.inrange_xy(self.getx(2), self.gety(2)) and \
                    S.inrange_xy(self.getx(3), self.gety(3)) and \
                    S.inrange_xy(self.getx(4), self.gety(4))
        #if not inrange: return False

        if self.getx(1) > self.getx(4): return False
        if self.getx(2) > self.getx(3): return False
        if self.getx(2) < self.getx(1) or self.getx(3) > self.getx(4): return False

        return True

    def fitness(self, S:Dataset) -> float:
        if not self.isvalid(S):
            raise RuntimeError('Invalid spline.')
        
        sse = 0.
        for dp in S.data:
            if dp.x >= self.getx(1) and dp.x <= self.getx(4):
                sse += (self.Y(self.X_inv(dp.x)) - dp.y) ** 2

        fit = -sse

        return fit
    
    def plot(self):
        c = [(k.x, k.y) for k in self.nodes]
        X = np.array(c)

        tt = np.linspace(0, 1, 200)
        xx = P(tt, X)

        plt.plot(xx[:,0], xx[:,1])

        # plot tangent lines
        plt.plot([self.getx(1), self.getx(2)], [self.gety(1), self.gety(2)], 'ro', linestyle="--")
        plt.plot([self.getx(3), self.getx(4)], [self.gety(3), self.gety(4)], 'ro', linestyle="--")
    

class BezierCurveConnector:
    def __init__(self, xl:float, xu:float) -> None:
        self.curves = []
        self.xl = xl
        self.xu = xu
    
    def connect(self, curve:BezierCurve):
        self.curves.append(curve)
    
    def get_chromo_length(self) -> int:
        n_curves = len(self.curves)
        
        if n_curves == 0: return 0
        n_nodes = 4 + (n_curves-1)*2
        chromo_length = n_nodes*2 + n_curves-1  # + (n_curves-1) --> time for the second node
        chromo_length -= 2  # x-limits condition

        return chromo_length
    
    def set_chromo(self, chromo:list):
        n_curves = len(self.curves)
        if n_curves == 0: return

        chromo = chromo.tolist()
        chromo_len = len(chromo)
        chromo = [self.xl] + chromo[:chromo_len-1] + [self.xu] + [chromo[-1]]
        self.curves[0].set_chromo(chromo[:8])

        chromo = chromo[4:]
        for i in range(1, n_curves):
            sub_chromo = chromo[:9]

            pm1 = Point()
            p1  = Point()
            p2  = None
            p3  = Point()
            p4  = Point()

            pm1.x = sub_chromo[0]
            pm1.y = sub_chromo[1]

            p1.x = sub_chromo[2]
            p1.y = sub_chromo[3]
            
            p2_t = sub_chromo[4]

            p3.x = sub_chromo[5]
            p3.y = sub_chromo[6]

            p4.x = sub_chromo[7]
            p4.y = sub_chromo[8]

            p2_dir = p1.add(pm1.minus()).norm()
            p2 = p1.add(p2_dir.scale(p2_t))

            sub_chromo = [
                p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y
            ]

            self.curves[i].set_chromo(sub_chromo)
            chromo = chromo[5:]
    
    def isvalid(self, S:Dataset) -> bool:
        for c in self.curves:
            if not c.isvalid(S): return False
        return True

    def fitness(self, S:Dataset) -> float:
        fitval = 0.
        for c in self.curves:
            fitval += c.fitness(S)
        return fitval
    
    def plot(self):
        for c in self.curves:
            c.plot()
