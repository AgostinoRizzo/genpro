import numpy as np
import itertools
from scipy.spatial import KDTree
import math


DERIV_IDENTIFIERS:dict[tuple,list] = {}


class SpaceSampler:
    def meshspace(self, xl, xu, npoints:int): pass
    def randspace(self, xl, xu, npoints:int): pass
    def get_meshsize(self, xl, xu, npoints:int) -> int: pass


class UnidimSpaceSampler(SpaceSampler):
    def meshspace(self, xl, xu, npoints:int):
        return np.linspace(xl, xu, min(1,npoints) if xl == xu else npoints)
    
    def randspace(self, xl, xu, npoints:int):
        return np.random.uniform(xl, xu, min(1,npoints) if xl == xu else npoints)
    
    def get_meshsize(self, xl, xu, npoints:int) -> int:
        assert npoints >= 0
        return npoints


class MultidimSpaceSampler(SpaceSampler):
    def meshspace(self, xl, xu, npoints:int):
        xsize = xl.size
        assert xsize == xu.size and xsize > 0 and npoints >= 0
        
        if npoints == 0:
            return np.empty((0, xsize))
        
        npoints_side = self.get_meshsize(xl, xu, npoints)
        
        npoints = 1
        sidesize = np.full(xsize, npoints_side, dtype=int)
        for i in range(xsize):
            if xl[i] == xu[i]: sidesize[i] = 1
            else: npoints *= npoints_side
        
        grid = np.mgrid[*[slice(xl[i],xu[i],sidesize[i]+0j) for i in range(xsize)]]
        X = np.empty((npoints, xsize))
        for i in range(xsize):
            X[:,i] = grid[i].reshape(npoints)
        
        return X
    
    def randspace(self, xl, xu, npoints:int):
        xsize = xl.size
        assert xsize == xu.size and npoints >= 0

        if npoints == 0:
            return np.empty((0, xsize))
        
        npoints_side = self.get_meshsize(xl, xu, npoints)
        
        npoints = 1
        for i in range(xsize):
            if xl[i] != xu[i]: npoints *= npoints_side

        X = np.empty((npoints, xsize))
        for i in range(xsize):
            X[:,i] = np.random.uniform(xl[i], xu[i], npoints) 
        
        return X
    
    def get_meshsize(self, xl, xu, npoints:int) -> int:
        xsize = xl.size
        assert xsize == xu.size and xsize > 0 and npoints >= 0
        return int(npoints**(1/xsize))


class MeshSpace:
    def __init__(self, mesh, X_data):
        if len(mesh.shape) == 1: mesh = np.asmatrix(mesh).T
        self.index = KDTree(mesh)
        self.X_data_map = np.empty(X_data.shape[0], dtype=np.int64)
        for i in range(self.X_data_map.size):
            _, i_mesh = self.index.query(X_data[i])
            self.X_data_map[i] = i_mesh
    
    def nearest_meshpoint(self, X_data_idx:int):
        return self.X_data_map[X_data_idx]


def get_all_derivs(nvars:int=1, max_derivdeg:int=2) -> list:
    global DERIV_IDENTIFIERS
    if (nvars,max_derivdeg) in DERIV_IDENTIFIERS:
        return DERIV_IDENTIFIERS[(nvars,max_derivdeg)]

    derivs = []
    for derivdeg in range(max_derivdeg+1):
        for deriv in itertools.product(range(nvars), repeat=derivdeg):
            derivs.append(deriv)
    
    DERIV_IDENTIFIERS[(nvars,max_derivdeg)] = derivs
    return derivs


def get_nested_hypercubes(points:list) -> list[tuple[np.array]]:
    """
    It is assumed all points in the input list are sorted and unique.
    A minimum of 2 points is required.

    Returns: list of hypercubes hc (as tuple) where
        hc[0] := l:np.array and hc[1] := u:np.array
    """

    assert len(points) >= 2

    hcs = []

    nvars = points[0].size
    axes_coords = [set() for _ in range(nvars)]
    
    for p in points:
        for i in range(nvars):
            axes_coords[i].add(p[i])
    
    for i in range(nvars):
        axes_coords[i] = sorted(axes_coords[i])
    
    axes_coords_idxs_lower = [range(len(axis_coords)-1) for axis_coords in axes_coords]
    for l_idx in itertools.product(*axes_coords_idxs_lower):

        l = tuple(axes_coords[var_idx][coord_idx  ] for var_idx, coord_idx in enumerate(l_idx))
        u = tuple(axes_coords[var_idx][coord_idx+1] for var_idx, coord_idx in enumerate(l_idx))
        
        hcs.append((np.array(l), np.array(u)))
    
    return hcs
