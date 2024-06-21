import numpy as np
import itertools


DERIV_IDENTIFIERS:dict[tuple,list] = {}


class SpaceSampler:
    def __init__(self, randstate:int=None):
        self.randgen = np.random.RandomState() if randstate is None else np.random.RandomState(randstate)
    def meshspace(self, xl, xu, size:int): pass
    def randspace(self, xl, xu, npoints:int): pass
    def get_meshsize(self, xl, xu, npoints:int) -> int: pass


class UnidimSpaceSampler(SpaceSampler):
    def __init__(self, randstate:int=None):
        super().__init__(randstate)
    
    def meshspace(self, xl, xu, size:int):
        return np.linspace(xl, xu, min(1,size) if xl == xu else size)
    
    def randspace(self, xl, xu, npoints:int):
        return self.randgen.uniform(xl, xu, min(1,npoints) if xl == xu else npoints)
    
    def get_meshsize(self, xl, xu, npoints:int) -> int:
        assert npoints >= 0
        return npoints


class MultidimSpaceSampler(SpaceSampler):
    def __init__(self, randstate:int=None):
        super().__init__(randstate)
    
    def meshspace(self, xl, xu, size:int):
        xsize = xl.size
        assert xsize == xu.size and size >= 0
        
        if size == 0:
            return np.empty((0, xsize))
        
        npoints = 1
        sidesize = np.full(xsize, size, dtype=int)
        for i in range(xsize):
            if xl[i] == xu[i]: sidesize[i] = 1
            else: npoints *= size
        
        grid = np.mgrid[*[slice(xl[i],xu[i],sidesize[i]+0j) for i in range(xsize)]]
        X = np.empty((npoints, xsize))
        for i in range(xsize):
            X[:,i] = grid[i].reshape(npoints)
        
        return X
    
    def randspace(self, xl, xu, npoints:int):
        xsize = xl.size
        assert xsize == xu.size and npoints >= 0

        X = np.empty((npoints, xsize))
        for i in range(xsize):
            X[:,i] = self.randgen.uniform(xl[i], xu[i], npoints) 
        
        return X
    
    def get_meshsize(self, xl, xu, npoints:int) -> int:
        xsize = xl.size
        assert xsize == xu.size and xsize > 0 and npoints > 0
        return int(npoints ** (1/xsize))


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