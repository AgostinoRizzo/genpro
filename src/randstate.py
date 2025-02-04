import random
import numpy as np


def setstate(state=None):
    random.seed(state)
    np.random.seed(state)

def getstate():
    return {'random': random.getstate(), 'numpy.random': np.random.get_state()}
