import random
import numpy as np


def setstate(state=None):
    random.seed(state)
    np.random.seed(state)
