import cProfile
from pstats import SortKey

__profile = cProfile.Profile(builtins=False)

def enable():
    global __profile
    __profile.enable()

def disable():
    global __profile
    __profile.disable()

def print_stats():
    global __profile
    __profile.print_stats(SortKey.CUMULATIVE)