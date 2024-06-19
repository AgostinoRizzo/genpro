import pytest
import numpy as np

import space


@pytest.mark.parametrize("xl,xu,size", [
    (1, 5, 5),
    (-5, 1, 10),
    (0,0,1),
    (0,0,0),
    (0,0,4),
    (5,3,4),
    (-1,-2,2)
    ])
def test_1d_space_sampler(xl, xu, size):
    spsampler = space.UnidimSpaceSampler()
    for points, israndom in [
            (spsampler.meshspace(xl, xu, size), False),
            (spsampler.randspace(xl, xu, size), True),
        ]:
        assert points.shape == (size,)
        for p in points:
            assert (p >= xl and p <= xu and xl <= xu) or \
                   (p <= xl and p >= xu and xl >= xu)
        if not israndom:
            assert size == 0 or (xl in points and xu in points)
            for i in range(size-1):
                assert (points[i] <= points[i+1] and xl <= xu) or \
                       (points[i] >= points[i+1] and xl >= xu)


@pytest.mark.parametrize("xl,xu,size", [
    ([1, 1], [5, 5], 5),
    ([1, 2, -3], [2, 3, 4], 2),
    ([-1], [1], 3),
    ([1, 2, 3], [1, 2, 3], 5),
    ([1, 2, 3], [1, 2, 3], 0),
    ([5, 5], [1, 1], 5),
])
def test_nd_space_sampler(xl, xu, size):
    xl = np.array(xl)
    xu = np.array(xu)
    xsize = xl.size
    assert xsize == xu.size
    spsampler = space.MultidimSpaceSampler()
    
    for points, israndom in [
            (spsampler.meshspace(xl, xu, size), False),
            (spsampler.randspace(xl, xu, size**xsize), True),
        ]:
        assert points.shape == (size**xsize, xsize)
        for i in range(xsize):
            assert ((points[:,i] >= xl[i]).all() and (points[:,i] <= xu[i]).all() and (xl <= xu).all()) or \
                   ((points[:,i] <= xl[i]).all() and (points[:,i] >= xu[i]).all() and (xl >= xu).all())
        if not israndom:
            assert size == 0 or ((xl == points[0,:]).all() and (xu == points[-1,:]).all())


@pytest.mark.parametrize("nvars,max_derivdeg,expected", [
    (1, 0, [()]), (1, 1, [(),(0,)]), (1, 2, [(),(0,),(0,0)]), (1, 3, [(),(0,),(0,0),(0,0,0)]),
    (2, 0, [()]), (2, 1, [(),(0,),(1,)]), (2, 2, [(),(0,),(1,),(0,0),(0,1),(1,0),(1,1)]),
    (3, 0, [()]), (3, 1, [(),(0,),(1,),(2,)]), (3, 2, [(),(0,),(1,),(2,),(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)])
])
def test_get_all_derivs(nvars, max_derivdeg, expected):
    expected = expected
    actual = space.get_all_derivs(nvars, max_derivdeg)
    assert set(actual) == set(expected)
