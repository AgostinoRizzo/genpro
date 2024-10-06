import pytest
import numpy as np

import space


@pytest.mark.parametrize("xl,xu,size,expected_size", [
    (1, 5, 5, 5),
    (-5, 1, 10, 10),
    (0,0,1,1),
    (0,0,5,1),
    (0,0,0,0),
    (-1,1,4,4),
    ( 1,1,4,1),
    ( 1,1,0,0),
    (5,3,4,4),
    (-1,-2,2,2)
    ])
def test_1d_space_sampler(xl, xu, size, expected_size):
    spsampler = space.UnidimSpaceSampler()
    for points, israndom in [
            (spsampler.meshspace(xl, xu, size), False),
            (spsampler.randspace(xl, xu, size), True),
        ]:
        assert points.shape == (expected_size,)
        for p in points:
            assert (p >= xl and p <= xu and xl <= xu) or \
                   (p <= xl and p >= xu and xl >= xu)
        if not israndom:
            assert expected_size == 0 or (xl in points and xu in points)
            for i in range(expected_size-1):
                assert (points[i] <= points[i+1] and xl <= xu) or \
                       (points[i] >= points[i+1] and xl >= xu)


@pytest.mark.parametrize("xl,xu,npoints,expected_npoints", [
    ([1, 1], [5, 5], 25, 25),
    ([1, 2, -3], [2, 3, 4], 9, 8),
    ([-1], [1], 3, 3),
    ([1, 2, 3], [2, 3, 4], 128, 125),
    ([1, 2, 3], [1, 2, 3], 25, 1),
    ([1, 2, 3], [1, 2, 4], 5, 1),
    ([1, 2, 3], [2, 3, 4], 0, 0),
    ([1, 2, 3], [1, 2, 3], 0, 0),
    ([1, 2, 3], [1, 2, 4], 0, 0),
    ([5, 5], [1, 1], 25, 25),
])
def test_nd_space_sampler(xl, xu, npoints, expected_npoints):
    xl = np.array(xl, dtype=float)
    xu = np.array(xu, dtype=float)
    xsize = xl.size
    assert xsize == xu.size
    spsampler = space.MultidimSpaceSampler()
    
    for points, israndom in [
            (spsampler.meshspace(xl, xu, npoints), False),
            (spsampler.randspace(xl, xu, npoints), True),
        ]:
        assert points.shape == (expected_npoints, xsize)
        for i in range(xsize):
            assert ((points[:,i] >= xl[i]).all() and (points[:,i] <= xu[i]).all() and (xl <= xu).all()) or \
                   ((points[:,i] <= xl[i]).all() and (points[:,i] >= xu[i]).all() and (xl >= xu).all())
        if not israndom:
            if expected_npoints == 1:
                for ix in range(xsize):
                    assert xl[ix] == points[0][ix] or xu[ix] == points[0][ix]
            elif expected_npoints > 1:
                assert ((xl == points[0,:]).all() and (xu == points[-1,:]).all())


@pytest.mark.parametrize("nvars,max_derivdeg,expected", [
    (1, 0, [()]), (1, 1, [(),(0,)]), (1, 2, [(),(0,),(0,0)]), (1, 3, [(),(0,),(0,0),(0,0,0)]),
    (2, 0, [()]), (2, 1, [(),(0,),(1,)]), (2, 2, [(),(0,),(1,),(0,0),(0,1),(1,0),(1,1)]),
    (3, 0, [()]), (3, 1, [(),(0,),(1,),(2,)]), (3, 2, [(),(0,),(1,),(2,),(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)])
])
def test_get_all_derivs(nvars, max_derivdeg, expected):
    space.DERIV_IDENTIFIERS.clear()
    actual = space.get_all_derivs(nvars, max_derivdeg)
    assert set(actual) == set(expected)


@pytest.mark.parametrize("points,expected_hcs", [
    ([],None),  # empty
    
    # 1-dim
    ([0],None),
    ([0, 1], [(0,1)]),
    ([0, 0.5, 1], [(0, 0.5), (0.5, 1)]),
    ([0, 0.2, 0.8, 1], [(0, 0.2), (0.2, 0.8), (0.8, 1)]),
    
    # 2-dim
    ([(0,0)],None),
    ([(0,0), (1,1)], [((0,0), (1,1))]),
    
    ([(0,0), (0.5, 0.5), (1,1)],
        [((0,0),(0.5,0.5)), ((0.5,0),(1,0.5)), ((0,0.5),(0.5,1)), ((0.5,0.5),(1,1))]),
    
    ([(0,0), (0.2, 0.2), (0.8, 0.8), (1,1)],
        [((0,0),(0.2,0.2)), ((0.2,0),(0.8,0.2)), ((0.8,0),(1,0.2)), ((0,0.2),(0.2,0.8)),
         ((0,0.8),(0.2,1)), ((0.2,0.2),(0.8,0.8)), ((0.8,0.2),(1,0.8)), ((0.2,0.8),(0.8,1)), ((0.8,0.8),(1,1))]),
    
    # 3-dim
    ([(0,0,0)],None),
    ([(0,0,0), (1,1,1)], [((0,0,0), (1,1,1))]),

    ([(0,0,0), (0.5, 0.5, 0.5), (1,1,1)],
        [((0,0,0),(0.5,0.5,0.5)), ((0.5,0,0),(1,0.5,0.5)), ((0,0.5,0),(0.5,1,0.5)), ((0.5,0.5,0),(1,1,0.5)),
         ((0,0,0.5),(0.5,0.5,1)), ((0.5,0,0.5),(1,0.5,1)), ((0,0.5,0.5),(0.5,1,1)), ((0.5,0.5,0.5),(1,1,1))])
])
def test_get_nested_hypercubes(points:list, expected_hcs:list[tuple]):
    
    points = [np.array(p, ndmin=1) for p in points]

    if len(points) < 2:
        with pytest.raises(AssertionError):
            space.get_nested_hypercubes(points)
    else:
        
        expected_hcs = set(expected_hcs)
        actual_hcs = space.get_nested_hypercubes(points)
        actual_hcs = [(tuple(hc[0]), tuple(hc[1])) for hc in actual_hcs]  # hcs[0] := l, hcs[1] := u
        for i in range(len(actual_hcs)):
            if len(actual_hcs[i][0]) == 1:  # actual_hcs[i][1] is of the same size!
                actual_hcs[i] = (actual_hcs[i][0][0], actual_hcs[i][1][0])
        actual_hcs = set(actual_hcs)

        assert actual_hcs == expected_hcs
