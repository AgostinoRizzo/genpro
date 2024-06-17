import pytest
import numpy as np
import math
from backprop import models


@pytest.mark.parametrize("deg", [0, 1, 2, 5, 10])
@pytest.mark.parametrize("nvars", [1, 2, 3, 5])
def test_poly_coeffsize(deg, nvars):
    poly = models.ModelFactory.create_poly(deg, nvars)
    if nvars == 1: assert type(poly) is models.Poly1d
    else: assert type(poly) is models.Polynd
    
    coeffs = poly.get_coeffs()
    actual_coeffs_size = math.comb(nvars+deg, deg)
    assert coeffs.ndim == 1
    assert coeffs.size == actual_coeffs_size
    assert np.count_nonzero(coeffs) == 0

    if nvars > 1:
        assert poly.C.shape == (deg+1,)*nvars
        assert len(poly.cidx) == nvars
        for i_var in range(nvars):
            assert poly.cidx[i_var].size == actual_coeffs_size
        for i_coeff in range(actual_coeffs_size):
            coeffidx_sum = 0
            for i_var in range(nvars):
                coeffidx_sum += poly.cidx[i_var][i_coeff]
            assert coeffidx_sum <= deg


@pytest.mark.parametrize("poly_str,c,x,y,y_pr,y_pr2", [
    ('P(x) = 2', [2.],
        [0., -1., 5.4, 1000.],
        [2., 2., 2., 2.],
        [0., 0., 0., 0.,],
        [0., 0., 0., 0.,]),
    ('P(x) = 2x^2 + x', [2., 1., 0.],
        [0., -1., 5.4, 1000.],
        [0., 1., 63.72, 2001000.],
        [1., -3., 22.6, 4001.],
        [4., 4., 4., 4.]),
    ('P(x) = x^5', [1., 0., 0., 0., 0., 0.],
        [0., -1., 5.4, 1000.],
        [0., -1., 4591.65024, 1.e15],
        [0., 5., 4251.528, 5.e12],
        [0., -20., 3149.28, 2.e10])
    ])
def test_poly1d(poly_str, c, x, y, y_pr, y_pr2):
    c = np.array(c)
    x = np.array(x)
    y = np.array(y)
    deg = c.size - 1
    poly = models.ModelFactory.create_poly(deg)
    poly.set_coeffs(c)
    assert np.allclose(poly(x), y)
    assert np.allclose(poly.get_deriv((0,))(x), y_pr)
    assert np.allclose(poly.get_deriv((0,0))(x), y_pr2)

    for c_idx, _ in enumerate(c):
        poly.set_coeff(c_idx, 0.)
    assert np.count_nonzero( poly.get_coeffs() ) == 0
    with pytest.raises(IndexError): poly.set_coeff(-c.size-1, 2)
    with pytest.raises(IndexError): poly.set_coeff(c.size, 4)
    with pytest.raises(IndexError): poly.set_coeff(c.size + 10, 0)
    with pytest.raises(AssertionError): poly.set_coeff((1, 2), 4)
    with pytest.raises(AssertionError): poly.set_coeff(0.5, 4)


@pytest.mark.parametrize("poly_str,deg,c_map,x,y,y_x0,y_x1,y_x0x0,y_x0x1,y_x1x0,y_x1x1", [
    ('P(x0,x1) = 2', 2, {(0,0): 2.},
        [[0., 0.], [-1., 2.], [5.4, -3.4], [1000., 1.2]],
        [2] * 4,
        [0.] * 4, [0.] * 4,
        [0.] * 4, [0.] * 4, [0.] * 4, [0.] * 4),
    ('P(x0,x1) = 2x0^5 + 0.5xy^3 - 1.5x^2y^2 + 2x - 5y + 1', 5, {(5,0): 2., (1,3): 0.5, (2,2): -1.5, (1,0): 2., (0,1): -5., (0,0): 1.},
        [[0., 0.], [-1., 2.], [5.4, -3.4], [1000., 1.2]],
        [1., -23., 8600.34528, 1.9999999978e15],
        [2., 28, 8298.132, 9.9999999957e12], [-5., -17., 386.068, -3597845.],
        [0., -52, 6263.88, 3.9999999996e10], [0., 18., 127.5, -7197.84], [0., 18., 127.5, -7197.84], [0., -9., -142.56, -2996400.])
    ])
def test_poly2d(poly_str, deg, c_map, x, y, y_x0, y_x1, y_x0x0, y_x0x1, y_x1x0, y_x1x1):
    x = np.array(x).T
    y = np.array(y)
    poly = models.ModelFactory.create_poly(deg=deg, nvars=2)

    C_size = deg + 1
    C = np.zeros((C_size, C_size))
    for idx, c in c_map.items():
        C[idx] = c
    poly.C = C

    C_poly = poly.get_coeffs()
    poly.set_coeffs(C_poly)
    assert C_poly.size == math.comb(2+deg, deg)
    assert np.array_equal(poly.C, C)

    assert np.allclose(poly(x), y)
    assert np.allclose(poly.get_deriv((0,)) (x), y_x0)
    assert np.allclose(poly.get_deriv((1,)) (x), y_x1)
    assert np.allclose(poly.get_deriv((0,0))(x), y_x0x0)
    assert np.allclose(poly.get_deriv((0,1))(x), y_x0x1)
    assert np.allclose(poly.get_deriv((1,0))(x), y_x1x0)
    assert np.allclose(poly.get_deriv((1,1))(x), y_x1x1)

    for c_idx, _ in c_map.items():
        poly.set_coeff(c_idx, 0.)
    assert np.count_nonzero( poly.get_coeffs() ) == 0
    with pytest.raises(AssertionError): poly.set_coeff(5, 4)
    with pytest.raises(AssertionError): poly.set_coeff((deg, 1), 4)
    with pytest.raises(AssertionError): poly.set_coeff((1, 2, 3), 1)


@pytest.mark.parametrize("poly_str,deg,c_map,x,y,y_x0x1,y_x1x0,y_x2x2", [
    ('P(x0,x1,x2) = 2', 4, {(0,0,0): 2.},
        [[0., 0., 0.], [-1., 2., 1.], [5.4, -3.4, -1.2], [10., 1.2, -0.5]],
        [2.] * 4, [0.] * 4, [0.] * 4, [0.] * 4),
    ('P(x0,x1,x2) = 2', 0, {(0,0,0): 2.},
        [[0., 0., 0.], [-1., 2., 1.], [5.4, -3.4, -1.2], [10., 1.2, -0.5]],
        [2.] * 4, [0.] * 4, [0.] * 4, [0.] * 4),
    ('P(x0,x1,x2) = x0^5x1^4x2^3', 12, {(5,4,3): 1.},
        [[0., 0., 0.], [-1., 2., 1.], [5.4, -3.4, -1.2], [10., 1.2, -0.5]],
        [0.0, -16.0, -1060298.6426128466, -25920.0],
        [0.0, 160.0, 1155009.414610944, -43199.99999999999],
        [0.0, 160.0, 1155009.414610944, -43199.99999999999],
        [0.0, -96.0, -4417911.010886862, -622080.0])
    ])
def test_poly3d(poly_str, deg, c_map, x, y, y_x0x1, y_x1x0, y_x2x2):
    x = np.array(x).T
    y = np.array(y)
    poly = models.ModelFactory.create_poly(deg=deg, nvars=3)

    C_size = deg + 1
    C = np.zeros((C_size,)*3)
    for idx, c in c_map.items():
        C[idx] = c
    poly.C = C

    C_poly = poly.get_coeffs()
    poly.set_coeffs(C_poly)
    assert C_poly.size == math.comb(3+deg, deg)
    assert np.array_equal(poly.C, C)
    
    assert np.allclose(poly(x), y)
    assert np.allclose(poly.get_deriv((0,1))(x), y_x0x1)
    assert np.allclose(poly.get_deriv((1,0))(x), y_x1x0)
    assert np.allclose(poly.get_deriv((2,2))(x), y_x2x2)

    for c_idx, _ in c_map.items():
        poly.set_coeff(c_idx, 0.)
    assert np.count_nonzero( poly.get_coeffs() ) == 0
    with pytest.raises(AssertionError): poly.set_coeff(5, 4)
    with pytest.raises(AssertionError): poly.set_coeff((2, 4), 4)
    with pytest.raises(AssertionError): poly.set_coeff((deg,1), 1)


@pytest.mark.parametrize("poly_str,deg,c_map,x,y,y_x0x4, y_x3x2, y_x2x1", [
    ('P(x0,x1,x2,x3,x4) = 2', 8, {(0,)*5: 2.},
        [[0., 0., 0., 0., 0.], [-1., 2., 1., 5.2, 4.], [5.4, -3.4, -1.2, -0.001, 10.01], [10., 1.2, -0.5, -4.5, -1]],
        [2.] * 4, [0.] * 4, [0.] * 4, [0.] * 4),
    ('P(x0,x1,x2,x3,x4) = 2', 0, {(0,)*5: 2.},
        [[0., 0., 0., 0., 0.], [-1., 2., 1., 5.2, 4.], [5.4, -3.4, -1.2, -0.001, 10.01], [10., 1.2, -0.5, -4.5, -1]],
        [2.] * 4, [0.] * 4, [0.] * 4, [0.] * 4),
    ('P(x0,x1,x2,x3,x4) = x0^2x1^2x2^2x3^2x4^2', 10, {(2,)*5: 1.},
        [[0., 0., 0., 0., 0.], [-1., 2., 1., 5.2, 4.], [5.4, -3.4, -1.2, -0.001, 10.01], [10., 1.2, -0.5, -4.5, -1]],
        [0.0, 1730.5600000000002, 0.04863803274570239, 729.0],
        [0.0, -1730.5600000000002, 0.0035992180223999998, -291.59999999999997],
        [0.0, 1331.2, 162.126775819008, 1296.0],
        [0.0, 3461.1200000000003, 0.04768434582912, -4860.0])
    ])
def test_poly5d(poly_str, deg, c_map, x, y, y_x0x4, y_x3x2, y_x2x1):
    x = np.array(x).T
    y = np.array(y)
    poly = models.ModelFactory.create_poly(deg=deg, nvars=5)

    C_size = deg + 1
    C = np.zeros((C_size,)*5)
    for idx, c in c_map.items():
        C[idx] = c
    poly.C = C

    C_poly = poly.get_coeffs()
    poly.set_coeffs(C_poly)
    assert C_poly.size == math.comb(5+deg, deg)
    assert np.array_equal(poly.C, C)
    
    assert np.allclose(poly(x), y)
    assert np.allclose(poly.get_deriv((0,4))(x), y_x0x4)
    assert np.allclose(poly.get_deriv((3,2))(x), y_x3x2)
    assert np.allclose(poly.get_deriv((2,1))(x), y_x2x1)

    for c_idx, _ in c_map.items():
        poly.set_coeff(c_idx, 0.)
    assert np.count_nonzero( poly.get_coeffs() ) == 0
    with pytest.raises(AssertionError): poly.set_coeff(5, 4)
    with pytest.raises(AssertionError): poly.set_coeff((2, 4), 4)
    with pytest.raises(AssertionError): poly.set_coeff((deg,1,0,0,0), 1)


""" # it just takes time.
('P(x0,x1,x2,x3,x4) = x0^5x1^5x2^5x3^5x4^5', 25, {(5,)*5: 1.},
    [[0., 0., 0., 0., 0.], [-1., 2., 1., 5.2, 4.], [5.4, -3.4, -1.2, -0.001, 10.01], [10., 1.2, -0.5, -4.5, -1]],
    [0.0, -124585257.20576002, -0.0005217230229163385, -14348906.999999996],
    [0.0, 778657857.5360001, -0.0002412971393959459, 35872267.49999999],
    [0.0, -598967582.7200001, -10.869229644090389, -159432299.99999997],
    [0.0, -1557315715.0720003, -0.0031968322482618786, 597871125.0])
"""


@pytest.mark.parametrize("deg,coeffs,out", [
    (4, [], '0'),
    (2, [0, 0, 0], '0'),
    (0, [2], '2.00'),
    (1, [1,2], '1.0*x + 2.0'),
    (2, [-8.21, 4, 2], '-8.21*x**2 + 4.0*x + 2.0'),
    ])
def test_to_sympy_poly1d(deg, coeffs, out):
    poly = models.ModelFactory.create_poly(deg)
    for cidx, c in enumerate(coeffs):
        poly.set_coeff(cidx, c)

    actual = set(str(poly.to_sympy(dps=3)).split(' + '))
    expected = set(out.split(' + '))
    assert actual == expected


@pytest.mark.parametrize("deg,nvars,coeffs,out", [
    (2, 2, {(0,0): 2}, '2.00'),
    (5, 2, {(0,0): 2, (1,0): 1}, '2.0 + 1.0*x0'),
    (5, 2, {(0,0): 2, (1,0): 1, (0,1): 3, (2,1): 4, (2,2): 5}, '2.0 + 1.0*x0 + 3.0*x1 + 4.0*x0**2*x1 + 5.0*x0**2*x1**2'),
    (1, 3, {(0,0,0): 2, (1,0,0): 1, (0,1,0): 1, (0,0,1): 1}, '2.0 + 1.0*x0 + 1.0*x1 + 1.0*x2'),
    ])
def test_to_sympy_polynd(deg, nvars, coeffs, out):
    poly = models.ModelFactory.create_poly(deg, nvars)
    for cidx, c in coeffs.items():
        poly.set_coeff(cidx, c)
    
    actual = set(str(poly.to_sympy(dps=3)).split(' + '))
    expected = set(out.split(' + '))
    assert actual == expected