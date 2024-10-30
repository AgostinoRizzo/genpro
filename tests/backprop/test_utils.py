import pytest
import numpy as np

import space
import dataset_misc3d
from backprop import utils
from symbols.parsing import parse_syntax_tree


@pytest.mark.parametrize("deriv,deriv_str", [
    ( (), '' ),
    ( (0,), 'd0' ),
    ( (1,2), 'd1d2' ),
    ( (0,12,20), 'd0d12d20' ),
    ( (0,0000), 'd0d0' ),
    ( (0,00,123,12), 'd0d0d123d12' ),
    ( (0,00,00,0), 'd0d0d0d0' ),
    ( (123,00,123,0), 'd123d0d123d0' ),
])
def test_parse_unparse_deriv(deriv:tuple[int], deriv_str:str):
    assert utils.deriv_to_string(deriv) == deriv_str
    assert utils.parse_deriv(deriv_str) == deriv
    assert utils.parse_deriv(deriv_str, parsefunc=True) == (deriv, '')


@pytest.mark.parametrize("deriv,deriv_str,func_str", [
    ( (), '   ', '' ),
    ( (), 'A', 'A' ),
    ( (), '   A ', 'A ' ),
    ( (0,), '  d  0  ', '' ),
    ( (1,2), 'd1 d2func', 'func' ),
    ( (0,12,20), '  d 0 d 1 2 d 2 0 ', '' ),
    ( (0,0000), 'd0d0  ', '' ),
    ( (0,00,123,12), 'd 0 d0d123 d12  ', '' ),
    ( (0,00,00,0), 'd0d 0 0  d0d0', '' ),
    ( (123,00,123,0), "d \n\t\t  123d0  d12  3d0", '' ),
    ( (0,12), "\n\n  d 0 d 1 2 a d 2 0 ", 'a d 2 0 ' ),
    ( (0,00,00), "d0 \nd 0 0  d0 ad0", 'ad0' ),
])
def test_parse_deriv(deriv:tuple[int], deriv_str:str, func_str:str):
    assert utils.parse_deriv(deriv_str) == deriv
    assert utils.parse_deriv(deriv_str, parsefunc=True) == (deriv, func_str)


@pytest.mark.parametrize("M,Msquare", [
    ( [[]], [[0.]] ),
    ( [[1]], [[1]] ),
    ( [[1],[1]], [[1,0], [1,0]] ),
    ( [[1,1]], [[1,1], [0,0]] ),
    ( [[1,1,1]], [[1,1,1], [0,0,0], [0,0,0]] ),
    ( [[1],[1],[1]], [[1,0,0], [1,0,0], [1,0,0]] ),

    # 3dim
    ( [[[]]], [[[0.]]] ),
    ( [[[1]]], [[[1]]] ),
    ( [[[1,2]],[[1,2]],[[1,2]]], [[[1,2,0],[0,0,0],[0,0,0]],[[1,2,0],[0,0,0],[0,0,0]],[[1,2,0],[0,0,0],[0,0,0]]] )
])
def test_squarify_mat(M, Msquare):
    M = np.array(M)
    Msquare = np.array(Msquare)
    Mactual = utils.squarify(M)
    if max(M.shape) == min(M.shape):
        assert id(M) == id(Mactual)
    assert Mactual.ndim == Msquare.ndim and max(Mactual.shape) == min(Mactual.shape)
    assert np.array_equal(Mactual, Msquare)


@pytest.mark.parametrize("expr,is_symm", [
    ('(sqrt((x0 * (x1 * x2))) * 0.001)', True),
    ('sqrt(((x1 * (x2 * x0)) * 0.01))', True),
    ('(x1 - (x1 + 1.53))', True),
    ('exp(square(x1))', False),
])
def test_symmetric(expr, is_symm):
    MESH_SIZE = 100

    S = dataset_misc3d.Resistance3()
    mesh = space.MeshSpace(S, S.knowledge, MESH_SIZE)

    stree = parse_syntax_tree(expr)
    y = stree(mesh.X)

    assert utils.is_symmetric(y, mesh.symm_Y_Ids) == is_symm
    n_symmetric = utils.count_symmetric(y, mesh.symm_Y_Ids)
    assert (is_symm and n_symmetric + 1 == mesh.symm_Y_Ids.shape[0]) or (not is_symm and n_symmetric + 1 < mesh.symm_Y_Ids.shape[0])