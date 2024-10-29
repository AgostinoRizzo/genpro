import pytest
import numpy as np

import dataset
import dataset_misc1d
import dataset_misc2d
import space
from symbols.const import ConstantSyntaxTree
from symbols.var import VariableSyntaxTree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.unaop import UnaryOperatorSyntaxTree
from symbols.misc import UnknownSyntaxTree
from gp import corrector, creator
from backprop.utils import is_symmetric


@pytest.fixture
def data_magman():
    S = dataset_misc1d.MagmanDatasetScaled()
    S.sample(size=100, noise=0.0, mesh=False)
    S_train = dataset.NumpyDataset(S)
    return S, S_train

@pytest.fixture
def data_resistance2():
    S = dataset_misc2d.Resistance2()
    S.xl = np.zeros(2)
    S.sample(size=100, noise=0.0, mesh=False)
    S_train = dataset.NumpyDataset(S)
    return S, S_train

backprop_node = ConstantSyntaxTree(2.0)

@pytest.mark.parametrize("data,stree,noroot,symm,k_image,k_deriv,partial,none", [
    (
        'data_magman',
        BinaryOperatorSyntaxTree('/',
            BinaryOperatorSyntaxTree('*',
                ConstantSyntaxTree(-0.05),
                VariableSyntaxTree(),
            ),
            backprop_node
        ),
        True,
        None,
        [1.] * 100,
        {(0,): [-1.] * 42 + [np.nan] * 16 + [1.] * 42},
        True,
        False
    ),

    (
        'data_magman',
        BinaryOperatorSyntaxTree('/',
            BinaryOperatorSyntaxTree('*',
                ConstantSyntaxTree(-0.05),
                VariableSyntaxTree(),
            ),
            UnaryOperatorSyntaxTree('cube', backprop_node)
        ),
        True,
        None,
        [1.] * 100,
        {(0,): [-1.] * 42 + [np.nan] * 16 + [1.] * 42},
        True,
        False
    ),

    (
        'data_magman',
        BinaryOperatorSyntaxTree('/',
            BinaryOperatorSyntaxTree('*',
                ConstantSyntaxTree(-0.05),
                VariableSyntaxTree(),
            ),
            UnaryOperatorSyntaxTree('cube',
                UnaryOperatorSyntaxTree('square', backprop_node))
        ),
        True,
        None,
        [np.nan] * 100,  # natural (lossless) softening.
        {(0,): [np.nan] * 100},  # lossy softening.
        True,
        False
    ),

    (
        'data_resistance2',
        BinaryOperatorSyntaxTree('/',
            BinaryOperatorSyntaxTree('*',
                VariableSyntaxTree(0),
                VariableSyntaxTree(1),
            ),
            backprop_node
        ),
        True,
        None,  # symm
        [np.nan] * 10 + ([np.nan] + [1.] * 9) * 9,
        {(0,): [np.nan] * 100, (1,): [np.nan] * 100},
        True,
        False
    ),

    (
        'data_resistance2',
        BinaryOperatorSyntaxTree('/',
            BinaryOperatorSyntaxTree('*',
                VariableSyntaxTree(0),
                VariableSyntaxTree(1),
            ),
            BinaryOperatorSyntaxTree('+',
                backprop_node,
                VariableSyntaxTree(1),
            )
        ),
        True,
        None,  # symm
        [np.nan] * 100,
        {(0,): [np.nan] * 100, (1,): [np.nan] * 100},
        True,
        False
    ),

    (
        'data_resistance2',
        BinaryOperatorSyntaxTree('/',
            BinaryOperatorSyntaxTree('*',
                VariableSyntaxTree(0),
                VariableSyntaxTree(1),
            ),
            BinaryOperatorSyntaxTree('+',
                backprop_node,
                BinaryOperatorSyntaxTree('+',
                    VariableSyntaxTree(0),
                    VariableSyntaxTree(1),
                )
            )
        ),
        True,
        None,  # symm
        [np.nan] * 100,
        {(0,): [np.nan] * 100, (1,): [np.nan] * 100},
        True,
        False
    ),

    (
        'data_resistance2',
        BinaryOperatorSyntaxTree('/',
            BinaryOperatorSyntaxTree('*',
                VariableSyntaxTree(0),
                VariableSyntaxTree(1),
            ),
            BinaryOperatorSyntaxTree('+',
                backprop_node,
                BinaryOperatorSyntaxTree('*',
                    ConstantSyntaxTree(-1.0),
                    BinaryOperatorSyntaxTree('+',
                        VariableSyntaxTree(0),
                        VariableSyntaxTree(1),
                    )
                )
            )
        ),
        True,
        None,  # symm
        [np.nan] * 10 + ([np.nan] + [1.] * 9) * 9,
        {(0,): [np.nan] * 100, (1,): [np.nan] * 100},
        True,
        False
    ),
])
def test_corrector(data, stree, noroot, symm, k_image, k_deriv, partial, none, request):
    S, S_train = request.getfixturevalue(data)
    max_depth     = 5
    max_length    = 20
    MESH_SIZE     = 100
    mesh          = space.MeshSpace(S_train, S.knowledge, MESH_SIZE)
    libsize       = 2000
    lib_maxdepth  = 3
    lib_maxlength = 10
    
    y_iqr = S_train.get_y_iqr()
    solutionCreator = creator.RandomSolutionCreator(nvars=S.nvars, y_iqr=y_iqr)

    corr = corrector.Corrector(S_train, S.knowledge, max_depth, max_length, mesh, libsize, lib_maxdepth, lib_maxlength, solutionCreator)
    new_stree, new_node, C_pulled, y_pulled = corr.correct(stree, backprop_node)

    assert new_stree.get_max_depth() <= max_depth

    assert C_pulled.noroot == noroot
    assert (C_pulled.symm is None and symm is None) or C_pulled.symm[0] == symm
    assert C_pulled.symm is None or C_pulled.symm[0] is None or C_pulled.symm[0] == is_symmetric(new_stree[mesh.X, ()], C_pulled.symm[1])
    assert set(C_pulled.origin_pconstrs.keys()) == set([()] + list(k_deriv.keys()))

    assert np.array_equal(C_pulled.origin_pconstrs[()], np.array(k_image), equal_nan=True)
    for d in k_deriv.keys():
        assert np.array_equal(C_pulled.origin_pconstrs[d], np.array(k_deriv[d]), equal_nan=True)

    assert C_pulled.partial == partial
    assert C_pulled.none == none
