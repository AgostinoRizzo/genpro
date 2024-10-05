import pytest
import numpy as np

import dataset
import dataset_misc1d
from symbols.const import ConstantSyntaxTree
from symbols.var import VariableSyntaxTree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.unaop import UnaryOperatorSyntaxTree
from symbols.misc import UnknownSyntaxTree
from gp import corrector


@pytest.fixture
def data():
    S = dataset_misc1d.MagmanDatasetScaled()
    S.sample(size=100, noise=0.0, mesh=False)
    S_train = dataset.NumpyDataset(S)
    return S, S_train

backprop_node = ConstantSyntaxTree(2.0)

@pytest.mark.parametrize("stree,noroot,k_image,k_deriv,partial,none", [
    (
        BinaryOperatorSyntaxTree('/',
            BinaryOperatorSyntaxTree('*',
                ConstantSyntaxTree(-0.05),
                VariableSyntaxTree(),
            ),
            backprop_node
        ),
        True,
        [1.] * 100,
        [-1.] * 42 + [np.nan] * 16 + [1.] * 42,
        True,
        False
    ),

    (
        BinaryOperatorSyntaxTree('/',
            BinaryOperatorSyntaxTree('*',
                ConstantSyntaxTree(-0.05),
                VariableSyntaxTree(),
            ),
            UnaryOperatorSyntaxTree('cube', backprop_node)
        ),
        True,
        [1.] * 100,
        [-1.] * 42 + [np.nan] * 16 + [1.] * 42,
        True,
        False
    ),

    (
        BinaryOperatorSyntaxTree('/',
            BinaryOperatorSyntaxTree('*',
                ConstantSyntaxTree(-0.05),
                VariableSyntaxTree(),
            ),
            UnaryOperatorSyntaxTree('cube',
                UnaryOperatorSyntaxTree('square', backprop_node))
        ),
        True,
        [np.nan] * 100,  # natural softening.
        [np.nan] * 100,  # unknown softening.
        True,
        False
    ),
])
def test_corrector(data, stree, noroot, k_image, k_deriv, partial, none):
    S, S_train = data
    max_depth = 5

    corr = corrector.Corrector(S_train, S.knowledge, max_depth)
    new_stree, new_node, C_pulled, y_pulled = corr.correct(stree, backprop_node)

    assert new_stree.get_max_depth() <= max_depth

    assert C_pulled.noroot == noroot
    assert set(C_pulled.origin_pconstrs.keys()) == set([(), (0,)])

    assert np.array_equal(C_pulled.origin_pconstrs[()], np.array(k_image), equal_nan=True)
    assert np.array_equal(C_pulled.origin_pconstrs[(0,)], np.array(k_deriv), equal_nan=True)

    assert C_pulled.partial == partial
    assert C_pulled.none == none