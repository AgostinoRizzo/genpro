import pytest

import dataset
import dataset_misc1d
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.misc import UnknownSyntaxTree
from backprop import lpbackprop
from backprop import jump_backprop
from backprop import constraints
from backprop import utils


def test_jump_backprop():
    # setup dataset.
    S = dataset_misc1d.MagmanDatasetScaled()
    S.load('../data/magman.csv')
    S.split(train_size=0.7, randstate=0)

    S_train = dataset.NumpyDataset(S)
    S_test  = dataset.NumpyDataset(S, test=True)

    # setup test model.
    unknown_stree_a = UnknownSyntaxTree('A')
    unknown_stree_b = UnknownSyntaxTree('B')
    stree     = BinaryOperatorSyntaxTree('/', unknown_stree_a, unknown_stree_b)
    stree_pr  = stree.diff().simplify()
    stree_pr2 = stree_pr.diff().simplify()
    stree_map = {(): stree, (0,): stree_pr, (0,0): stree_pr2}

    # apply lp backprop + jump backprop.
    best_unkn_models = {}
    best_eval = None

    def onsynth_callback(synth_unkn_models:dict):
        nonlocal best_unkn_models
        nonlocal best_eval
        
        hist, __best_unkn_models, __best_eval = jump_backprop.jump_backprop(stree_map, synth_unkn_models, S_train, S_test, max_rounds=2)
        if best_eval is None or __best_eval.better_than(best_eval):
            best_unkn_models = __best_unkn_models
            best_eval = __best_eval

    lpbackprop.lpbackprop(S.knowledge, stree, onsynth_callback)

    # check outcome.
    assert best_eval is not None

    assert best_eval.training['mse' ] == pytest.approx(0.27012293133946363)
    assert best_eval.training['rmse'] == pytest.approx(0.5197335195457992)
    assert best_eval.training['r2'  ] == pytest.approx(0.7420818803797324)

    assert best_eval.testing['mse' ] == pytest.approx(0.31569039417756356)
    assert best_eval.testing['rmse'] == pytest.approx(0.5618633233959692)
    assert best_eval.testing['r2'  ] == pytest.approx(0.7032210142992348)

    assert best_eval.knowledge['mse0'] == pytest.approx(8.164718157143927e-05)
    assert best_eval.knowledge['mse1'] == pytest.approx(0.02984748149830924)
    assert best_eval.knowledge['mse2'] == pytest.approx(10.337145725667774)