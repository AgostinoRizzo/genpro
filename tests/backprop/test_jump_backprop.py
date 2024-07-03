import pytest

import dataset
import dataset_misc1d
from backprop import backprop
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
    unknown_stree_a = backprop.UnknownSyntaxTree('A')
    unknown_stree_b = backprop.UnknownSyntaxTree('B')
    stree     = backprop.BinaryOperatorSyntaxTree('/', unknown_stree_a, unknown_stree_b)
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

    assert best_eval.training['mse' ] == pytest.approx(0.27002394406794616)
    assert best_eval.training['rmse'] == pytest.approx(0.5196382819499985)
    assert best_eval.training['r2'  ] == pytest.approx(0.7421763951653138)

    assert best_eval.testing['mse' ] == pytest.approx(0.3155965372282962)
    assert best_eval.testing['rmse'] == pytest.approx(0.5617797942506443)
    assert best_eval.testing['r2'  ] == pytest.approx(0.703309248755266)

    assert best_eval.knowledge['mse0'] == pytest.approx(7.912192457223037e-05)
    assert best_eval.knowledge['mse1'] == pytest.approx(0.029880375624880235)
    assert best_eval.knowledge['mse2'] == pytest.approx(10.318817878914349)