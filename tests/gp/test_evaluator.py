import pytest

import dataset
import dataset_misc2d
from gp.evaluator import R2Evaluator, KnowledgeEvaluator, LayeredEvaluator
from symbols.parsing import parse_syntax_tree


@pytest.fixture
def data_resistance2():
    S = dataset_misc2d.Resistance2()
    S.sample(size=100, noise=0.0, mesh=False)
    S_train = dataset.NumpyDataset(S)
    return S, S_train


@pytest.mark.parametrize("data,expr", [
    ('data_resistance2', '((x0*x1)/(x0+x1))'),
    ('data_resistance2','((x1 / (x1 + x0)) * x0)')
    #('data_resistance2', 'sqrt(square(((x0 * x1) / (x1 + x0))))')
])
def test_evaluator(data, expr, request):
    S, S_train = request.getfixturevalue(data)

    X_mesh            = S_train.spsampler.meshspace(S_train.xl, S_train.xu, 100)
    r2_evaluator      = R2Evaluator(S_train)
    know_evaluator    = KnowledgeEvaluator(S.knowledge, X_mesh)
    layered_evaluator = LayeredEvaluator(know_evaluator, r2_evaluator)
    
    stree = parse_syntax_tree(expr)
    r2    = r2_evaluator.evaluate(stree)
    n, nv = know_evaluator.evaluate(stree)
    leval = layered_evaluator.evaluate(stree)

    assert r2 == 1.0
    assert nv == 0
    assert leval.fea_ratio == 1.0 and leval.r2 == 1.0
