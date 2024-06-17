import pytest
import sympy

import dataset
import dataset_feynman
import dataset_hlab


# mock empty dataset.
class EmptyDataset(dataset.Dataset):
    def sample(self, size:int=100, noise:float=0., mesh:bool=False):
        pass
    def func(self, x):
        return x
    def get_sympy(self, evaluated:bool=False):
        return sympy.Symbol('x')


def load_dataset(from_module) -> list:
    dsets = []
    for name in dir(from_module):
        attr = getattr(from_module, name)
        if type(attr) == type and issubclass(attr, dataset.Dataset):
            dsets.append(attr())
    return dsets


APPROX_ABS = 1.0e-8
MAX_SAMPLE_SIZE = 200
DATA_SETS = [
        dataset.MagmanDatasetScaled(),
        dataset.ABSDataset(), dataset.ABSDatasetScaled(),
        EmptyDataset()
    ] + \
    load_dataset(dataset_feynman) + \
    load_dataset(dataset_hlab)


@pytest.mark.parametrize("S", DATA_SETS)
def test_evaluate_knowledge(S):
    K = S.knowledge

    func = S.get_sympy(evaluated=True)
    f_symbs = func.free_symbols
    assert len(f_symbs) == 1

    x = next(iter(f_symbs))
    func_pr = sympy.diff(func, x)
    func_pr2 = sympy.diff(func, x, 2)

    func = sympy.lambdify(x, func, 'numpy')
    func_pr = sympy.lambdify(x, func_pr, 'numpy')
    func_pr2 = sympy.lambdify(x, func_pr2, 'numpy')

    model_map = {(): func, (0,): func_pr, (0,0): func_pr2}
    K_eval = K.evaluate(model_map)

    assert K_eval['mse0'] == pytest.approx(0., abs=APPROX_ABS)
    assert K_eval['mse1'] == pytest.approx(0., abs=APPROX_ABS)
    assert K_eval['mse2'] == pytest.approx(0., abs=APPROX_ABS)


@pytest.mark.parametrize("S", DATA_SETS)
@pytest.mark.parametrize("sample_size", [0, 1, 2, MAX_SAMPLE_SIZE])
def test_evaluate_dataset(S, sample_size):
    S.clear()
    S.sample(size=sample_size, noise=0., mesh=False)
    S.split()

    func = S.get_sympy(evaluated=True)
    f_symbs = func.free_symbols
    assert len(f_symbs) == 1

    x = next(iter(f_symbs))
    func_pr = sympy.diff(func, x)
    func_pr2 = sympy.diff(func, x, 2)

    func = sympy.lambdify(x, func, 'numpy')
    func_pr = sympy.lambdify(x, func_pr, 'numpy')
    func_pr2 = sympy.lambdify(x, func_pr2, 'numpy')

    S_eval = S.evaluate(func)
    S_numpy_eval = dataset.NumpyDataset(S).evaluate(func)
    S_numpy_test_eval = dataset.NumpyDataset(S, test=True).evaluate(func)
    S_eval_extra = S.evaluate_extra(func)

    for eval_res in [S_eval.training, S_eval.testing,
                     S_numpy_eval, S_numpy_test_eval,
                     S_eval_extra]:
        assert eval_res['mse' ] == pytest.approx(0., abs=APPROX_ABS)
        assert eval_res['rmse'] == pytest.approx(0., abs=APPROX_ABS)
        assert eval_res['r2'  ] == pytest.approx(1., abs=APPROX_ABS)