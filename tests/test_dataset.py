import pytest
import sympy
import numpy as np

import dataset
import dataset_misc1d
import dataset_misc2d
import dataset_feynman
import dataset_hlab
import space


# mock empty dataset.
class EmptyDataset(dataset.Dataset1d):
    def __init__(self):
        super().__init__(xl=-1., xu=1.)
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
        dataset_misc1d.MagmanDatasetScaled(),
        dataset_misc1d.ABSDataset(), dataset_misc1d.ABSDatasetScaled(),
        dataset_misc2d.Resistance2(),
        EmptyDataset()
    ] + \
    load_dataset(dataset_feynman) + \
    load_dataset(dataset_hlab)


@pytest.mark.parametrize("S", DATA_SETS)
def test_evaluate_knowledge(S):
    K = S.knowledge

    func = S.get_sympy(evaluated=True)
    f_symbs = func.free_symbols
    assert len(f_symbs) == S.nvars

    varnames = S.get_varnames()
    assert len(varnames) == S.nvars

    f_symbs_map = {}
    for s in f_symbs: f_symbs_map[s.name] = s
    f_symbs = [f_symbs_map[varnames[varidx]] for varidx in range(S.nvars)]
    assert len(f_symbs) == S.nvars

    all_derivs = space.get_all_derivs(S.nvars, max_derivdeg=2)
    model_map = {}

    for deriv in all_derivs:
        
        xs = [f_symbs[varidx] for varidx in deriv]
        func_deriv = func if len(xs) == 0 else sympy.diff(func, *xs)
        func_deriv_lamb_multi = sympy.lambdify(f_symbs, func_deriv, 'numpy')
            
        func_deriv_lamb = func_deriv_lamb_multi if S.nvars == 1 else \
            lambda x, func_deriv_lamb_multi=func_deriv_lamb_multi: \
                func_deriv_lamb_multi(*( ([np.empty(0)]*S.nvars) if x.size == 0 else x.T ))

        model_map[deriv] = func_deriv_lamb
    
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
    assert len(f_symbs) == S.nvars

    varnames = S.get_varnames()
    assert len(varnames) == S.nvars

    f_symbs_map = {}
    for s in f_symbs: f_symbs_map[s.name] = s
    f_symbs = [f_symbs_map[varnames[varidx]] for varidx in range(S.nvars)]
    assert len(f_symbs) == S.nvars

    func_multi = sympy.lambdify(f_symbs, func, 'numpy')
    func = func_multi if S.nvars == 1 else \
        (lambda x: func_multi(*([np.empty(0)]*S.nvars)) if x.size == 0 else func_multi(*x.T))
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