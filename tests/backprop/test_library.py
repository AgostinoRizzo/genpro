import pytest
import numpy as np
from backprop.library import Library
import dataset, dataset_misc1d
from gp import creator as gp_creator


@pytest.fixture
def dataknow():
    S = dataset_misc1d.MagmanDatasetScaled()
    S.sample(size=200, noise=0.03, mesh=False)
    return dataset.NumpyDataset(S), S.knowledge


def test_uniqueness(dataknow):
    SIZE       = 2000
    MAX_DEPTH  =    3
    MAX_LENGTH =   15

    data, know = dataknow
    solutionCreator = gp_creator.PTC2RandomSolutionCreator(nvars=data.nvars)
    lib = Library(SIZE, MAX_DEPTH, MAX_LENGTH, data, know, solutionCreator)
    
    all_semantics = lib.sem_index.index.data
    n_semantics = all_semantics.shape[0]

    for i in range(n_semantics - 1):
        for j in range(i + 1, n_semantics):
            d = np.linalg.norm(all_semantics[i] - all_semantics[j])
            assert d > 0.0
    
    _, _, d = lib.find_best_similarity()
    assert d > 0.0
