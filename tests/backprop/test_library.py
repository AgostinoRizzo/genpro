import pytest
import numpy as np
from backprop.library import Library
import dataset, dataset_misc1d


@pytest.fixture
def data():
    S = dataset_misc1d.MagmanDatasetScaled()
    S.sample(size=200, noise=0.03, mesh=False)
    return dataset.NumpyDataset(S)


def test_uniqueness(data):
    lib = Library(2000, 3, data)
    
    all_semantics = lib.sem_index.data
    n_semantics = all_semantics.shape[0]

    for i in range(n_semantics - 1):
        for j in range(i + 1, n_semantics):
            d = np.linalg.norm(all_semantics[i] - all_semantics[j])
            assert d > Library.DIST_EPSILON
    
    _, _, d = lib.find_best_similarity()
    assert d > Library.DIST_EPSILON
