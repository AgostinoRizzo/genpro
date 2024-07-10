import dataset
import dataset_misc1d
from backprop import backprop
from backprop import gp_backprop


def test_gp_backprop_lpeval():
    # setup dataset.
    S = dataset_misc1d.ABSDataset()
    S.load('../data/abs-noise.csv')
    S.split(train_size=0.7, randstate=0)

    S_train = dataset.NumpyDataset(S)
    S_test  = dataset.NumpyDataset(S, test=True)

    # generate random population.
    population = gp_backprop.random_population(popsize=10, max_depth=2, randstate=0)
    assert len(population) == 10
    for stree in population: assert stree.get_max_depth() <= 2

    # evaluate population.
    sorted_population, eval_map = gp_backprop.evaluate(population, S_train, S_test)
    best_stree = sorted_population[0]
    best_eval = eval_map[id(best_stree)]

    assert type(best_stree) is backprop.BinaryOperatorSyntaxTree and best_stree.operator == '-' and \
           type(best_stree.left) is backprop.UnknownSyntaxTree and \
           type(best_stree.right) is backprop.VariableSyntaxTree
    assert best_eval.costvals == [0, 3]

