GENetic PROgramming Library (GENPRO)
===

Genpro is a Python library for experimenting with **genetic programming**, primarily conceived to solve **Symbolic Regression (SR)** problems.
As a result of research work for the master's thesis, the library features the proposed **knowledge backpropagation** technique to heuristically solve SR problems with prior knowledge (positivity, monotonicity, and symmetry) encoded as formal constraints.
Moreover, the library offers visualization tools to enhance analysis. Several Jupyter Notebooks are also available.


### Basic Usage
To solve a *Symbolic Regression* (SR) problem via the Genpro library, first let us import and instantiate the corresponding dataset. We will use the magnetic manipulation system, *magman*[^1], as problem for our example.
```Python
import dataset_misc1d

S = dataset_misc1d.MagmanDataset()
S.sample(size=100, noise=0.05)
```
In the code above, 100 data points are sampled uniformly over the input domain. A normally distributed noise is added to the target variable $y'=y + N(0.05\sigma_y)$. In case of a sampled dataset without noise, the corresponding parameter can be omitted (or set to 0).<br>
Alternatively, when available, an observed dataset can be loaded as a `csv` file via:
```Python
S.load('/data/magman.csv')
```
For some of the already included SR problems, a corresponding real-world dataset can be found in the directory `/data/` of the git repository.

After sampling/loading the dataset, this can be properly split, for training and testing, as follows:
```Python
S.split(train_size=0.7)  # 7:3 split ratio
```

For efficiency reasons, training and testing datasets can be constructed over the [NumPy](https://numpy.org/) library, namely:
```Python
import dataset

S_train = dataset.NumpyDataset(S)
S_test  = dataset.NumpyDataset(S, test=True)
```

We are now ready to construct the major genetic operators, and configure our Genetic Programming (GP) algorithm.
We start by instantiating a *solution creator* featuring the PTC2[^2] algorithm, an *evaluator* according to the mean squared error metric, and a *tournament selector* with a group size of 5.
```Python
from gp import creator   as gp_creator
from gp import evaluator as gp_evaluator
from gp import selector  as gp_selector

solutionCreator = gp_creator.PTC2RandomSolutionCreator(nvars=S.nvars)

evaluator = gp_evaluator.MSEEvaluator(S_train)
selector  = gp_selector.TournamentSelector(group_size=5)
```
Concerning recombination, a *subtree crossover* with limited tree size (maximum length and depth) is instantiated, with a *multi-mutator* where one of the following mutations is randomly applied:
*   replace subtree with a new randomly created one (via the PTC2 algorithm) according to the tree size constraints;
*   random change of a function symbol;
*   add a normally distributed change $\Delta\sim N(0,1)$ to all/one numeric parameter/s.
```Python
from gp import crossover as gp_crossover
from gp import mutator   as gp_mutator

MAX_STREE_DEPTH  = 8
MAX_STREE_LENGTH = 20

crossover = gp_crossover.SubTreeCrossover(MAX_STREE_DEPTH, MAX_STREE_LENGTH)

multiMutator = gp_mutator.MultiMutator(
      gp_mutator.SubtreeReplacerMutator(MAX_STREE_DEPTH, MAX_STREE_LENGTH, solutionCreator),
      gp_mutator.FunctionSymbolMutator(),
      gp_mutator.NumericParameterMutator(all=True),
      gp_mutator.NumericParameterMutator(all=False)
      )
```

Finally, the GP algorithm can be configured with the genetic operators above, and the additional hyperparameters such as the population size and the total number of generations:
```Python
from gp import gp

POPSIZE       = 1000
GENERATIONS   =  100
MUTATION_RATE =    0.15  # 15%
ELITISM       =    1     # a single elite

settings = gp.GPSettings(
      POPSIZE, GENERATIONS,
      MAX_STREE_DEPTH, MAX_STREE_LENGTH,
      S_train, S_test,
      creator=solutionCreator,
      evaluator=evaluator,
      selector=selector,
      crossover=crossover,
      mutator=multiMutator,
      mutrate=MUTATION_RATE,
      elitism=ELITISM)

symb_regressor = gp.GP(settings)
best_stree, best_eval = symb_regressor.evolve()
```

Finally, after obtaining the best found model, we can test it over the `S_test` dataset by creating a dedicated evaluator:
```Python
test_evaluator = gp_evaluator.MSEEvaluator(S_test)
best_stree.clear_output()  # needed to clear intermediate outputs of the model
                           # since a new set of input points will be submitted

print(best_stree)  # print the analytical expression of the best model
print(f"Train MSE: {best_eval.get_value()}")
print(f"Test  MSE: {test_evaluator.evaluate(best_stree).get_value()}")
```

Additionally, statistics about the evolution process (e.g. qualities by generation) can be properly visualized via:
```Python
symb_regressor.stats.plot()
```

<!--
Main features
===
* Syntax tree encoding with visitors
* knowledge backpropagation technique
* Constrained and unconstrained library
-->


### References

[^1]: Damsteeg, Jan-Willem, Subramanya P. Nageshrao, and Robert Babuska. "Model-based real-time control of a magnetic manipulator system." 2017 IEEE 56th Annual Conference on Decision and Control (CDC). IEEE, 2017.

[^2]: Luke, Sean. "Two fast tree-creation algorithms for genetic programming." IEEE Transactions on Evolutionary Computation 4.3 (2002): 274-283.