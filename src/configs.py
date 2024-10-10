import dataset
from gp import gp
from gp import creator as gp_creator
from gp import evaluator as gp_evaluator, selector as gp_selector
from gp import crossover as gp_crossover, mutator as gp_mutator
from gp import corrector as gp_corrector
from symbols import syntax_tree
import randstate


class SymbregConfig:
    def get_symbreg(self):
        return None


class GPConfig(SymbregConfig):
    SAMPLE_SIZE = 100
    TRAIN_SIZE  = 0.7
    NOISE       = 0.0
    MESH_SIZE   = 100

    POPSIZE         = 200
    MAX_STREE_DEPTH = 5
    GENERATIONS     = 25
    GROUP_SIZE      = 5  # tournament selector.
    MUTATION_RATE   = 0.15
    ELITISM         = 1

    LIBSIZE      = 2000
    LIB_MAXDEPTH = 3

    RANDSTATE = None #124

    def __init__(self, S:dataset.Dataset, datafile:str=None, noisy:bool=False, constrained:bool=True):
        randstate.setstate(GPConfig.RANDSTATE)

        if datafile is None:
            S.sample(size=GPConfig.SAMPLE_SIZE, noise=GPConfig.NOISE if noisy else 0.0, mesh=False)
        else:
            S.load(datafile)

        S.split(train_size=GPConfig.TRAIN_SIZE)

        self.S = S
        self.S_train = dataset.NumpyDataset(S)
        self.S_test  = dataset.NumpyDataset(S, test=True)

        y_iqr = S_train.get_y_iqr()

        syntax_tree.SyntaxTreeInfo.set_problem(self.S_train)

        self.solutionCreator = gp_creator.RandomSolutionCreator(nvars=S.nvars, y_iqr=y_iqr)

        self.multiMutator = gp_mutator.MultiMutator(
            gp_mutator.SubtreeReplacerMutator(GPConfig.MAX_STREE_DEPTH, self.solutionCreator),
            gp_mutator.FunctionSymbolMutator(),
            gp_mutator.NumericParameterMutator(all=True, y_iqr=y_iqr),
            #gp.NumericParameterMutator(all=False, y_iqr=y_iqr)
            )

        X_mesh              = self.S_train.spsampler.meshspace(self.S_train.xl, self.S_train.xu, GPConfig.MESH_SIZE)
        know_evaluator      = gp_evaluator.KnowledgeEvaluator(S.knowledge, X_mesh)
        r2_evaluator        = gp_evaluator.R2Evaluator(self.S_train)
        r2_test_evaluator   = gp_evaluator.R2Evaluator(self.S_test)
        self.evaluator      = gp_evaluator.LayeredEvaluator(know_evaluator, r2_evaluator) if constrained else \
                              gp_evaluator.UnconstrainedLayeredEvaluator(know_evaluator, r2_evaluator)
        self.test_evaluator = gp_evaluator.LayeredEvaluator(know_evaluator, r2_test_evaluator) if constrained else \
                              gp_evaluator.UnconstrainedLayeredEvaluator(know_evaluator, r2_test_evaluator)

        self.selector  = gp_selector.TournamentSelector(GPConfig.GROUP_SIZE)
        self.crossover = gp_crossover.SubTreeCrossover(GPConfig.MAX_STREE_DEPTH)
        self.corrector = gp_corrector.Corrector(
            self.S_train, S.knowledge, GPConfig.MAX_STREE_DEPTH, X_mesh, GPConfig.LIBSIZE, GPConfig.LIB_MAXDEPTH) \
            if constrained else None

    def create_symbreg(self):
        return gp.GP \
            (
                self.POPSIZE,
                self.GENERATIONS,
                self.MAX_STREE_DEPTH,
                self.S_train,
                self.S_test,
                creator=self.solutionCreator,
                evaluator=self.evaluator,
                selector=self.selector,
                crossover=self.crossover,
                mutator=self.multiMutator,
                corrector=self.corrector,
                mutrate=self.MUTATION_RATE,
                elitism=self.ELITISM,
                knowledge=self.S.knowledge
            )



import dataset_feynman1d
import dataset_feynman2d
import dataset_misc1d
import dataset_misc2d
import dataset_misc3d

SYMBREG_BENCHMARKS = \
[
    # problem, dataset filename (sampled data if None)

    # feynman 1d.
    (dataset_feynman1d.FeynmanICh6Eq20a (), None),
    (dataset_feynman1d.FeynmanICh29Eq4  (), None),
    (dataset_feynman1d.FeynmanICh34Eq27 (), None),
    (dataset_feynman1d.FeynmanIICh8Eq31 (), None),
    (dataset_feynman1d.FeynmanIICh27Eq16(), None),
    
    # misc 1d.
    (dataset_misc1d.MagmanDatasetScaled(), None),
    (dataset_misc1d.MagmanDatasetScaled(), 'data/magman.csv'),

    # feynman 2d.
    (dataset_feynman2d.FeynmanICh6Eq20(), None)

    # misc 2d.
    (dataset_misc2d.Resistance2(), None)

    # misc 3d.
    (dataset_misc3d.Gravity    (), None)
    (dataset_misc3d.Resistance3(), None)
]