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
    SAMPLE_SIZE        = 20
    SAMPLE_TRAIN_SIZE  = 0.5
    DATASET_TRAIN_SIZE = 0.7
    NOISE              = 0.05
    MESH_SIZE          = 100

    POPSIZE          = 200
    MAX_STREE_DEPTH  = 5
    MAX_STREE_LENGTH = 20
    GENERATIONS      = 25
    GROUP_SIZE       = 4  # tournament selector.
    MUTATION_RATE    = 0.15
    ELITISM          = 1

    LIBSIZE       = 2000
    LIB_MAXDEPTH  = 3
    LIB_MAXLENGTH = 10

    RANDSTATE = None #124

    def __init__(self, S:dataset.Dataset, datafile:str=None, noisy:bool=False, constrained:bool=True):
        randstate.setstate(GPConfig.RANDSTATE)

        if datafile is None:
            S.sample(size=GPConfig.SAMPLE_SIZE, noise=(GPConfig.NOISE if noisy else 0.0), mesh=False)
            S.split(train_size=GPConfig.SAMPLE_TRAIN_SIZE)
        else:
            S.load(datafile)
            S.split(train_size=GPConfig.DATASET_TRAIN_SIZE)

        self.S = S
        self.S_train = dataset.NumpyDataset(S)
        self.S_test  = dataset.NumpyDataset(S, test=True)

        syntax_tree.SyntaxTreeInfo.set_problem(self.S_train)

        const_prob = 0.0 if self.S.knowledge.has_symmvars() else 0.5
        self.solutionCreator = gp_creator.PTC2RandomSolutionCreator(nvars=S.nvars, const_prob=const_prob)

        self.multiMutator = gp_mutator.MultiMutator(
            gp_mutator.SubtreeReplacerMutator(GPConfig.MAX_STREE_DEPTH, GPConfig.MAX_STREE_LENGTH, self.solutionCreator),
            gp_mutator.FunctionSymbolMutator(),
            gp_mutator.NumericParameterMutator(all=True),
            #gp.NumericParameterMutator(all=False, y_iqr=y_iqr)
            )

        mesh                = space.MeshSpace(self.S_train, self.S.knowledge, GPConfig.MESH_SIZE)
        know_evaluator      = gp_evaluator.KnowledgeEvaluator(self.S.knowledge, mesh)
        r2_evaluator        = gp_evaluator.R2Evaluator(self.S_train)
        r2_test_evaluator   = gp_evaluator.R2Evaluator(self.S_test)
        self.evaluator      = gp_evaluator.LayeredEvaluator(know_evaluator, r2_evaluator) if constrained else \
                              gp_evaluator.UnconstrainedLayeredEvaluator(know_evaluator, r2_evaluator)
        self.test_evaluator = gp_evaluator.LayeredEvaluator(know_evaluator, r2_test_evaluator) if constrained else \
                              gp_evaluator.UnconstrainedLayeredEvaluator(know_evaluator, r2_test_evaluator)

        self.selector  = gp_selector.TournamentSelector(GPConfig.GROUP_SIZE)
        self.crossover = gp_crossover.SubTreeCrossover(GPConfig.MAX_STREE_DEPTH, GPConfig.MAX_STREE_LENGTH)
        self.corrector = gp_corrector.Corrector(
            self.S_train, self.S.knowledge, GPConfig.MAX_STREE_DEPTH, GPConfig.MAX_STREE_LENGTH, mesh, GPConfig.LIBSIZE, GPConfig.LIB_MAXDEPTH, GPConfig.LIB_MAXLENGTH, self.solutionCreator) \
            if constrained else None

    def create_symbreg(self):
        settings = gp.GPSettings \
            (
                self.POPSIZE,
                self.GENERATIONS,
                self.MAX_STREE_DEPTH,
                self.MAX_STREE_LENGTH,
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
        return gp.GP(settings)



import dataset_feynman1d
import dataset_feynman2d
import dataset_misc1d
import dataset_misc2d
import dataset_misc3d
import dataset_miscnd
import dataset_physics

SYMBREG_BENCHMARKS = \
[
    # problem, dataset filename (sampled data if None)

    # feynman 1d (partial domain definition for all).
    (dataset_feynman1d.FeynmanICh6Eq20a (), None),
    (dataset_feynman1d.FeynmanICh29Eq4  (), None),  # can be remove x/speed_of_light (same as FeynmanICh34Eq27)
    (dataset_feynman1d.FeynmanICh34Eq27 (), None),
    (dataset_feynman1d.FeynmanIICh8Eq31 (), None),
    (dataset_feynman1d.FeynmanIICh27Eq16(), None),  # almost same as FeynmanIICh8Eq31 but less "scaling" needed
    
    # misc 1d.
    (dataset_misc1d.MagmanDatasetScaled(), None),  # partial domain definition.
    (dataset_misc1d.MagmanDatasetScaled(), 'data/magman.csv'),

    # misc 2d.
    (dataset_misc2d.Resistance2(), None),  # partial domain definition.

    # misc 3d.
    (dataset_misc3d.Gravity    (), None),  # partial domain definition.
    (dataset_misc3d.Resistance3(), None),  # partial domain definition.

    # from counterexample-driven GP.
    (dataset_misc2d.Keijzer14(), None),  # partial domain definition.
    # nguyen1 + nguyen3 + nguyen4 (1dim + even/odd symm).
    (dataset_misc2d.Pagie1(), None),

    # from hlab (physics).
    (dataset_physics.AircraftLift(), None),
    (dataset_physics.RocketFuelFlow(), None),

    # from shape-constrained SR.
    (dataset_miscnd.WavePower            (), None),
    (dataset_feynman2d.FeynmanICh6Eq20   (), None),
    (dataset_feynman2d.FeynmanICh41Eq16  (), None),
    (dataset_feynman2d.FeynmanICh48Eq20  (), None),
    (dataset_feynman2d.FeynmanIICh6Eq15a (), None),
    (dataset_feynman2d.FeynmanIICh11Eq27 (), None),
    (dataset_feynman2d.FeynmanIICh11Eq28 (), None),
    (dataset_feynman2d.FeynmanIIICh10Eq19(), None),
]