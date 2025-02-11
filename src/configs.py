import dataset
import space
from gp import gp
from gp import creator as gp_creator
from gp import evaluator as gp_evaluator, selector as gp_selector
from gp import crossover as gp_crossover, mutator as gp_mutator
from gp import corrector as gp_corrector
from symbols import syntax_tree
import randstate
from enum import Enum


class SymbregConfig:
    def get_symbreg(self):
        return None


"""
    SAMPLE_SIZE        = 100
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

    LIBSIZE       = 10000
    LIB_MAXDEPTH  = 3
    LIB_MAXLENGTH = 10

    RANDSTATE = None #124
"""

class FitnessConfig(Enum):
    DATA_ONLY = 1
    LAYERED   = 2

class CorrectorConfig(Enum):
    OFF        = 1
    IMPROVE    = 2
    STOCHASTIC = 3


class GPConfig(SymbregConfig):
    SAMPLE_SIZE = 150
    SAMPLE_TRAIN_SIZE  = 1.0/3.0
    DATASET_TRAIN_SIZE = 0.7
    NOISE       = 0.05
    MESH_SIZE   = 100
    TEST_MESH_SIZE   = 200

    POPSIZE          = 500
    MAX_STREE_DEPTH  = 8
    MAX_STREE_LENGTH = 20
    GENERATIONS      = 50 #20
    GROUP_SIZE       = 3  # tournament selector.
    MUTATION_RATE    = 0.15
    ELITISM          = 1

    LIBSIZE       = 20000
    LIB_MAXDEPTH  = 3 #5
    LIB_MAXLENGTH = 10 #15

    BACKPROP_TRIALS = 2

    RANDSTATE = None #1245

    def __init__(self, S:dataset.Dataset, datafile:str=None, noisy:bool=False,
                 fitness_config:FitnessConfig=FitnessConfig.DATA_ONLY,
                 corrector_config:CorrectorConfig=CorrectorConfig.OFF):

        randstate.setstate(GPConfig.RANDSTATE)

        if corrector_config != CorrectorConfig.OFF:  # when corrector is active...
            self.GENERATIONS = 20

        if datafile is None:
            S.sample(size=GPConfig.SAMPLE_SIZE, noise=(GPConfig.NOISE if noisy else 0.0), mesh=False)
            S.split(train_size=GPConfig.SAMPLE_TRAIN_SIZE)
        else:
            S.load(datafile)
            S.split(train_size=GPConfig.DATASET_TRAIN_SIZE)

        self.S = S
        self.S_train = dataset.NumpyDataset(S)
        self.S_test  = dataset.NumpyDataset(S, test=True)
        self.S_test_extra  = dataset.NumpyDataset(S, test_extra=True)

        syntax_tree.SyntaxTreeInfo.set_problem(self.S_train)

        const_prob = 0.0 if self.S.knowledge.has_symmvars() else 0.5
        self.solutionCreator = gp_creator.PTC2RandomSolutionCreator(nvars=S.nvars, const_prob=const_prob)

        self.multiMutator = gp_mutator.MultiMutator(
            gp_mutator.SubtreeReplacerMutator(GPConfig.MAX_STREE_DEPTH, GPConfig.MAX_STREE_LENGTH, self.solutionCreator),
            gp_mutator.FunctionSymbolMutator(),
            gp_mutator.NumericParameterMutator(all=True),
            gp_mutator.NumericParameterMutator(all=False)
            )

        mesh                = space.MeshSpace(self.S_train, self.S.knowledge, GPConfig.MESH_SIZE)
        test_mesh           = space.MeshSpace(self.S_train, self.S.knowledge, GPConfig.TEST_MESH_SIZE)
        know_evaluator      = gp_evaluator.KnowledgeEvaluator(self.S.knowledge, mesh)
        test_know_evaluator = gp_evaluator.KnowledgeEvaluator(self.S.knowledge, test_mesh)
        
        linscaler           = None #gp_evaluator.LinearScaler(self.S_train.y) if corrector_config == CorrectorConfig.OFF else \
                              #gp_evaluator.ConstraintsPassLinearScaler(self.S_train.y)
        nmse_evaluator            = gp_evaluator.NMSEEvaluator(self.S_train, linscaler=linscaler)
        nmse_test_evaluator       = gp_evaluator.NMSEEvaluator(self.S_test)
        self.nmse_test_extra_evaluator = gp_evaluator.NMSEEvaluator(self.S_test_extra)
        self.r2_evaluator              = gp_evaluator.R2Evaluator(self.S_train, linscaler=linscaler)
        self.r2_test_evaluator         = gp_evaluator.R2Evaluator(self.S_test)
        self.r2_test_extra_evaluator   = gp_evaluator.R2Evaluator(self.S_test_extra)
        
        self.evaluator      = gp_evaluator.LayeredEvaluator(know_evaluator, nmse_evaluator, know_pressure=(0.0 if fitness_config==FitnessConfig.DATA_ONLY else 1.0))
        self.test_evaluator = gp_evaluator.LayeredEvaluator(test_know_evaluator, nmse_test_evaluator, know_pressure=(0.0 if fitness_config==FitnessConfig.DATA_ONLY else 1.0))
        self.test_extra_evaluator = gp_evaluator.LayeredEvaluator(test_know_evaluator, self.nmse_test_extra_evaluator, know_pressure=(0.0 if fitness_config==FitnessConfig.DATA_ONLY else 1.0))

        self.selector  = gp_selector.TournamentSelector(GPConfig.GROUP_SIZE)
        self.crossover = gp_crossover.SubTreeCrossover(GPConfig.MAX_STREE_DEPTH, GPConfig.MAX_STREE_LENGTH)
        self.corrector = None
        if corrector_config == CorrectorConfig.IMPROVE:
            self.corrector = gp_corrector.Corrector(
                self.S_train, self.S.knowledge, GPConfig.MAX_STREE_DEPTH, GPConfig.MAX_STREE_LENGTH, mesh,
                GPConfig.LIBSIZE, GPConfig.LIB_MAXDEPTH, GPConfig.LIB_MAXLENGTH, self.solutionCreator)
        elif corrector_config == CorrectorConfig.STOCHASTIC:
            self.corrector = gp_corrector.StochasticCorrector(
                self.S_train, self.S.knowledge, GPConfig.MAX_STREE_DEPTH, GPConfig.MAX_STREE_LENGTH, mesh,
                GPConfig.LIBSIZE, GPConfig.LIB_MAXDEPTH, GPConfig.LIB_MAXLENGTH, self.solutionCreator)
        
        if self.corrector is not None:
            self.corrector.backprop_trials = self.BACKPROP_TRIALS

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
import dataset_feynmannd
import dataset_misc1d
import dataset_misc2d
import dataset_misc3d
import dataset_physics

SYMBREG_BENCHMARKS = \
[
    # problem, dataset filename (sampled data if None)

    # feynman 1d (partial domain definition for all).
    (dataset_feynman1d.FeynmanICh6Eq20a (), None),
    (dataset_feynman1d.FeynmanIICh8Eq31 (), None),
    
    # misc 1d.
    (dataset_misc1d.MagmanDataset(), None),  # partial domain definition.
    (dataset_misc1d.MagmanDataset(), 'data/magman.csv'),
    (dataset_misc1d.Nguyen7(), None),  # partial domain definition.
    (dataset_misc1d.R1(), None),  # partial domain definition.
    (dataset_misc1d.R2(), None),  # partial domain definition.

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
    (dataset_feynman2d.FeynmanICh6Eq20   (), None),
    (dataset_feynmannd.FeynmanICh41Eq16  (), None),
    (dataset_feynmannd.FeynmanICh48Eq20  (), None),
    (dataset_feynmannd.FeynmanIICh6Eq15a (), None),
    (dataset_feynmannd.FeynmanIICh11Eq27 (), None),
    (dataset_feynmannd.FeynmanIICh11Eq28 (), None),
    (dataset_feynmannd.FeynmanIIICh10Eq19(), None),
]