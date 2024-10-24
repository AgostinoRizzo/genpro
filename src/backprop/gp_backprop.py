import logging
import random
import numpy as np

import dataset
from symbols.syntax_tree import SyntaxTree
from backprop import lpbackprop
from backprop import jump_backprop
from backprop import utils
from gp.evaluator import Evaluator
from gp.evaluation import Evaluation
from backprop import config
from backprop import qp, models


class KnowledgeBackpropEvaluator(Evaluator):
    def __init__(self, knowledge:dataset.DataKnowledge):
        self.knowledge = knowledge
    
    def evaluate(self, stree:SyntaxTree):
        sat, stree_cost = lpbackprop.lpbackprop(self.knowledge, stree, None)
        return stree_cost


def evaluate(population:list[SyntaxTree], S_train:dataset.NumpyDataset, S_test:dataset.NumpyDataset) \
    -> tuple[list[SyntaxTree], dict]:
    
    cost_map = {}
    evaluator = KnowledgeBackpropEvaluator(S_train.knowledge)

    for stree in population:  # TODO: trees in population are already simplified.
        logging.debug(f"\nEvaluating: {stree}")
        cost_map[id(stree)] = evaluator.evaluate(stree)
    
    sorted_population = gp.sort_population(population, cost_map)
    return sorted_population, cost_map
            

# return a sorted population with new evaluation map and whether the problem was satisfiable.
def expand(population:list[SyntaxTree], S_train:dataset.NumpyDataset, S_test:dataset.NumpyDataset) \
    -> tuple[list[SyntaxTree], dict, bool]:

    satisfiable = False
    eval_map = {}

    for stree in population:  # TODO: trees in population are already simplified.
        logging.debug(f"\nExpanding: {stree}")
        stree_pr  = stree.diff().simplify()
        stree_pr2 = stree_pr.diff().simplify()

        best_unkn_models = {}  # for this tree.
        best_eval = None

        def onsynth_callback(synth_unkn_models:dict):
            nonlocal best_unkn_models
            nonlocal best_eval
            
            hist, __best_unkn_models, __best_eval = \
                jump_backprop.jump_backprop(stree, stree_pr, stree_pr2, synth_unkn_models, S_train, S_test, max_rounds=config.MAX_BACKJUMP_ROUNDS)

            if best_eval is None or __best_eval.better_than(best_eval):
                best_unkn_models = __best_unkn_models
                best_eval = __best_eval

        if lpbackprop.lpbackprop(S_train.knowledge, stree, onsynth_callback):
            satisfiable = True
        
        if best_eval is not None:
            # set best unknown models of this tree.
            for unkn_label, unkn_model in best_unkn_models.items():
                stree.set_unknown_model(unkn_label, unkn_model)
            eval_map[id(stree)] = best_eval
        else:
            # TODO: this should not happen because we invoke on satisfiable strees.
            #assert False
            # set default (identity) as unknown model of this tree.
            stree.set_all_unknown_models(lambda x: x)  # TODO: manage default model.
            eval_map[id(stree)] = dataset.Evaluation( S_train.evaluate(stree.compute_output),
                                                      S_test.evaluate(stree.compute_output),
                                                      S_train.knowledge.evaluate( (stree.compute_output,
                                                                                   stree_pr.compute_output,
                                                                                   stree_pr2.compute_output) ) )
    
    sorted_population = gp.sort_population(population, eval_map)
    return sorted_population, eval_map, satisfiable


class BackpropEvaluation(Evaluation):
    def __init__(self, std, nan):
        super().__init__(minimize=True)
        self.std = std
        self.nan = nan
    
    def better_than(self, other) -> bool:
        if self.nan < other.nan: return True
        if self.nan > other.nan: return False
        return self.std < other.std
    
    def get_value(self) -> float:
        return self.std if self.nan == 0 else -self.std

    def __str__(self) -> str:
        return f"std: {self.std}, nan: {self.nan}"


def compute_dispersion(y):
    nnan_count = np.sum(~np.isnan(y) & ~np.isinf(y))
    if nnan_count <= 1: return 100000.0  # TODO: infty
    num = np.std(y[~np.isnan(y) & ~np.isinf(y)])
    if num == 0:
        print()
    #if num < 2: return 100000.0  # TODO: infty
    return num / nnan_count


class Backpropagator:
    def __init__(self, S_data:dataset.NumpyDataset, know:dataset.DataKnowledge):
        self.S_data = S_data
        self.know = know
        self.rounds = 1
        self.data_std = compute_dispersion(S_data.y)

    def backprop(self, stree:SyntaxTree) -> SyntaxTree:
        synth_nodes = set()
        std_counts = [self.data_std]
        nan_counts = [0]

        for _ in range(self.rounds):
            nodeSelector = backprop.SyntaxTreeNodeSelector(random.randrange(stree.get_nnodes()))
            stree.accept(nodeSelector)
            node = nodeSelector.node

            if id(node) in synth_nodes or id(node) == id(stree): continue

            stree.set_parent()

            try:
                # data backprop.
                stree(self.S_data.X)
                y_data_pulled, _ = node.pull_output(self.S_data.y)
                S_data_pulled = dataset.NumpyDataset(nvars=self.S_data.nvars)
                S_data_pulled.xl = self.S_data.xl
                S_data_pulled.xu = self.S_data.xu
                S_data_pulled.numlims.set_bounds(self.S_data.xl, self.S_data.xu)
                S_data_pulled.X = self.S_data.X
                S_data_pulled.y = y_data_pulled
                S_data_pulled.on_y_changed()

                # knowledge backprop.
                S_know = self.know.synth_dataset(())
                stree(S_know.X)
                y_know_pulled, _ = node.pull_output(S_know.y, flatten=True)                
                S_know_pulled = dataset.NumpyDataset(nvars=self.S_data.nvars)
                S_know_pulled.xl = self.S_data.xl
                S_know_pulled.xu = self.S_data.xu
                S_know_pulled.numlims.set_bounds(self.S_data.xl, self.S_data.xu)
                S_know_pulled.X = S_know.X
                S_know_pulled.y = y_know_pulled
                S_know_pulled.on_y_changed()

                # collect stats.
                if np.isnan( compute_dispersion(S_data_pulled.y) ):
                    print()
                if compute_dispersion(S_data_pulled.y) == 0:
                    print()
                
                std_counts.append( compute_dispersion(S_data_pulled.y) )
                nan_counts.append( np.sum(np.isnan(S_know_pulled.y)) )

                """synth_node = self.__synth_node_qp(S_data_pulled, S_know_pulled)
                if synth_node is not None:
                    found = type(stree) is backprop.BinaryOperatorSyntaxTree and stree.operator == '/' and \
                        type(stree.left) is backprop.VariableSyntaxTree and id(node) == id(stree.right)
                    
                    #if found: print(f"\nReplacing {stree}")
                    stree = gp.replace_subtree(stree, node, synth_node)
                    stree.clear_cache()
                    synth_nodes.add(id(synth_node))
                    #if found: print(f"With {stree}")"""

            except RuntimeError:  # TODO: use backprop exception.
                continue
        
        std_counts = np.array(std_counts)
        nan_counts = np.array(nan_counts)
        #if np.isnan(std_counts).any():
        #    raise RuntimeError()
        return BackpropEvaluation( std_counts.min(), nan_counts.min() )
    
    def __synth_node_random(self, S_data, S_know):
        creator = gp.RandomSolutionCreator(nvars=S_data.nvars, randstate=None)
        pop = creator.create_population(popsize=1000, max_depth=2)

        best_p = None
        best_mse = None

        for p in pop:
            y = p(S_data.X)
            K_y = utils.flatten( p(S_know.X) )

            data_mse = np.nanmean((y - S_data.y)**2)
            K_mse = np.mean((K_y - S_know.y)**2)

            if np.isnan(K_mse) or K_mse != 0: continue
            if best_p is None or data_mse < best_mse:
                best_p = p
                best_mse = data_mse
        
        return best_p
    
    def __synth_node_qp(self, S_data, S_know):
        # remove nan and inf values from X and Y.
        S_data.clear()
        
        # create knowledge
        K = self.__create_knowledge(S_know)

        constrs = {}
        derivs = K.get_derivs()
        for deriv in derivs:
            constrs[deriv] = qp.get_constraints(K, break_points=[], deriv=deriv, sample_size=10)

        P_coeffs = qp.qp_solve(
            constrs=constrs,
            polydeg=3,
            model_nvars=S_data.nvars,
            limits=S_data.numlims,
            S=S_data
            )

        P = models.ModelFactory.create_poly(deg=3, nvars=S_data.nvars)
        P.set_coeffs(P_coeffs)
        
        #return P.to_stree()
        unkn = backprop.UnknownSyntaxTree('A')
        unkn.set_unknown_model('A', P)
        return unkn
    
    def __create_knowledge(self, S_know):
        K = dataset.DataKnowledge(limits=self.S_data.numlims, spsampler=self.know.spsampler)

        l = None
        u = None
        sign = None
        for i, x in enumerate(S_know.X):
            curr_sign = '+' if S_know.y[i] >= 0.0 else '-'
            if sign is None:
                l = x[0]
                sign = curr_sign
            elif curr_sign != sign:
                u = S_know.X[i-1][0]
                K.add_sign((), l, u, sign)
                l = x[0]
                u = None
                sign = curr_sign
        if l is not None and u is None:
            u = S_know.X[-1][0]
            K.add_sign((), l, u, sign)
        
        return K
