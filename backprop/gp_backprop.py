import sys
sys.path.append('..')

from functools import cmp_to_key

import dataset
import backprop
import lpbackprop
import jump_backprop
import utils
import logging


def random_population(popsize:int, max_depth:int, check_duplicates:bool=True) -> list[backprop.SyntaxTree]:
    assert popsize >= 1
    return backprop.SyntaxTreeGenerator().create_random(max_depth, popsize, check_duplicates)


def __sort(population:list[backprop.SyntaxTree], eval_map:dict) -> list[backprop.SyntaxTree]:
    def strees_cmp(stree1, stree2) -> int:
        nonlocal eval_map
        stree1_eval = eval_map[id(stree1)]
        stree2_eval = eval_map[id(stree2)]
        if stree1_eval.better_than(stree2_eval): return -1
        if stree2_eval.better_than(stree1_eval): return  1
        return 0
    return sorted(population, key=cmp_to_key(strees_cmp))


def evaluate(population:list[backprop.SyntaxTree], S_train:dataset.NumpyDataset, S_test:dataset.NumpyDataset) \
    -> tuple[list[backprop.SyntaxTree], dict, bool]:
    
    some_sat = False
    cost_map = {}

    for stree in population:  # TODO: trees in population are already simplified.
        logging.debug(f"\nEvaluating: {stree}")
        sat, stree_cost = lpbackprop.lpbackprop(S_train.knowledge, stree, None)
        cost_map[id(stree)] = stree_cost
        if sat: some_sat = True
    
    sorted_population = __sort(population, cost_map)
    return sorted_population, cost_map, some_sat
            

# return a sorted population with new evaluation map and whether the problem was satisfiable.
def expand(population:list[backprop.SyntaxTree], S_train:dataset.NumpyDataset, S_test:dataset.NumpyDataset) \
    -> tuple[list[backprop.SyntaxTree], dict, bool]:

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
                jump_backprop.jump_backprop(stree, stree_pr, stree_pr2, synth_unkn_models, S_train, S_test, max_rounds=1)

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
            assert False
            # set default (identity) as unknown model of this tree.
            for unkn_label, unkn_model in best_local_unkn_models.items():
                stree.set_unknown_model(unkn_label, lambda x: x)  # TODO: manage default model.
            eval_map[id(stree)] = dataset.Evaluation( S_train.evaluate(stree.compute_output),
                                                      S_test.evaluate(stree.compute_output),
                                                      S_train.knowledge.evaluate( (stree.compute_output,
                                                                                   stree_pr.compute_output,
                                                                                   stree_pr2.compute_output) ) )
    
    sorted_population = __sort(population, eval_map)
    return sorted_population, eval_map, satisfiable