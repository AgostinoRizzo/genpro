import sys
sys.path.append('..')

import dataset
import backprop
import lpbackprop
import jump_backprop
import utils


def random_population(popsize:int, max_depth:int) -> list[backprop.SyntaxTree]:
    assert popsize >= 1
    return backprop.SyntaxTreeGenerator().create_random(max_depth, popsize)


# trees in population are changed (unknown models are set).
def evaluate(population:list[backprop.SyntaxTree], S_train:dataset.NumpyDataset, S_test:dataset.NumpyDataset) \
    -> backprop.SyntaxTree:  # best (global) individual.

    best_global_stree = None
    best_global_eval = None

    for stree in population:  # trees in population are already simplified.
        print(stree)
        stree_pr  = stree.diff().simplify()
        stree_pr2 = stree_pr.diff().simplify()

        best_local_unkn_models = {}  # for this tree.
        best_local_eval = None

        def onsynth_callback(synth_unkn_models:dict):
            nonlocal best_local_unkn_models
            nonlocal best_local_eval
            
            hist, __best_local_unkn_models, __best_local_eval = \
                jump_backprop.jump_backprop(stree, stree_pr, synth_unkn_models, S_train, S_test, max_rounds=1)

            if best_local_eval is None or __best_local_eval.better_than(best_local_eval):
                best_local_unkn_models = __best_local_unkn_models
                best_local_eval = __best_local_eval

        lpbackprop.lpbackprop(S_train.knowledge, stree, onsynth_callback)
        
        if best_local_eval is not None:
            # set best unknown models of this tree.
            for unkn_label, unkn_model in best_local_unkn_models.items():
                stree.set_unknown_model(unkn_label, unkn_model)
            
            # update best global.
            if best_global_eval is None or best_local_eval.better_than(best_global_eval):
                best_global_unkn_models = best_local_unkn_models
                best_global_stree = stree
                best_global_eval = best_local_eval
    
    return best_global_stree, best_global_eval
            

