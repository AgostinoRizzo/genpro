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
def evaluate(population:list[backprop.SyntaxTree], S:dataset.Dataset) \
    -> backprop.SyntaxTree:  # best (global) individual.

    best_global_k_mse = None
    best_global_r2 = None
    best_global_stree = None

    for stree in population:  # trees in population are already simplified.
        print(stree)
        stree_pr  = stree.diff().simplify()
        stree_pr2 = stree_pr.diff().simplify()

        best_local_k_mse = None
        best_local_r2 = None
        best_local_unkn_models = {}  # for this tree.

        def onsynth_callback(synth_unkn_models:dict):
            nonlocal best_local_k_mse
            nonlocal best_local_r2
            nonlocal best_local_unkn_models
            
            hist, __best_local_unkn_models, __best_local_r2, __best_local_k_mse = \
                jump_backprop.jump_backprop(stree, stree_pr, synth_unkn_models, S, max_rounds=1)

            if best_local_k_mse is None or utils.compare_fit(__best_local_k_mse, __best_local_r2, best_local_k_mse, best_local_r2):
                best_local_unkn_models = __best_local_unkn_models
                best_local_k_mse = __best_local_k_mse
                best_local_r2 = __best_local_r2

        lpbackprop.lpbackprop(S.knowledge, stree, onsynth_callback)
        
        if best_local_k_mse is not None:
            # set best unknown models of this tree.
            for unkn_label, unkn_model in best_local_unkn_models.items():
                stree.set_unknown_model(unkn_label, unkn_model)
            
            # update best global.
            if best_global_k_mse is None or utils.compare_fit(best_local_k_mse, best_local_r2, best_global_k_mse, best_global_r2):
                best_global_unkn_models = best_local_unkn_models
                best_global_k_mse = best_local_k_mse
                best_global_r2 = best_local_r2
                best_global_stree = stree
    
    return best_global_stree
            

