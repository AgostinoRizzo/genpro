import sys
import random
import numpy as np
from scipy.special import softmax
from qpsolvers import solve_ls
import logging
sys.path.append('..')

import dataset
import backprop
import utils
import qp


# TODO: factorize with lpbackprop.py.
SAMPLE_SIZE = 20
FIT_POLYDEG = 8


class HistoryEntry:
    def __init__(self, msg:str, model_name:str,
                 pulled_S:dataset.Dataset, pulled_constrs:dict[qp.Constraints], violated_constrs:list,
                 fit_model):
        self.msg = msg
        self.model_name = model_name
        self.pulled_S = pulled_S
        self.pulled_constrs = pulled_constrs
        self.violated_constrs = violated_constrs
        self.fit_model = fit_model

class History:
    def __init__(self):
        self.entries = []
    
    def log_pull(self, unkn_stree:backprop.UnknownSyntaxTree,
                 pulled_S:dataset.Dataset, pulled_constrs:dict[qp.Constraints], violated_constrs:list=[]):
        self.entries.append(
            HistoryEntry(f"Pull from {unkn_stree}", unkn_stree.label, pulled_S, pulled_constrs, violated_constrs, unkn_stree.model))


def __fit_pulled_dataset(pulled_S:dataset.NumpyDataset, pulled_constrs:dict[dict[qp.Constraints]],
                         unknown_stree:backprop.UnknownSyntaxTree, unkn_name:str,
                         global_stree:backprop.SyntaxTree,
                         hist:History, phase:str) -> tuple[callable,callable,callable]:  # returns a fit model (and its first derivative) if successful, None otherwise.

    if pulled_S.is_empty():
        logging.debug(f"-----> 0 data!!")
        return None, None, None

    # TODO: use synth model in case no solution is returned.
    #P = np.poly1d( qp.qp_solve(pulled_constrs[unkn_name], FIT_POLYDEG, pulled_data) )
    assert type(unknown_stree.model) is np.poly1d

    data_W = utils.compute_data_weight(pulled_S, unknown_stree, global_stree)

    weak_constrs = None if phase == 'data_fit' else pulled_constrs[unkn_name]
    P = np.poly1d( qp.qp_solve(unknown_stree.constrs,
                               unknown_stree.model.c.size - 1,
                               pulled_S, data_W,
                               unknown_stree.coeffs_mask,
                               weak_constrs=weak_constrs) )
    P, _ = utils.simplify_poly(P, unknown_stree.constrs)
    P_d1 = np.polyder(P, m=1)
    P_d2 = np.polyder(P, m=2)
    return P, P_d1, P_d2


# synth_unkn_models:dict of unknown model name to (unknown model:callable, constrs:dict).
def jump_backprop(stree_d0:backprop.SyntaxTree, stree_d1:backprop.SyntaxTree, stree_d2:backprop.SyntaxTree, synth_unkn_models:dict,
                  S_train:dataset.NumpyDataset, S_test:dataset.NumpyDataset, max_rounds:int=1):

    hist = History()
    stree_d0.set_parent()
    stree_d1.set_parent()
    stree_d2.set_parent()

    #
    # set all synth_unkn_models.
    #
    for synth_unkn in synth_unkn_models.keys():
        unkn_model, unkn_model_d1, unkn_model_d2, coeffs_mask, constrs = synth_unkn_models[synth_unkn]
        for stree in [stree_d0, stree_d1, stree_d2]:
            stree.set_unknown_model(synth_unkn, unkn_model, coeffs_mask, constrs)
            stree.set_unknown_model(synth_unkn + "'" , unkn_model_d1)
            stree.set_unknown_model(synth_unkn + "''", unkn_model_d2)
    
    #
    # init best stree found.
    #
    best_unkn_models = {}
    best_eval = None

    for phase in ['data_fit', 'data_knowledge_fit']:
        #if phase == 'data_knowledge_fit': continue  # TODO: manage how to activate it.

        for _ in range(max_rounds):

            # for each unknown model.
            for unkn_name in sorted(synth_unkn_models.keys(), reverse=False):  # TODO: establish jumping policy/heuristic.

                # pull dataset and constraints.
                pulled_S = {}
                pulled_constrs = {}

                for (derivdeg, stree) in [(0, stree_d0)]:  #[(0, stree_d0), (1, stree_d1)]:

                    if unkn_name not in pulled_constrs.keys():
                        pulled_constrs[unkn_name] = {}
                    violated_constrs = []                        
                            
                    for unkn_model_derivdeg in range(derivdeg + 1):
                        if unkn_model_derivdeg > 0: continue # TODO: manage how to activate it.

                        unkn_name_d = unkn_name + ("'" * unkn_model_derivdeg)
                        if stree.count_unknown_model(unkn_name_d) != 1:
                            logging.debug(f"Cannot pull from {unkn_name_d} for derivative {str(derivdeg)}: no unique occurence")
                            continue

                        unknown_stree = stree.get_unknown_stree(unkn_name_d)
                        logging.debug(f"Pulling from {unkn_name_d} w.r.t. derivative {str(derivdeg)} (phase = {phase})")

                        #
                        # pull dataset from 'unknown_stree' 
                        #
                        if derivdeg == 0:
                            yl = 0; yu = 0
                            pulled_S[unkn_name] = dataset.NumpyDataset()
                            pulled_S[unkn_name].xl = S_train.xl
                            pulled_S[unkn_name].xu = S_train.xu

                            stree.compute_output(S_train.X)
                            #try:
                            pulled_Y, _ = unknown_stree.pull_output(S_train.Y)
                            #if type(pulled_y) is not float and not np.issubdtype(type(pulled_y), np.floating): continue  # TODO: invalid numerical backprop (use np masked array).
                            
                            pulled_S[unkn_name].X_from(S_train.X)
                            pulled_S[unkn_name].Y = pulled_Y
                            pulled_S[unkn_name].on_Y_changed()

                            #except backprop.PullError:
                            #    pass # TODO: just ignore the data point? (now it is like this).
                            
                            pulled_S[unkn_name].clear()  # remove nan and inf values from X and Y.
                            pulled_S[unkn_name].remove_outliers()
                            #pulled_S[unkn_name].minmax_scale_y()
                            pulled_S[unkn_name].synchronize_Y_limits()

                        #
                        # pull constraints from 'unknown_stree' 
                        #
                        if unkn_model_derivdeg not in pulled_constrs[unkn_name]:
                            pulled_constrs[unkn_name][unkn_model_derivdeg] = qp.Constraints()
                        if phase == 'data_knowledge_fit':
                            constrs = qp.get_constraints(S_train.knowledge, dict(), derivdeg, SAMPLE_SIZE)
                            # TODO: what about symm constraints?! (use those from ASP?)
                            for (dp, relopt) in constrs.eq_ineq:
                                out = stree.compute_output(np.array([dp.x]))
                                if relopt.check(out, dp.y): continue
                                try:
                                    pulled_th, pulled_relopt = unknown_stree.pull_output(np.array([dp.y]), relopt)
                                    pulled_th = pulled_th[0]
                                    if type(pulled_th) is not float and not np.issubdtype(type(pulled_th), np.floating): continue  # TODO: invalid numerical backprop (raise PullViolation?!).
                                    
                                    pulled_constrs[unkn_name][unkn_model_derivdeg].eq_ineq \
                                        .append( (dataset.DataPoint(dp.x, pulled_th), pulled_relopt) )
                                
                                except backprop.PullError:
                                    pass
                                except backprop.PullViolation:
                                    violated_constrs.append( (dataset.DataPoint(dp.x, dp.y), relopt) )
                            
                            if pulled_constrs[unkn_name][unkn_model_derivdeg].isempty():  # skip the rest (TODO).
                                break

                unknown_stree = stree_d0.get_unknown_stree(unkn_name)

                #
                # fit pulled dataset.
                #
                fit_model = None
                fit_model_d1 = None
                fit_model_d2 = None
                if len(violated_constrs) == 0:
                    fit_model, fit_model_d1, fit_model_d2 = __fit_pulled_dataset(pulled_S[unkn_name], pulled_constrs, unknown_stree, unkn_name, stree, hist, phase)
                else:
                    hist.log_pull(unknown_stree, pulled_S[unkn_name], pulled_constrs[unkn_name], violated_constrs)
                
                #
                # if model fit, then update the model of 'unknown_stree'.
                #
                if fit_model is not None:
                    for stree in [stree_d0, stree_d1, stree_d2]:
                        stree.set_unknown_model(unknown_stree.label, fit_model, unknown_stree.coeffs_mask, unknown_stree.constrs)  # keep the same coeffs mask and constraints.
                        stree.set_unknown_model(unknown_stree.label + "'" , fit_model_d1)
                        stree.set_unknown_model(unknown_stree.label + "''", fit_model_d2)
                    hist.log_pull(unknown_stree, pulled_S[unkn_name], pulled_constrs[unkn_name])
                    # TODO: set fit_model to others strees (i.e. derivatives).
            
            #
            # end unknown model jumping.
            #

            #
            # compute model fitting.
            # r2 over data points is ok now since 'data_fit' is the only activated phase.
            #
            model_eval = dataset.Evaluation( S_train.evaluate(stree_d0.compute_output),
                                             S_test.evaluate(stree_d0.compute_output),
                                             S_train.knowledge.evaluate( (stree_d0.compute_output,
                                                                          stree_d1.compute_output,
                                                                          stree_d2.compute_output) ) )
            if best_eval is None or model_eval.better_than(best_eval):
                for unkn_label in synth_unkn_models.keys():
                    unkn_stree = stree_d0.get_unknown_stree(unkn_label)
                    best_unkn_models[unkn_label] = unkn_stree.model
                best_eval = model_eval
            else:
                # stop rounds and go to next phase.
                #stop = True
                pass #break

    return hist, best_unkn_models, best_eval