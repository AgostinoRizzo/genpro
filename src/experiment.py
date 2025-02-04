from configs import SYMBREG_BENCHMARKS, GPConfig
import randstate
from time import time
import statistics
import numpy as np
import csv
import sys


APPEND_MODE = True

perftable_header = [
    'Problem',
    'Data-Config',
    'Algo-Config',
    'Train-NMSE',
    'Train-R2',
    'Test-NMSE',
    'Test-R2',
    'Train-Fea-Ratio',
    'Test-Fea-Ratio',
    'Extra-NMSE',
    'Extra-R2',
    'Avg-Train-NMSE',
    'Avg-Train-Fea-Ratio',
    'Ext-Conv-Fea',
    'Ext-Conv-Unfea',
    'Preproc-Time',
    'Evo-Time',
    'Model-Length',
    'Model'
]
perftable = []


np.seterr(all='ignore')

if not APPEND_MODE:
    ans = input('Append mode disabled. Are you sure to continue? [yes/no] ')
    if ans != 'yes': sys.exit()


# run experiments on benchmarks...
for S, datafile in SYMBREG_BENCHMARKS:
    
    data_configs = ['nonoise', 'noisy'] if datafile is None else ['dataset']
    for data_conf in data_configs:

        for constrained in [True, False]:
            
            algo_config = 'KBP-GP' if constrained else 'GP'
            print(f"Testing {S.get_name()}-{data_conf}-{algo_config}...")

            preproc_start_time = time()
            symbreg_config = GPConfig(S, datafile=datafile, noisy=(data_conf=='noisy'), constrained=constrained)
            symb_regressor = symbreg_config.create_symbreg()
            preproc_end_time = time()

            try:
                start_time = time()
                best_stree, best_eval = symb_regressor.evolve()
                end_time = time()
            except Exception as e:
                error_header = f"Exception on {S.get_name()}-{data_conf}-{algo_config} [RState={randstate.getstate()}]"
                print(f"  └─── {error_header}. See results/perf.log for details.")
                logfile = open('results/perf.log', 'a')
                logfile.write(f"{error_header}\n")
                logfile.write(f"{str(e)}\n\n")
                logfile.close()
                continue

            best_stree.clear_output()
            extra_eval = S.evaluate_extra(best_stree)
            best_stree.clear_output()
            train_r2 = symbreg_config.r2_evaluator.evaluate(best_stree).value
            best_stree.clear_output()
            test_r2 = symbreg_config.r2_test_evaluator.evaluate(best_stree).value


            best_test_eval = symbreg_config.test_evaluator.evaluate(best_stree)
            qualities_stats = symb_regressor.stats.get_qualities_stats()
            fesibility_stats = symb_regressor.stats.get_feasibility_stats()
            
            fea_front, _ = symb_regressor.fea_front_tracker.get_head(0)
            unfea_front, _ = symb_regressor.fea_front_tracker.get_head(1)
            data_lu = (0.0,1.0)
            length_lu = (1,symbreg_config.MAX_STREE_LENGTH)

            perftable.append([
                S.get_name(),  # Problem
                data_conf,  # Data-Config
                algo_config,  # Algo-Config
                best_eval.data_eval.value,  # Train-NMSE
                train_r2,  # Train-R2
                best_test_eval.data_eval.value,  # Test-NMSE
                test_r2,  # Test-R2
                best_eval.fea_ratio,  # Train-Fea-Ratio
                best_test_eval.fea_ratio,  # Test-Fea-Ratio
                extra_eval['nmse'],  # Extra-NMSE
                extra_eval['r2'],  # Extra-R2
                statistics.mean(qualities_stats.qualities['currAvg']),    # Avg-Train-NMSE
                statistics.mean(fesibility_stats.fea_ratio['currAvg']),    # Avg-Train-Fea-Ratio
                fea_front.compute_extend_of_convergence(data_lu, length_lu),  # Ext-Conv-Fea
                unfea_front.compute_extend_of_convergence(data_lu, length_lu),  # Ext-Conv-unfea
                preproc_end_time - preproc_start_time,  # Preproc-Time
                end_time - start_time,  # Evo-Time
                best_stree.cache.nnodes,  # Model-Length
                str(best_stree.simplify())  # Model
            ])


# save performance table as csv file.
file_access_mode = 'a' if APPEND_MODE else 'w'
with open('results/perf.csv', file_access_mode, newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    if not APPEND_MODE:
        csvwriter.writerow(perftable_header)
    csvwriter.writerows(perftable)
