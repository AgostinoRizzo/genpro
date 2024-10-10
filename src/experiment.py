from configs import SYMBREG_BENCHMARKS, GPConfig
import randstate
from time import time
import numpy as np
import csv
import sys


APPEND_MODE = True

perftable_header = [
    'Problem',
    'Data-Config',
    'Constrained',
    'Train-R2',
    'Test-R2',
    'Fea-Ratio',
    'Extra-R2',
    'BestAvg-Train-R2',
    'BestWorst-Train-R2',
    'BestAvg-Fea-Ratio',
    'BestWorst-Fea-Ratio',
    'Size',
    'Time',
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
            
            constrained_str = 'constrained' if constrained else 'unconstrained'
            print(f"Testing {S.get_name()}-{data_conf}-{constrained_str}...")

            symbreg_config = GPConfig(S, datafile=datafile, noisy=data_conf=='noisy', constrained=constrained)
            symb_regressor = symbreg_config.create_symbreg()

            try:
                start_time = time()
                best_stree, best_eval = symb_regressor.evolve()
                end_time = time()
            except Exception as e:
                error_header = f"Exception on {S.get_name()}-{data_conf}-{constrained_str} [RState={randstate.getstate()}]"
                print(f"  └─── {error_header}. See results/perf.log for details.")
                logfile = open('results/perf.log', 'a')
                logfile.write(f"{error_header}\n")
                logfile.write(f"{str(e)}\n\n")
                logfile.close()
                continue

            best_stree.clear_output()
            extra_eval = S.evaluate_extra(best_stree)
            best_stree.clear_output()

            best_test_eval = symbreg_config.test_evaluator.evaluate(best_stree)

            perftable.append([
                S.get_name(),  # Problem
                data_conf,  # Config
                constrained,  # Constrained
                best_eval.r2,  # Train-R2
                best_test_eval.r2,  # Test-R2
                best_eval.fea_ratio,  # Fea-Ratio
                extra_eval['r2'],  # Extra-R2
                max(symb_regressor.stats.qualities['currAvg']),    # BestAvg-Train-R2
                max(symb_regressor.stats.qualities['currWorst']),  # BestWorst-Train-R2
                max(symb_regressor.stats.fea_ratio['currAvg']),    # BestAvg-Fea-Ratio
                max(symb_regressor.stats.fea_ratio['currWorst']),  # BestWorst-Fea-Ratio
                best_stree.cache.nnodes,  # Size
                end_time - start_time,  # Time
                str(best_stree.simplify())  # Model
            ])


# save performance table as csv file.
file_access_mode = 'a' if APPEND_MODE else 'w'
with open('results/perf.csv', file_access_mode, newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    if not APPEND_MODE:
        csvwriter.writerow(perftable_header)
    csvwriter.writerows(perftable)
