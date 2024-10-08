from configs import SYMBREG_BENCHMARKS, GPConfig
from time import time
import numpy as np
import csv


APPEND_MODE = False

perftable_header = [
    'Problem',
    'Train-R2',
    'Test-R2',
    'Fea-Ratio',
    #'Extra-R2',
    'Time'
]
perftable = []


np.seterr(all='ignore')

# run experiments on benchmarks...
for S, datafile in SYMBREG_BENCHMARKS:
    symbreg_config = GPConfig(S, datafile)
    symb_regressor = symbreg_config.create_symbreg()

    start_time = time()
    best_stree, best_eval = symb_regressor.evolve()
    end_time = time()

    best_stree.clear_output()

    perftable.append([
        S.get_name(),
        best_eval.r2,
        symbreg_config.test_evaluator.evaluate(best_stree).r2,
        best_eval.fea_ratio,
        end_time - start_time
    ])


# save performance table as csv file.
file_access_mode = 'a' if APPEND_MODE else 'w'
with open('results/perf.csv', file_access_mode, newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    if not APPEND_MODE:
        csvwriter.writerow(perftable_header)
    csvwriter.writerows(perftable)
