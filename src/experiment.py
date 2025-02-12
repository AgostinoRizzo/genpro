from configs import SYMBREG_BENCHMARKS, GPConfig, FitnessConfig, CorrectorConfig
from gp.stats import series_float_to_string, series_int_to_string
import randstate
from time import time
import statistics
import numpy as np
import pandas as pd
import csv
import sys
import traceback


APPEND_MODE = True
PERF_FILENAME = 'results/perf.csv'

algo_configs = \
[
    # Algorith Configuration: (Name, Fitness-Config, Corrector-Config).
    ('GP'         , FitnessConfig.DATA_ONLY, CorrectorConfig.OFF       ),
    ('GP-L'       , FitnessConfig.LAYERED  , CorrectorConfig.OFF       ),
    ('KBP-GP'     , FitnessConfig.DATA_ONLY, CorrectorConfig.IMPROVE   ),
    ('KBP-GP-L'   , FitnessConfig.LAYERED  , CorrectorConfig.IMPROVE   ),
    ('S-KBP-GP'   , FitnessConfig.DATA_ONLY, CorrectorConfig.STOCHASTIC),
    ('S-KBP-GP-L' , FitnessConfig.LAYERED  , CorrectorConfig.STOCHASTIC)
]

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
    'Test-Extra-NMSE',
    'Test-Extra-R2',
    'Avg-Train-NMSE',
    'Avg-Train-Fea-Ratio',
    'Ext-Conv-Fea',
    'Ext-Conv-Unfea',
    'Preproc-Time',
    'Evo-Time',
    'Actual-Libsize',
    'Model-Length',
    'Model',
    'Popstat-CurrTop-NMSE',
    'Popstat-CurrTop-Fea',
    'Popstat-CurrTop-Length'
]
perftable = []

if len(sys.argv) > 1 and sys.argv[1] == '-s':
    perf_df = pd.read_csv(PERF_FILENAME)
    print( perf_df.groupby(['Problem', 'Data-Config','Algo-Config',]).size() )
    sys.exit()

if not APPEND_MODE:
    ans = input('Append mode disabled. Are you sure to continue? [yes/no] ')
    if ans != 'yes': sys.exit()


# run experiments on benchmarks...
np.seterr(all='ignore')
for S, datafile in SYMBREG_BENCHMARKS:
    
    data_configs = ['nonoise', 'noisy'] if datafile is None else ['dataset']
    for data_conf in data_configs:

        i_algo_config = 0
        while i_algo_config < len(algo_configs):
            
            algo_config_name, fitness_config, corrector_config = algo_configs[i_algo_config]
            i_algo_config += 1

            print(f"Testing {S.get_name()}-{data_conf}-{algo_config_name}...")

            preproc_start_time = time()
            symbreg_config = GPConfig(S, datafile=datafile, noisy=(data_conf=='noisy'), fitness_config=fitness_config, corrector_config=corrector_config)
            symb_regressor = symbreg_config.create_symbreg()
            preproc_end_time = time()

            try:
                start_time = time()
                best_stree, best_eval = symb_regressor.evolve()
                end_time = time()
            except Exception as e:
                error_header = f"Exception on {S.get_name()}-{data_conf}-{algo_config_name}"# [RState={randstate.getstate()}]"
                print(f"  └─── {error_header}. See results/perf.log for details.")
                logfile = open('results/perf.log', 'a')
                logfile.write(f"{error_header}\n")
                logfile.write(f"{str(e)}\n")
                logfile.write(f"{traceback.format_exc()}\n\n")
                logfile.close()
                i_algo_config -= 1
                continue

            # R2 train, test, test_extra.
            best_stree.clear_output()
            train_r2 = symbreg_config.r2_evaluator.evaluate(best_stree).value
            best_stree.clear_output()
            test_r2 = symbreg_config.r2_test_evaluator.evaluate(best_stree).value
            best_stree.clear_output()
            text_extra_r2 = symbreg_config.r2_test_extra_evaluator.evaluate(best_stree).value
            best_stree.clear_output()

            # NMSE test_extra
            test_extra_nmse = symbreg_config.nmse_test_extra_evaluator.evaluate(best_stree).value
            best_stree.clear_output()

            # (best) test.
            best_test_eval = symbreg_config.test_evaluator.evaluate(best_stree)

            # stats.
            qualities_stats = symb_regressor.stats.get_qualities_stats()
            fesibility_stats = symb_regressor.stats.get_feasibility_stats()
            properties_stats = symb_regressor.stats.get_properties_stats()
            
            fea_front, _ = symb_regressor.fea_front_tracker.get_head(0)
            unfea_front, _ = symb_regressor.fea_front_tracker.get_head(1)
            data_lu = (0.0,1.0)
            length_lu = (1,symbreg_config.MAX_STREE_LENGTH)

            perftable.append([
                S.get_name(),  # Problem
                data_conf,  # Data-Config
                algo_config_name,  # Algo-Config
                best_eval.data_eval.value,  # Train-NMSE
                train_r2,  # Train-R2
                best_test_eval.data_eval.value,  # Test-NMSE
                test_r2,  # Test-R2
                best_eval.fea_ratio,  # Train-Fea-Ratio
                best_test_eval.fea_ratio,  # Test-Fea-Ratio
                test_extra_nmse,  # Test-Extra-NMSE
                text_extra_r2,  # Test-Extra-R2
                statistics.mean(qualities_stats.qualities['currAvg']),    # Avg-Train-NMSE
                statistics.mean(fesibility_stats.fea_ratio['currAvg']),    # Avg-Train-Fea-Ratio
                fea_front.compute_extend_of_convergence(data_lu, length_lu),  # Ext-Conv-Fea
                unfea_front.compute_extend_of_convergence(data_lu, length_lu),  # Ext-Conv-unfea
                preproc_end_time - preproc_start_time,  # Preproc-Time
                end_time - start_time,  # Evo-Time
                0 if corrector_config == CorrectorConfig.OFF else symbreg_config.corrector.lib.get_size(),  # Actual-Libsize
                best_stree.cache.nnodes,  # Model-Length
                str(best_stree.simplify()),  # Model
                series_float_to_string(qualities_stats.qualities ['currTop']),  # Popstat-CurrTop-NMSE
                series_float_to_string(fesibility_stats.fea_ratio['currTop']),  # Popstat-CurrTop-Fea
                series_int_to_string  (properties_stats.lengths  ['currTop'])   # Popstat-CurrTop-Length
            ])


# save performance table as csv file.
file_access_mode = 'a' if APPEND_MODE else 'w'
with open(PERF_FILENAME, file_access_mode, newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    if not APPEND_MODE:
        csvwriter.writerow(perftable_header)
    csvwriter.writerows(perftable)
