import sys
sys.path.append('..')

import logging
import csv
import random
import time

import dataset
import dataset_feynman
import dataset_hlab
import gp_backprop
import gp
import numbs
import sympy


BENCHMARKS:list[dataset.Dataset] = [
    #(dataset.MagmanDatasetScaled(), 'magman.csv', False, 'data'),
    #(dataset.MagmanDatasetScaled(), None, False, 'sample'),
    #(dataset.MagmanDatasetScaled(), None, True, 'sample-noise'),
    #(dataset.ABSDataset(), 'abs.csv', False, 'data'),
    #(dataset.ABSDataset(), 'abs-noise.csv', False, 'data-noise'),
    #(dataset_feynman.FeynmanICh6Eq20a (), None, False, 'sample'),
    #(dataset_feynman.FeynmanICh6Eq20a (), None, True, 'sample-noise'),
    #(dataset_feynman.FeynmanICh29Eq4  (), None, False, 'sample'),
    #(dataset_feynman.FeynmanICh29Eq4  (), None, True, 'sample-noise'),
    #(dataset_feynman.FeynmanICh34Eq27 (), None, False, 'sample'),
    #(dataset_feynman.FeynmanICh34Eq27 (), None, True, 'sample-noise'),
    #(dataset_feynman.FeynmanIICh8Eq31 (), None, False, 'sample'),
    #(dataset_feynman.FeynmanIICh8Eq31 (), None, True, 'sample-noise'),
    #(dataset_feynman.FeynmanIICh27Eq16(), None, False, 'sample'),
    #(dataset_feynman.FeynmanIICh27Eq16(), None, True, 'sample-noise'),
    #(dataset_hlab.NguyenF1(), 'nguyen-f1.csv', False, 'data'),
    #(dataset_hlab.NguyenF4(), 'nguyen-f4.csv', False, 'data'),
    #(dataset_hlab.NguyenF7(), 'nguyen-f7.csv', False, 'data'),
    #(dataset_hlab.Keijzer7(), 'keijzer-7.csv', False, 'data'),
    (dataset_hlab.Keijzer8(), 'keijzer-8.csv', False, 'data'),
]

SAMPLE_SIZE = 200
NOISE = 0.03

POPSIZE = 20
MAX_STREE_DEPTH = 2
GENERATIONS = 20
GROUP_SIZE = 5  # tournament selector.
MUTATION_RATE = 0.15
ELITISM = 1

NBESTS = 5
RSEED = 0

RESULTS_APPEND_MODE = True


exprs = ''
perftable_header = [
    'Problem'  , 'Nops',
    'Train-MSE', 'Train-RMSE', 'Train-R2',
    'Test-MSE' , 'Test-RMSE' , 'Test-R2' ,
    'K-MSE0'   , 'K-MSE1'    , 'K-MSE2'  , 'K-MSE-Mean',
    'Extra-MSE', 'Extra-RMSE', 'Extra-R2',
    'EvoTime'  , 'ExpTime'   , 'TotTime'
    ]
perftable = []

logging.basicConfig(level=logging.INFO, format='')
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

random.seed(RSEED)

for S, datafile, addnoise, desc in BENCHMARKS:

    S_name = S.get_name() + '-' + desc
    logging.info(f"\n--- SR Problem {S_name} ---")
    
    if datafile is None: S.sample(size=SAMPLE_SIZE, noise=(NOISE if addnoise else 0.), mesh=False)
    else: S.load(f"../data/{datafile}")
    S.split()
    logging.info(f"\n--- Dataset loaded (Train size: {len(S.data)}, Test size: {len(S.test)}) ---")

    S.index()
    numbs.init(S)

    S_train = dataset.NumpyDataset(S)
    S_test  = dataset.NumpyDataset(S, test=True)

    logging.info(f"--- Generating initial population ({POPSIZE} individuals) ---")
    population = gp_backprop.random_population(popsize=POPSIZE, max_depth=MAX_STREE_DEPTH)
    
    logging.info(f"--- Evolve current population ({GENERATIONS} generations) ---")
    start_time = time.time()
    symb_regressor = \
         gp.GP(population, GENERATIONS, S_train, S_test,
               evaluator=gp_backprop.KnowledgeBackpropEvaluator(S.knowledge),
               selector=gp.TournamentSelector(GROUP_SIZE),
               crossover=gp.SubTreeCrossover(),
               mutator=gp.Mutator(MAX_STREE_DEPTH),
               mutrate=MUTATION_RATE,
               elitism=ELITISM,
               nbests=NBESTS)
    bests, eval_map = symb_regressor.evolve()
    best_stree = bests[0]
    best_eval = eval_map[id(best_stree)]
    evolution_time = time.time() - start_time

    logging.info("--- Best syntax tree ---")
    logging.info(best_stree)
    logging.info(best_eval)
    for i in range(1, len(bests)):
        logging.info(bests[i])
        logging.info(eval_map[id(bests[i])])
    
    logging.info(f"--- Expanding top strees ---")
    start_time = time.time()
    expanded_strees, eval_map, satisfiable = gp_backprop.expand(bests, S_train, S_test)
    best_stree = expanded_strees[0]
    best_eval = eval_map[id(best_stree)]
    expansion_time = time.time() - start_time
    for i in range(len(expanded_strees)):
        logging.info(expanded_strees[i])
        logging.info(eval_map[id(expanded_strees[i])])
    
    # compute extrapolation measure (w.r.t. the real model).
    extra_eval = S.evaluate_extra(best_stree.compute_output)
    
    sympy_expr = best_stree.to_sympy()
    sympy_expr_simpl = best_stree.to_sympy(dps=2)
    
    nops = sympy_expr_simpl.count_ops(visual=False)
    perftable.append([
        S_name, nops,
        best_eval.training ['mse'] , best_eval.training ['rmse'], best_eval.training ['r2'],
        best_eval.testing  ['mse'] , best_eval.testing  ['rmse'], best_eval.testing  ['r2'],
        best_eval.knowledge['mse0'], best_eval.knowledge['mse1'], best_eval.knowledge['mse2'],
        (best_eval.knowledge['mse0']+best_eval.knowledge['mse1']+best_eval.knowledge['mse2']) / 3.,
        extra_eval['mse'] , extra_eval['rmse'], extra_eval['r2'],
        evolution_time, expansion_time, evolution_time+expansion_time
        ])

    exprs += f"% {S_name}\n{sympy.latex(sympy_expr)}\n{sympy.latex(sympy_expr_simpl)}\n\n"

    logging.info(f"--- Saving plots in results/{S_name}.pdf and results/{S_name}-wide.pdf ---")
    S.plot(width=8, height=7, model=best_stree.compute_output, savename=f"results/{S_name}.pdf")
    S.plot(width=8, height=7, model=best_stree.compute_output, zoomout=4., savename=f"results/{S_name}-wide.pdf")

# save performance table as csv file.
logging.info(f"\n--- Saving performance table in results/perf.csv ---")
file_access_mode = 'a' if RESULTS_APPEND_MODE else 'w'
with open('results/perf.csv', file_access_mode, newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    if not RESULTS_APPEND_MODE:
        csvwriter.writerow(perftable_header)
    csvwriter.writerows(perftable)

# save LaTeX expressions as tex file.
logging.info(f"--- Saving LaTeX expressions in results/exprs.tex ---")
with open('results/exprs.tex', file_access_mode) as texfile:
    texfile.write(exprs)