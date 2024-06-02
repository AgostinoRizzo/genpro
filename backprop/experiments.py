import sys
import logging
sys.path.append('..')

import csv
import dataset
import dataset_feynman
import dataset_hlab
import gp_backprop
import numbs
import sympy


BENCHMARKS:list[dataset.Dataset] = [
    (dataset.MagmanDatasetScaled(), 'magman.csv', 'data'),
    #(dataset.MagmanDatasetScaled(), None, 'sample'),
    #(dataset.ABSDataset(), 'abs.csv', 'data'),
    #(dataset.ABSDataset(), 'abs-noise.csv', 'data-noise'),
    #(dataset_feynman.FeynmanICh6Eq20a (), None, 'sample'),
    #(dataset_feynman.FeynmanICh29Eq4  (), None, 'sample'),
    #(dataset_feynman.FeynmanICh34Eq27 (), None, 'sample'),
    #(dataset_feynman.FeynmanIICh8Eq31 (), None, 'sample'),
    #(dataset_feynman.FeynmanIICh27Eq16(), None, 'sample'),
    #(dataset_hlab.NguyenF1(), 'nguyen-f1.csv', 'data'),
    #(dataset_hlab.NguyenF4(), 'nguyen-f4.csv', 'data'),
    #(dataset_hlab.NguyenF7(), 'nguyen-f7.csv', 'data'),
    #(dataset_hlab.Keijzer7(), 'keijzer-7.csv', 'data'),
    #(dataset_hlab.Keijzer8(), 'keijzer-8.csv', 'data'),
]

SAMPLE_SIZE = 200
NOISE = 0.03

POPSIZE = 20
MAX_STREE_DEPTH = 2


exprs = ''
perftable_header = [
    'Problem'  , 'Nops',
    'Train-MSE', 'Train-RMSE', 'Train-R2',
    'Test-MSE' , 'Test-RMSE' , 'Test-R2' ,
    'K-MSE0',    'K-MSE1',     'K-MSE2'
    ]
perftable = []

logging.basicConfig(level=logging.INFO, format='')
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

for S, datafile, desc in BENCHMARKS:

    S_name = S.get_name() + '-' + desc
    logging.info(f"\n--- SR Problem {S_name} ---")
    
    if datafile is None: S.sample(size=SAMPLE_SIZE, noise=NOISE, mesh=False)
    else: S.load(f"../data/{datafile}")
    S.split()
    logging.info(f"\n--- Dataset loaded (Train size: {len(S.data)}, Test size: {len(S.test)}) ---")

    S.index()
    numbs.init(S)

    S_train = dataset.NumpyDataset(S)
    S_test  = dataset.NumpyDataset(S, test=True)

    logging.info(f"--- Generating initial population ({POPSIZE} individuals) ---")
    population = gp_backprop.random_population(popsize=POPSIZE, max_depth=MAX_STREE_DEPTH)
    
    logging.info(f"--- Evaluating current population ---")
    sorted_population, eval_map, satisfiable = gp_backprop.evaluate(population, S_train, S_test)
    best_stree = sorted_population[0]
    best_eval = eval_map[id(best_stree)]

    logging.info(f"--- Evaluation was {'satisfiable' if satisfiable else 'unsatisfiable'} ---")
    logging.info("--- Best syntax tree ---")
    logging.info(best_stree)
    logging.info(best_eval)
    for i in range(len(sorted_population)):
        logging.info(sorted_population[i])
        logging.info(eval_map[id(sorted_population[i])])
    
    logging.info(f"--- Expanding top strees ---")
    expanded_strees, eval_map, satisfiable = gp_backprop.expand(sorted_population[:5], S_train, S_test)
    best_stree = expanded_strees[0]
    best_eval = eval_map[id(best_stree)]
    for i in range(len(expanded_strees)):
        logging.info(expanded_strees[i])
        logging.info(eval_map[id(expanded_strees[i])])
    
    sympy_expr = best_stree.to_sympy()
    sympy_expr_simpl = best_stree.to_sympy(dps=2)
    
    nops = sympy_expr_simpl.count_ops(visual=False)
    perftable.append([
        S_name, nops,
        best_eval.training ['mse'] , best_eval.training ['rmse'], best_eval.training ['r2'],
        best_eval.testing  ['mse'] , best_eval.testing  ['rmse'], best_eval.testing  ['r2'],
        best_eval.knowledge['mse0'], best_eval.knowledge['mse1'], best_eval.knowledge['mse2']
        ])

    exprs += f"% {S_name}\n{sympy.latex(sympy_expr)}\n{sympy.latex(sympy_expr_simpl)}\n\n"

    logging.info(f"--- Saving plots in results/{S_name}.pdf and results/{S_name}-wide.pdf ---")
    S.plot(width=8, height=7, model=best_stree.compute_output, savename=f"results/{S_name}.pdf")
    S.plot(width=8, height=7, model=best_stree.compute_output, zoomout=4., savename=f"results/{S_name}-wide.pdf")

# save performance table as csv file.
logging.info(f"\n--- Saving performance table in results/perf.csv ---")
with open('results/perf.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(perftable_header)
    csvwriter.writerows(perftable)

# save LaTeX expressions as tex file.
logging.info(f"--- Saving LaTeX expressions in results/exprs.tex ---")
with open('results/exprs.tex', 'w') as texfile:
    texfile.write(exprs)