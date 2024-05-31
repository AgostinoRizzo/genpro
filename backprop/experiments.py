import sys
import logging
sys.path.append('..')

import csv
import dataset
import gp_backprop
import numbs
import sympy


BENCHMARKS:list[dataset.Dataset] = [
    (dataset.MagmanDatasetScaled(), 'magman.csv'),
    (dataset.MagmanDatasetScaled(), None)
]

SAMPLE_SIZE = 250
NOISE = 0.03

POPSIZE = 50
MAX_STREE_DEPTH = 2


exprs = ''
perftable_header = [
    'Problem'  , 'Nops',
    'Train-MSE', 'Train-RMSE', 'Train-R2',
    'Test-MSE' , 'Test-RMSE' , 'Test-R2' ,
    'K-MSE'    , 'K-RMSE'
    ]
perftable = []

logging.basicConfig(level=logging.INFO, format='')
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

for S, datafile in BENCHMARKS:

    S_name = S.get_name() + ('' if datafile is None else '-data')
    logging.info(f"\n--- Loading dataset {S_name} ({SAMPLE_SIZE} points) ---")
    
    if datafile is None: S.sample(size=SAMPLE_SIZE, noise=NOISE, mesh=False)
    else: S.load('../data/magman.csv')
    S.split()

    S.index()
    numbs.init(S)

    S_train = dataset.NumpyDataset(S)
    S_test  = dataset.NumpyDataset(S, test=True)

    logging.info(f"--- Generating initial population ({POPSIZE} individuals) ---")
    population = gp_backprop.random_population(popsize=POPSIZE, max_depth=MAX_STREE_DEPTH)
    
    logging.info(f"--- Evaluating current population ---")
    best_stree, best_eval, satisfiable = gp_backprop.evaluate(population, S_train, S_test)

    if best_stree is None:
        logging.info(f"--- No syntax tree found: problem is {'satisfiable' if satisfiable else 'unsatisfiable'} ---")
    else:
            
        logging.info("--- Best syntax tree ---")
        logging.info(best_stree)
        logging.info(best_eval)
        
        sympy_expr = best_stree.to_sympy()
        sympy_expr_simpl = best_stree.to_sympy(dps=2)
        
        nops = sympy_expr_simpl.count_ops(visual=False)
        perftable.append([
            S_name, nops,
            best_eval.training ['mse'], best_eval.training ['rmse'], best_eval.training['r2'],
            best_eval.testing  ['mse'], best_eval.testing  ['rmse'], best_eval.testing ['r2'],
            best_eval.knowledge['mse'], best_eval.knowledge['rmse']
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