import clingo
import numpy as np
import logging
from functools import cmp_to_key

import dataset
import numlims
import space
from backprop import backprop
from backprop import utils
from backprop import qp
from backprop import config
from backprop import models
from backprop import clingo_context


ASP_ENCODINGS_DIR = "backprop/lpbackprop/"
ASP_ENCODINGS = ["backprop.lp", "add_sub_opt.lp", "muldiv_opt.lp", "pow_opt.lp",
                 "exp_opt.lp", "log_opt.lp", "sqrt_opt.lp"]
SAMPLE_SIZE = 20
SYNTH_POLYDEG = 6


class ASPSpecBuilder(backprop.SyntaxTreeVisitor): # TODO: avoid multiple facts for unkn nodes (A, A', ...)
    def __init__(self):
        self.spec = ''
        self.node_id = 1
        self.node_id_map = {}

    def visitUnaryOperator(self, stree:backprop.UnaryOperatorSyntaxTree):
        self.__map_node_id(stree)
        self.__map_node_id(stree.inner)
        self.spec += 'un_tree_node(' + \
            f"\"{self.node_id_map[id(stree)]}\","+ \
            f"\"{stree.operator}\"," + \
            f"\"{self.node_id_map[id(stree.inner)]}\").\n"
    
    def visitBinaryOperator(self, stree:backprop.BinaryOperatorSyntaxTree):
        self.__map_node_id(stree)
        self.__map_node_id(stree.left)
        self.__map_node_id(stree.right)
        self.spec += 'bin_tree_node(' + \
            f"\"{self.node_id_map[id(stree)]}\","+ \
            f"\"{stree.operator}\"," + \
            f"\"{self.node_id_map[id(stree.left)]}\"," + \
            f"\"{self.node_id_map[id(stree.right)]}\").\n"

    def visitConstant(self, stree:backprop.ConstantSyntaxTree):
        self.__map_node_id(stree)
        const_val = int(stree.val) # TODO: just positivity?
        self.spec += 'const_tree_node(' + \
            f"\"{self.node_id_map[id(stree)]}\",{const_val}).\n" + \
            f"const({const_val}).\n"
    
    def visitVariable(self, stree:backprop.VariableSyntaxTree):
        self.__map_node_id(stree)
        self.spec += f"var_tree_node(\"{self.node_id_map[id(stree)]}\").\n"
    
    def visitUnknown(self, stree:backprop.UnknownSyntaxTree):
        self.__map_node_id(stree)
        self.spec += 'unkn_tree_node(' + \
            f"\"{self.node_id_map[id(stree)]}\").\n"
        #self.spec += 'deriv(' + \
        #    f"\"{self.node_id_map[id(stree)]}\"," + \
        #    f"\"d0{self.node_id_map[id(stree)]}\").\n"  # TODO: implement in clingo using some built-in function for all nodes.

    def map_root(self, stree:backprop.SyntaxTree, deriv:tuple[int]):
        if type(stree) is not backprop.UnknownSyntaxTree:
            self.node_id_map[id(stree)] = f"{utils.deriv_to_string(deriv)}m"

    def __map_node_id(self, node):
        if id(node) in self.node_id_map.keys():
            return
        if type(node) is backprop.ConstantSyntaxTree:
            self.node_id_map[id(node)] = int(node.val)  # TODO: improve int check
        if type(node) is backprop.VariableSyntaxTree:
            self.node_id_map[id(node)] = str(node)
        elif type(node) is backprop.UnknownSyntaxTree:
            self.node_id_map[id(node)] = f"{node.label}"
        else:
            self.node_id_map[id(node)] = f"m{self.node_id}"
            self.node_id += 1


class AspModelCost:
    def __init__(self, costvals:list[int]):
        self.costvals = costvals  # according to clasp's cost output.
    
    def append(self, cval:int):
        self.costvals.append(cval)
    
    def better_than(self, other) -> bool:
        for i in range(min(len(self.costvals), len(other.costvals))):
            if self.costvals[i] < other.costvals[i]: return True
            if self.costvals[i] > other.costvals[i]: return False
        return len(self.costvals) < len(other.costvals)
    
    def __str__(self) -> str:
        return str(self.costvals)


def build_knowledge_spec(K:dataset.DataKnowledge, model_name:str):
    """
    Returns
        spec:str,
        break_points:list,
        break_point_coords_map:dict,
        break_point_coords_invmap:dict
    """

    spec = ''
    
    # map all break points.
    break_points, break_point_coords_map, break_point_coords_invmap = utils.map_break_points(K)
    break_points = [np.array(bp, ndmin=1) for bp in break_points]

    def np_array_cmp(a1, a2) -> int:
        if (a1 < a2).all(): return -1
        if (a2 < a1).all(): return 1
        return 0
    break_points = sorted(break_points, key=cmp_to_key(np_array_cmp))

    # root spec.
    for deriv in K.derivs.keys():
        model_id = utils.deriv_to_string(deriv) + model_name
        for dp in K.derivs[deriv]:
            if dp.y == 0:  # just roots.
                dp_x_term = break_point_coords_map[dp.x] if np.isscalar(dp.x) else \
                            tuple(break_point_coords_map[coord] for coord in dp.x)
                spec += f"root(\"{model_id}\",{dp_x_term}).\n"
    
    # sign spec.
    for deriv in K.sign.keys():
        model_id = utils.deriv_to_string(deriv) + model_name
        for (_l,_u,sign,th) in K.sign[deriv]:
            if th == 0:  # just pos or neg.
                # for each sub-interval (contiguously) of (l,u) w.r.t. break_points.
                subpoints = [p for p in break_points if np.all(p >= _l) and np.all(p <= _u)]  # break_points is sorted.
                if len(subpoints) == 1: subpoints = subpoints * 2  # len(subpoints) cannot be 0.

                hcs = space.get_nested_hypercubes(subpoints)
                for hc in hcs:
                    l = hc[0]
                    u = hc[1]
                    if l.size == 1:  # u has same size!
                        l = l[0]; u = u[0]
                    l_term = break_point_coords_map[l] if np.isscalar(l) else \
                             tuple(break_point_coords_map[coord] for coord in l)
                    u_term = break_point_coords_map[u] if np.isscalar(u) else \
                             tuple(break_point_coords_map[coord] for coord in u)
                    spec += f"sign(\"{model_id}\",\"{sign}\",{l_term},{u_term}).\n"
    
    # symmetry spec.
    for deriv in K.symm.keys():
        model_id = utils.deriv_to_string(deriv) + model_name
        (x, iseven) = K.symm[deriv]
        x_term = break_point_coords_map[x] if np.isscalar(x) else \
                 tuple(break_point_coords_map[coord] for coord in x)
        spec += f"{ 'even' if iseven else 'odd' }_symm(\"{model_id}\",{x_term}).\n"

    spec += ":- tree_node(N), undef(N, _).\n"  # TODO
    spec += f"infty({break_point_coords_map[ K.numlims.INFTY]}).\n"
    spec += f"infty({break_point_coords_map[-K.numlims.INFTY]}).\n"
    for idx in range(K.nvars): spec += f"varidx(\"{idx}\").\n"
    return spec, break_points, break_point_coords_map, break_point_coords_invmap


def synthesize_unknown(unkn_label:str, K:dataset.DataKnowledge, break_points:list, nvars:int):
    """
    returns a model (e.g. polynomial), constraints:dict.
    """

    pulled_constrs = {}
    derivs = K.get_derivs()
    for deriv in derivs:
        pulled_constrs[deriv] = qp.get_constraints(K, break_points, deriv, SAMPLE_SIZE)
    
    poly_coeffs_pos = qp.qp_solve(pulled_constrs, SYNTH_POLYDEG, nvars, K.numlims, s_val= 1)  # in decreasing power.
    poly_coeffs_neg = qp.qp_solve(pulled_constrs, SYNTH_POLYDEG, nvars, K.numlims, s_val=-1)
    poly_coeffs = poly_coeffs_pos if np.sum(poly_coeffs_pos**2) >= np.sum(poly_coeffs_neg**2) else poly_coeffs_neg
        
    P = models.ModelFactory.create_poly(deg=SYNTH_POLYDEG, nvars=nvars)
    P.set_coeffs(poly_coeffs)
    P.simplify_from_qp(pulled_constrs)

    return (P, None, pulled_constrs)  # TODO:coeffs_mask=None and pulled_constrs refer to the 0th derivative (image).


def synthesize_unknowns(unknown_labels:list[str], break_points:list, break_point_coords_invmap:dict,  # invmap: from asp to float.
                        asp_model, onsynth_callback, K_origin:dataset.DataKnowledge):  # passing unknown_labels for efficiency
    
    def map_symbol(s:clingo.Symbol):
        nonlocal break_point_coords_invmap
        if s.type is clingo.SymbolType.Number:
            return break_point_coords_invmap[s.number]
        mapped_symbol = np.empty(len(s.arguments))
        for i in range(mapped_symbol.size):
            mapped_symbol[i] = break_point_coords_invmap[s.arguments[i].number]
        return mapped_symbol

    logging.debug(f"\n--- ASP Model ---\n{asp_model}\n")
    #print(f"\n--- ASP Model ---\n")
    #for m in str(asp_model).split(): print(m)
    #print()

    # build knowledge from ASP model.
    unkn_knowledge_map = {}
    for unkn in unknown_labels: unkn_knowledge_map[unkn] = dataset.DataKnowledge(limits=K_origin.dataset.numlims, spsampler=K_origin.spsampler)

    for atom in asp_model.symbols(shown=True):
        unkn = atom.arguments[0].string
        deriv, unkn = utils.parse_deriv(unkn, parsefunc=True)

        if atom.name == 'sign_unkn':
            pn = atom.arguments[1].string
            lb = map_symbol( atom.arguments[2] )
            ub = map_symbol( atom.arguments[3] )
            unkn_knowledge_map[unkn].add_sign(deriv, lb, ub, pn)
        
        elif atom.name == 'root_unkn':
            x = map_symbol( atom.arguments[1] )
            unkn_knowledge_map[unkn].add_deriv(deriv, dataset.DataPoint(x, 0))
        
        elif atom.name == 'noroot_unkn':
            unkn_knowledge_map[unkn].add_noroot(deriv)
        
        elif atom.name == 'even_symm_unkn':
            x = map_symbol( atom.arguments[1] )
            unkn_knowledge_map[unkn].add_symm(deriv, x, iseven=True)
        
        elif atom.name == 'odd_symm_unkn':
            x = map_symbol( atom.arguments[1] )
            unkn_knowledge_map[unkn].add_symm(deriv, x, iseven=False)
        
        elif atom.name == 'undef_unkn':
            raise RuntimeError('undef_unkn not implemented.')  # TODO

    # synthesize each unknown model.
    logging.debug('Synthesizing each unknown model...')
    synth_unkn_models = {}
    for unkn in unknown_labels:
        synth_unkn_models[unkn] = synthesize_unknown(unkn, unkn_knowledge_map[unkn], break_points, K_origin.nvars)
    logging.debug('End unknown model synthesizing')
        
    onsynth_callback(synth_unkn_models)    


# returns True when the problem is satisfiable + best model cost.
def lpbackprop(K:dataset.DataKnowledge, stree:backprop.SyntaxTree, onsynth_callback) -> bool:
    #
    # build ASP specification of stree and stree' (facts).
    #
    # TODO: take stree_map from outside?!
    all_derivs = K.get_derivs()
    stree_map = backprop.SyntaxTree.diff_all(stree, all_derivs, include_zeroth=True)

    aspSpecBuilder = ASPSpecBuilder()
    for deriv, stree in stree_map.items(): aspSpecBuilder.map_root(stree, deriv)
    for deriv, stree in stree_map.items(): stree.accept(aspSpecBuilder)
    stree_spec = aspSpecBuilder.spec
    #logging.debug(stree_spec)

    #
    # build ASP prior knowledge (K) specification.
    #
    K_spec, break_points, break_point_coords_map, break_point_coords_invmap = \
        build_knowledge_spec(K, aspSpecBuilder.node_id_map[id(stree_map[()])])
    #logging.debug(K_spec)

    #
    # invoke ASP solver with stree_spec and K_spec (knowledge backprop).
    #
        
    # set encodings.
    clingo_ctl = clingo.Control()
    for enc in ASP_ENCODINGS: clingo_ctl.load(ASP_ENCODINGS_DIR + enc)
    clingo_ctl.add('stree_spec', [], stree_spec)  # ASP facts (stree).
    clingo_ctl.add('K_spec'    , [], K_spec    )  # ASP facts (knowledge).
    clingo_ctl.add('show'      , [], """
        #show root_unkn/2.
        #show noroot_unkn/1.
        #show sign_unkn/4.
        #show even_symm_unkn/2.
        #show odd_symm_unkn/2.
    """)

    # set solving options.
    if onsynth_callback is None:
        clingo_ctl.configuration.solve.models = 0  # required by opt_mode=opt.
        clingo_ctl.configuration.solve.opt_mode = 'opt'  # find an optimum model (requires models=0).
    else:
        clingo_ctl.configuration.solve.models = 0  # compute all models.
        clingo_ctl.configuration.solve.opt_mode = 'optN'  # find optimum, then enumerate optimal models (models=0).
        clingo_ctl.configuration.solve.project = 'show'

    # compute asp model(s) and synthesize each unknown node in stree (when onsynth_callback is provided).
    unknown_labels = None
    if onsynth_callback is not None:
        unkn_collector = backprop.UnknownSyntaxTreeCollector()
        stree_map[()].accept(unkn_collector)
        unknown_labels = unkn_collector.unknown_labels
    
    logging.debug('Clingo grounding...')
    clingo_ctl.ground([('stree_spec', []), ('K_spec', []), ('show', []), ('base', [])], context=clingo_context.ClingoContext())

    nopt_models = 0
    best_model_cost = None

    def on_model(asp_model):
        nonlocal nopt_models
        nonlocal best_model_cost
        
        logging.debug(f"Model found (optimality proven={asp_model.optimality_proven}, cost={asp_model.cost})")
        model_cost = AspModelCost(asp_model.cost)
        if best_model_cost is None or model_cost.better_than(best_model_cost):
            best_model_cost = model_cost
        
        if onsynth_callback is None:
            return True
        
        if not asp_model.optimality_proven: return
        synthesize_unknowns(unknown_labels, break_points, break_point_coords_invmap, asp_model, onsynth_callback, K)
        nopt_models += 1
        return nopt_models < config.LPBACKPROP_MAX_NMODELS
    
    logging.debug('Clingo solving...')
    solve_result = clingo_ctl.solve(on_model=on_model)
    logging.debug(solve_result)
    
    nodes_counter = backprop.SyntaxTreeNodeCounter()
    stree_map[()].accept(nodes_counter)
    if best_model_cost is None:  # when unsat.
        best_model_cost = AspModelCost([1e5+nodes_counter.nnodes])  # TODO: manage 1e5 + ...
    else:  # when sat, add number of nodes as additional metric.
        best_model_cost.append(nodes_counter.nnodes)
    
    logging.debug(f"Best model cost: {best_model_cost}")
    return solve_result.satisfiable, best_model_cost





if __name__ == '__main__':
    S = dataset.MagmanDatasetScaled()
    unknown_stree_a = backprop.UnknownSyntaxTree('A')
    unknown_stree_b = backprop.UnknownSyntaxTree('B')
    stree = backprop.BinaryOperatorSyntaxTree('/', unknown_stree_a, unknown_stree_b)

    def onsynth_callback(synth_unkn_models:dict):
        print('--- On Synth ---')
        for unkn in synth_unkn_models.keys():
            print(f"{unkn}(x) =\n{synth_unkn_models[unkn]}")
        print()

    lpbackprop(S.knowledge, stree, onsynth_callback)
