import clingo
import numpy as np
import logging

import dataset
import numlims
from backprop import backprop
from backprop import utils
from backprop import qp
from backprop import config
from backprop import models


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
        self.spec += 'const_tree_node(' + \
            f"\"{self.node_id_map[id(stree)]}\",{stree.val}).\n" + \
            f"const({stree.val}).\n"
    
    def visitUnknown(self, stree:backprop.UnknownSyntaxTree):
        self.__map_node_id(stree)
        self.spec += 'unkn_tree_node(' + \
            f"\"{self.node_id_map[id(stree)]}\").\n"
        self.spec += 'deriv(' + \
            f"\"{self.node_id_map[id(stree)]}\"," + \
            f"\"{self.node_id_map[id(stree)]}'\").\n"  # TODO: implement in clingo using some built-in function for all nodes.

    def map_root(self, stree:backprop.SyntaxTree, derivdeg:int=0):
        if type(stree) is not backprop.UnknownSyntaxTree:
            self.node_id_map[id(stree)] = 'm' + ("'" * derivdeg)

    def __map_node_id(self, node):
        if id(node) in self.node_id_map.keys():
            return
        if type(node) is backprop.ConstantSyntaxTree:
            self.node_id_map[id(node)] = int(node.val)  # TODO: improve int check
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


def build_knowledge_spec(K:dataset.DataKnowledge, model_name:str):  # -> spec:str, break_points_map:dict, break_points_invmap:dict
    spec = ''

    # map all break points.
    break_points_map, break_points_invmap = utils.map_break_points(K)
    break_points = sorted(break_points_map.keys())

    # root spec.
    for deriv in K.derivs.keys():
        derivdeg = len(deriv)
        model_id = model_name + ("'" * derivdeg)
        for dp in K.derivs[deriv]:
            if dp.y == 0:  # just roots.
                spec += f"root(\"{model_id}\",{break_points_map[dp.x]}).\n"
    
    # sign spec.
    for deriv in K.sign.keys():
        derivdeg = len(deriv)
        model_id =  model_name + ("'" * derivdeg)
        for (_l,_u,sign,th) in K.sign[deriv]:
            if th == 0:  # just pos or neg.
                # for each sub-interval (contiguously) of (l,u) w.r.t. break_points.
                subpoints = [p for p in break_points if p >= _l and p <= _u]  # break_points is sorted.
                if len(subpoints) == 1: subpoints = subpoints * 2  # len(subpoints) cannot be 0.
                for i in range(len(subpoints) - 1):
                    l = subpoints[i]
                    u = subpoints[i + 1]
                    spec += f"sign(\"{model_id}\"," + \
                        f"\"{sign}\"," + \
                        f"{break_points_map[l]}," + \
                        f"{break_points_map[u]}).\n"
    
    # symmetry spec.
    for deriv in K.symm.keys():
        derivdeg = len(deriv)
        model_id = model_name + ("'" * derivdeg)
        (x, iseven) = K.symm[deriv]
        spec += f"{ 'even' if iseven else 'odd' }_symm(" + \
            f"\"{model_id}\"," + \
            f"{break_points_map[x]}).\n"

    spec += ":- tree_node(N), undef(N, _).\n"  # TODO
    return spec, break_points_map, break_points_invmap


def synthesize_unknown(unkn_label:str, K:dataset.DataKnowledge, break_points:set):
    """
    returns a model (e.g. polynomial), constraints:dict.
    """

    pulled_constrs = {}
    for derivdeg in range(3):
        pulled_constrs[derivdeg] = qp.get_constraints(K, break_points, derivdeg, SAMPLE_SIZE)
    
    poly_coeffs_pos = qp.qp_solve(pulled_constrs, SYNTH_POLYDEG, K.numlims, s_val= 1)  # in decreasing power.
    poly_coeffs_neg = qp.qp_solve(pulled_constrs, SYNTH_POLYDEG, K.numlims, s_val=-1)
    poly_coeffs = poly_coeffs_pos if np.sum(poly_coeffs_pos**2) >= np.sum(poly_coeffs_neg**2) else poly_coeffs_neg
        
    P = models.ModelFactory.create_poly(SYNTH_POLYDEG)
    P.set_coeffs(poly_coeffs)
    P.simplify_from_qp(pulled_constrs)

    return (P, None, pulled_constrs)  # TODO:coeffs_mask=None and pulled_constrs refer to the 0th derivative (image).


def synthesize_unknowns(unknown_labels:list[str], break_points_map:dict, break_points_invmap:dict,  # invmap: from asp to float.
                        asp_model, onsynth_callback, K_origin:dataset.DataKnowledge):  # passing unknown_labels for efficiency
    
    logging.debug(f"\n--- ASP Model ---\n{asp_model}\n")

    # build knowledge from ASP model.
    unkn_knowledge_map = {}
    for unkn in unknown_labels: unkn_knowledge_map[unkn] = dataset.DataKnowledge(limits=K_origin.dataset.numlims)

    for atom in asp_model.symbols(shown=True):
        unkn = atom.arguments[0].string
        derivdeg = unkn.count("'")
        unkn = unkn.replace("'", '')

        if atom.name == 'sign_unkn':
            pn = atom.arguments[1].string
            lb = break_points_invmap[ atom.arguments[2].number ]
            ub = break_points_invmap[ atom.arguments[3].number ]
            unkn_knowledge_map[unkn].add_sign(derivdeg, lb, ub, pn)
        
        elif atom.name == 'root_unkn':
            x = break_points_invmap[ atom.arguments[1].number ]
            unkn_knowledge_map[unkn].add_deriv(derivdeg, dataset.DataPoint(x, 0))
        
        elif atom.name == 'noroot_unkn':
            unkn_knowledge_map[unkn].add_noroot(derivdeg)
        
        elif atom.name == 'even_symm_unkn':
            x = break_points_invmap[ atom.arguments[1].number ]
            unkn_knowledge_map[unkn].add_symm(derivdeg, x, iseven=True)
        
        elif atom.name == 'odd_symm_unkn':
            x = break_points_invmap[ atom.arguments[1].number ]
            unkn_knowledge_map[unkn].add_symm(derivdeg, x, iseven=False)
        
        elif atom.name == 'undef_unkn':
            raise RuntimeError('undef_unkn not implemented.')  # TODO

    # synthesize each unknown model.
    logging.debug('Synthesizing each unknown model...')
    synth_unkn_models = {}
    for unkn in unknown_labels:
        synth_unkn_models[unkn] = synthesize_unknown(unkn, unkn_knowledge_map[unkn], set(break_points_map.keys()))
    logging.debug('End unknown model synthesizing')
        
    onsynth_callback(synth_unkn_models)    


# returns True when the problem is satisfiable + best model cost.
def lpbackprop(K:dataset.DataKnowledge, stree:backprop.SyntaxTree, onsynth_callback) -> bool:
    #
    # build ASP specification of stree and stree' (facts).
    #
    # TODO: take stree_map from outside?!
    stree_map = {(): stree.simplify()}  # TODO: generate all derivatives (multivar).
    stree_map.update( {(0,): stree.diff().simplify()} )
    stree_map.update( {(0,0): stree_map[(0,)].diff().simplify()} )

    aspSpecBuilder = ASPSpecBuilder()
    for deriv, stree in stree_map.items(): aspSpecBuilder.map_root(stree, len(deriv))
    for deriv, stree in stree_map.items(): stree.accept(aspSpecBuilder)
    stree_spec = aspSpecBuilder.spec
    #logging.debug(stree_spec)

    #
    # build ASP prior knowledge (K) specification.
    #
    K_spec, break_points_map, break_points_invmap = build_knowledge_spec(K, aspSpecBuilder.node_id_map[id(stree_map[()])])
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
    clingo_ctl.ground([('stree_spec', []), ('K_spec', []), ('show', []), ('base', [])])

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
        synthesize_unknowns(unknown_labels, break_points_map, break_points_invmap, asp_model, onsynth_callback, K)
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
