import sys
sys.path.append('..')

import clingo
import numpy as np

import dataset
import backprop
import numbs
import utils
import qp


ASP_ENCODINGS = ["lpbackprop/backprop.lp", "lpbackprop/add_sub_opt.lp",
                 "lpbackprop/muldiv_opt.lp", "lpbackprop/pow_opt.lp"]
SAMPLE_SIZE = 5
SYNTH_POLYDEG = 6


class ASPSpecBuilder(backprop.SyntaxTreeVisitor): # TODO: avoid multiple facts for unkn nodes (A, A', ...)
    def __init__(self):
        self.spec = ''
        self.node_id = 1
        self.node_id_map = {}

    def visitBinaryOperator(self, stree:backprop.BinaryOperatorSyntaxTree):
        self.__map_node_id(stree)
        self.__map_node_id(stree.left)
        self.__map_node_id(stree.right)
        self.spec += 'bin_tree_node(' + \
            f"{self.node_id_map[id(stree)]},"+ \
            f"\"{stree.operator}\"," + \
            f"{self.node_id_map[id(stree.left)]}," + \
            f"{self.node_id_map[id(stree.right)]}).\n"

    def visitConstant(self, stree:backprop.ConstantSyntaxTree):
        self.__map_node_id(stree)
        self.spec += 'const_tree_node(' + \
            f"{self.node_id_map[id(stree)]}).\n" + \
            f"const({stree.val}).\n"
    
    def visitUnknown(self, stree:backprop.UnknownSyntaxTree):
        self.__map_node_id(stree)
        self.spec += 'unkn_tree_node(' + \
            f"{self.node_id_map[id(stree)]}).\n"
        self.spec += 'deriv(' + \
            f"{self.node_id_map[id(stree)]}," + \
            f"{self.node_id_map[id(stree)][:-1]}'\").\n"  # TODO: implement in clingo using some built-in function for all nodes.

    def map_root(self, stree:backprop.UnknownSyntaxTree, derivdeg:int=0):
        self.node_id_map[id(stree)] = '"m' + ("'" * derivdeg) + '"'

    def __map_node_id(self, node):
        if id(node) in self.node_id_map.keys():
            return
        if type(node) is backprop.ConstantSyntaxTree:
            self.node_id_map[id(node)] = int(node.val)  # TODO: improve int check
        elif type(node) is backprop.UnknownSyntaxTree:
            self.node_id_map[id(node)] = f"\"{node.label}\""
        else:
            self.node_id_map[id(node)] = f"\"m{self.node_id}\""
            self.node_id += 1


def build_knowledge_spec(K:dataset.DataKnowledge):  # -> spec:str, break_points_map:dict, break_points_invmap:dict
    spec = ''

    # map all break points.
    break_points_map, break_points_invmap = utils.map_break_points(K)
    break_points = sorted(break_points_map.keys())

    # root spec.
    for derivdeg in K.derivs.keys():
        model_id = "'" * derivdeg
        for dp in K.derivs[derivdeg]:
            if dp.y == 0:  # just roots.
                spec += f"root(\"m{model_id}\",{break_points_map[dp.x]}).\n"
    
    # sign spec.
    for derivdeg in K.sign.keys():
        model_id = "'" * derivdeg
        for (_l,_u,sign,th) in K.sign[derivdeg]:
            if th == 0:  # just pos or neg.
                # for each sub-interval (contiguously) of (l,u) w.r.t. break_points.
                subpoints = [p for p in break_points if p >= _l and p <= _u]  # break_points is sorted.
                if len(subpoints) == 1: subpoints = subpoints * 2  # len(subpoints) cannot be 0.
                for i in range(len(subpoints) - 1):
                    l = subpoints[i]
                    u = subpoints[i + 1]
                    spec += f"sign(\"m{model_id}\"," + \
                        f"\"{sign}\"," + \
                        f"{break_points_map[l]}," + \
                        f"{break_points_map[u]}).\n"
    
    # symmetry spec.
    for derivdeg in K.symm.keys():
        model_id = "'" * derivdeg
        (x, iseven) = K.symm[derivdeg]
        spec += f"{ 'even' if iseven else 'odd' }_symm(" + \
            f"\"m{model_id}\"," + \
            f"{break_points_map[x]}).\n"

    spec += ":- tree_node(N), undef(N, _).\n"  # TODO
    return spec, break_points_map, break_points_invmap


# returns a model (e.g. polynomial), constraints:dict.
def synthesize_unknown(unkn_label:str, K:dataset.DataKnowledge, break_points:set):
    pulled_constrs = {}
    for derivdeg in range(3):
        pulled_constrs[derivdeg] = qp.get_constraints(K, break_points, derivdeg, SAMPLE_SIZE)
    
    poly_coeffs = qp.qp_solve(pulled_constrs, SYNTH_POLYDEG)  # in decreasing power.
    P = np.poly1d(poly_coeffs)
    P, coeffs_mask = utils.simplify_poly(P, pulled_constrs)

    return (P, coeffs_mask, pulled_constrs)


def synthesize_unknowns(stree:backprop.SyntaxTree, unknown_labels:list[str], break_points_map:dict, break_points_invmap:dict,  # invmap: from asp to float.
                        asp_model, onsynth_callback:callable):  # passing unknown_labels for efficiency
    if not asp_model.optimality_proven: return
    print(f"--- ASP Model ---\n{asp_model}")

    # build knowledge from ASP model.
    unkn_knowledge_map = {}
    for unkn in unknown_labels: unkn_knowledge_map[unkn] = dataset.DataKnowledge()

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
        
        elif atom.name == 'even_symm_unkn':
            x = break_points_invmap[ atom.arguments[1].number ]
            unkn_knowledge_map[unkn].add_symm(derivdeg, x, iseven=True)
        
        elif atom.name == 'odd_symm_unkn':
            x = break_points_invmap[ atom.arguments[1].number ]
            unkn_knowledge_map[unkn].add_symm(derivdeg, x, iseven=False)
        
        elif atom.name == 'undef_unkn':
            raise RuntimeError('undef_unkn not implemented.')  # TODO

    # synthetize each unknown model.
    synth_unkn_models = {}
    for unkn in unknown_labels:
        synth_unkn_models[unkn] = synthesize_unknown(unkn, unkn_knowledge_map[unkn], set(break_points_map.keys()))
        
    onsynth_callback(synth_unkn_models)    


def lpbackprop(K:dataset.DataKnowledge, stree:backprop.SyntaxTree, onsynth_callback:callable):
    #
    # build ASP specification of stree and stree' (facts).
    #
    stree = stree.simplify()
    stree_pr = stree.diff().simplify()
    aspSpecBuilder = ASPSpecBuilder()
    aspSpecBuilder.map_root(stree)
    aspSpecBuilder.map_root(stree_pr, 1)
    stree.accept(aspSpecBuilder)
    stree_pr.accept(aspSpecBuilder)
    stree_spec = aspSpecBuilder.spec
    print(stree_spec)

    #
    # build ASP prior knowledge (K) specification.
    #
    K_spec, break_points_map, break_points_invmap = build_knowledge_spec(K)
    print(K_spec)

    #
    # invoke ASP solver with stree_spec and K_spec (knowledge backprop).
    #
        
    # set encodings.
    clingo_ctl = clingo.Control()
    for enc in ASP_ENCODINGS: clingo_ctl.load(enc)
    clingo_ctl.add('stree_spec', [], stree_spec)  # ASP facts (stree).
    clingo_ctl.add('K_spec'    , [], K_spec    )  # ASP facts (knowledge).
    clingo_ctl.add('show'      , [], """
        #show root_unkn/2.
        #show sign_unkn/4.
        #show even_symm_unkn/2.
        #show odd_symm_unkn/2.
    """)

    # set solving options.
    clingo_ctl.configuration.solve.models = 0  # compute all models.
    clingo_ctl.configuration.solve.opt_mode = 'optN'  # find optimum, then enumerate optimal models (models=0).

    # compute models and synthesize each unknown node in stree.
    unkn_collector = backprop.UnknownSyntaxTreeCollector()
    stree.accept(unkn_collector)
    unknown_labels = unkn_collector.unknown_labels
    clingo_ctl.ground([('stree_spec', []), ('K_spec', []), ('show', []), ('base', [])])
    print(clingo_ctl.solve(on_model= \
        lambda asp_model: synthesize_unknowns(stree, unknown_labels, break_points_map, break_points_invmap, asp_model, onsynth_callback)))





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
