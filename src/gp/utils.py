from functools import cmp_to_key

from symbols.syntax_tree import SyntaxTree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.unaop import UnaryOperatorSyntaxTree


def sort_population(population:list[SyntaxTree], eval_map:dict) -> list[SyntaxTree]:
    def strees_cmp(stree1, stree2) -> int:
        nonlocal eval_map
        stree1_eval = eval_map[id(stree1)]
        stree2_eval = eval_map[id(stree2)]
        if stree1_eval.better_than(stree2_eval): return -1
        if stree2_eval.better_than(stree1_eval): return  1
        return 0
    return sorted(population, key=cmp_to_key(strees_cmp))


def replace_subtree(stree:SyntaxTree,
                    sub_stree:SyntaxTree,
                    new_sub_stree:SyntaxTree) -> SyntaxTree:
    stree.set_parent()

    if sub_stree.parent is None:
        return new_sub_stree
    if type(sub_stree.parent) is BinaryOperatorSyntaxTree:
        if   id(sub_stree) == id(sub_stree.parent.left) : sub_stree.parent.left  = new_sub_stree
        elif id(sub_stree) == id(sub_stree.parent.right): sub_stree.parent.right = new_sub_stree
    elif type(sub_stree.parent) is UnaryOperatorSyntaxTree:
        if id(sub_stree) == id(sub_stree.parent.inner): sub_stree.parent.inner = new_sub_stree
    
    return stree


def generate_trunks(max_depth:int, nvars:int, knowledge):
    from backprop import lpbackprop
    from backprop import xgp
    solutionCreator = xgp.RandomTemplateSolutionCreator(nvars=nvars)
    all_trunks = []
    satunsat_trunks = {'sat': [], 'unsat': []}

    for _ in range(100):
        trunk = solutionCreator.create_population(1, max_depth=max_depth)[0]
        if type(trunk) is UnknownSyntaxTree or \
           trunk in all_trunks or \
           check_unsat_trunk(satunsat_trunks, trunk): continue
        all_trunks.append(trunk)
        #print(f"Checking trunk: {trunk}")
        sat, _ = lpbackprop.lpbackprop(knowledge, trunk, None)
        if sat:
            satunsat_trunks['sat'].append(trunk)
            #print(f"SAT  : {trunk}")
        else:
            satunsat_trunks['unsat'].append(trunk)
            #print(f"UNSAT: {trunk}")
    return satunsat_trunks

def check_unsat_trunk(trunks:map, stree) -> bool:
    for unsat_trunk in trunks['unsat']:
        """if type(stree) is BinaryOperatorSyntaxTree and stree.operator == '*' and \
            type(stree.left) is backprop.VariableSyntaxTree and type(stree.right) is backprop.ConstantSyntaxTree and \
                type(unsat_trunk) is BinaryOperatorSyntaxTree and \
                type(unsat_trunk.left) is UnknownSyntaxTree and type(unsat_trunk.right) is UnknownSyntaxTree:
                    print()"""
        if stree.match(unsat_trunk):
            return True
    return False