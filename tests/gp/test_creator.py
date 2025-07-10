from gp.creator import PTC2RandomSolutionCreator
from symbols.const import ConstantSyntaxTree
from symbols.unaop import UnaryOperatorSyntaxTree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.grammar import can_nest


def is_nonterminal(stree) -> bool:
    return type(stree) is UnaryOperatorSyntaxTree or type(stree) is BinaryOperatorSyntaxTree


def test_ptc2_random_creator():
    for nvars in [1, 2, 3, 5, 8]:
        creator = PTC2RandomSolutionCreator(nvars, simplify=False, unique=False)
        
        for max_depth in [0, 1, 2, 8, 20]:
            for max_length in [1, 2, 8, 20]:

                for create_consts in [True, False]:
                    nconsts = 0

                    for stree in creator.create_population(100, max_depth, max_length, create_consts=create_consts):
                        
                        assert stree.get_max_depth() <= max_depth
                        assert stree.get_nnodes() <= max_length
                        
                        if is_nonterminal(stree):
                            for node in stree.cache.nodes:
                                if   type(node) is UnaryOperatorSyntaxTree and is_nonterminal(node.inner):
                                    assert can_nest(node.operator, node.inner.operator)
                                elif type(node) is BinaryOperatorSyntaxTree:
                                    assert not is_nonterminal(node.left ) or can_nest(node.operator, node.left.operator )
                                    assert not is_nonterminal(node.right) or can_nest(node.operator, node.right.operator)
                        
                        if type(stree) is ConstantSyntaxTree:
                            nconsts += 1
                    
                    # we assume to find at least a constant in the population when create_consts is True.
                    assert max_length > 1 or (create_consts and nconsts > 0) or (not create_consts and nconsts == 0)
