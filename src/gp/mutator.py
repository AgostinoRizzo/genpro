import random
from symbols.syntax_tree import SyntaxTree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.unaop import UnaryOperatorSyntaxTree
from symbols.visitor import SyntaxTreeNodeCounter, SyntaxTreeNodeSelector, ConstantSyntaxTreeCollector
from symbols.grammar import get_nesting_operators, get_una_nesting_operators, get_bin_nesting_operators
from gp import utils, gp


class Mutator:
    def mutate(self, stree:SyntaxTree) -> SyntaxTree:
        pass

class MultiMutator(Mutator):
    def __init__(self, *mutators):
        self.mutators = list(mutators)
    
    def mutate(self, stree:SyntaxTree) -> SyntaxTree:
        return random.choice(self.mutators).mutate(stree)

class SubtreeReplacerMutator(Mutator):
    def __init__(self, max_depth, max_length, solutionCreator):
        self.max_depth = max_depth
        self.max_length = max_length
        self.solutionCreator = solutionCreator
    
    def mutate(self, stree:SyntaxTree) -> SyntaxTree:
        nodesCounter = SyntaxTreeNodeCounter()
        stree.accept(nodesCounter)
        nnodes = nodesCounter.nnodes

        nodeSelector = SyntaxTreeNodeSelector(random.randrange(nnodes))
        stree.accept(nodeSelector)
        sub_stree = nodeSelector.node

        # replace subtree with random branch.

        stree.set_parent()
        sub_stree_depth = sub_stree.get_depth()
        
        new_substree_max_depth = self.max_depth - sub_stree_depth
        new_substree_max_length = self.max_length - (stree.get_nnodes() - sub_stree.get_nnodes())
        assert new_substree_max_depth >= 0 and new_substree_max_length > 0

        parent_opt = None if not sub_stree.has_parent() else sub_stree.parent.operator
        new_sub_stree = self.solutionCreator.create_population(1, new_substree_max_depth, new_substree_max_length, parent_opt=parent_opt)[0]
        stree = utils.replace_subtree(stree, sub_stree, new_sub_stree)

        stree.set_parent()
        if new_sub_stree.has_parent():
            new_sub_stree.parent.invalidate_output()
        
        return stree.simplify()
    
class FunctionSymbolMutator(Mutator):
    def mutate(self, stree:SyntaxTree) -> SyntaxTree:
        nodesCounter = SyntaxTreeNodeCounter()
        stree.accept(nodesCounter)
        nnodes = nodesCounter.nnodes

        nodeSelector = SyntaxTreeNodeSelector(random.randrange(nnodes))
        stree.accept(nodeSelector)
        sub_stree = nodeSelector.node

        stree.set_parent()
  
        # change a single function symbol.

        if   type(sub_stree) is BinaryOperatorSyntaxTree:
            sub_stree.operator = random.choice(
                [opt for opt in get_bin_nesting_operators(sub_stree.operator) if opt != sub_stree.operator])
            
            sub_stree.invalidate_output()

        elif type(sub_stree) is UnaryOperatorSyntaxTree:
            sub_stree.operator = random.choice(
                [opt for opt in get_una_nesting_operators(sub_stree.operator) if opt != sub_stree.operator])

            sub_stree.invalidate_output()

        return stree

class NumericParameterMutator(Mutator):
    def __init__(self, all:bool=True):
        self.all = all
    
    def mutate(self, stree:SyntaxTree) -> SyntaxTree:
        constsCollector = ConstantSyntaxTreeCollector()
        stree.accept(constsCollector)

        stree.set_parent()

        if len(constsCollector.constants) == 0: return stree
        constsToMutate = constsCollector.constants if self.all else [random.choice(constsCollector.constants)]
        for c in constsToMutate:
            c.val += random.gauss(mu=0.0, sigma=1.0)
            c.invalidate_output()

        return stree