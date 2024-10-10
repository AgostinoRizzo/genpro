import numpy as np
import random
import string

from symbols.syntax_tree import SyntaxTree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.unaop import UnaryOperatorSyntaxTree
from symbols.const import ConstantSyntaxTree
from symbols.var   import VariableSyntaxTree
from symbols.misc  import UnknownSyntaxTree


class SyntaxTreeGenerator:
    OPERATORS = BinaryOperatorSyntaxTree.OPERATORS + UnaryOperatorSyntaxTree.OPERATORS

    def __init__(self, nvars:int=1, y_iqr:float=1.0):
        self.unkn_counter = 0
        self.nvars = nvars
        self.y_iqr = y_iqr
    
    def create_random(self, max_depth:int, n:int=1,
                      check_duplicates:bool=True, noconsts:bool=False, leaf_types:list[str]=['const', 'var']) -> list[SyntaxTree]:
        assert max_depth >= 0 and n >= 0
        strees = []
        for _ in range(n):
            self.unkn_counter = 0
            new_stree = self.__create_random(random.randint(0, max_depth), leaf_types)
            while not new_stree.validate() or (check_duplicates and new_stree in strees) or (noconsts and type(new_stree) is ConstantSyntaxTree):
                self.unkn_counter = 0
                new_stree = self.__create_random(random.randint(0, max_depth), leaf_types)
            strees.append(new_stree)
        return strees

    def __create_random(self, depth:int, leaf_types:list[str]):
        assert self.unkn_counter < len(string.ascii_uppercase)
        if depth <= 0:
            leaf_t = random.choice(leaf_types)
            if leaf_t == 'const':
                stree = ConstantSyntaxTree(val=random.uniform(-self.y_iqr, self.y_iqr))
                if np.isnan(stree.val): raise RuntimeError('NaN in random tree generation.')
            elif leaf_t == 'var':
                stree = VariableSyntaxTree(idx=random.randrange(self.nvars))
            elif leaf_t == 'unknown':
                stree = UnknownSyntaxTree(string.ascii_uppercase[self.unkn_counter], nvars=self.nvars)
                self.unkn_counter += 1
            else:
                raise RuntimeError('Cannot generator leaf node.')
            return stree
        
        operator = random.choice(SyntaxTreeGenerator.OPERATORS)
        
        stree = None
        if operator in UnaryOperatorSyntaxTree.OPERATORS:
            stree = UnaryOperatorSyntaxTree(operator, self.__create_random(depth - 1, leaf_types))

        else:
            stree = BinaryOperatorSyntaxTree(operator,
                        self.__create_random(depth - 1, leaf_types),
                        self.__create_random(depth - 1, leaf_types) if operator != '^' else ConstantSyntaxTree(random.choice([2.0,3.0,4.0])))  # TODO: only ^2 managed.
        
        return stree.simplify()
