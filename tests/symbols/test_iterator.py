import pytest

from symbols.parsing import parse_syntax_tree
from symbols.iterator import InorderIterator, PreorderIterator, PostorderIterator, SingleNodeIterator, DefaultIterator
from symbols.binop import BinaryOperatorSyntaxTree as Binop
from symbols.unaop import UnaryOperatorSyntaxTree as Unop
from symbols.const import ConstantSyntaxTree as Const
from symbols.var   import VariableSyntaxTree as Var


@pytest.mark.parametrize("expr, iterator_class, node_order",
[
    ('((2.1 + x0) * exp((x2 - sqrt(x3))))', InorderIterator,    [Const, Binop, Var, Binop, Unop, Var, Binop, Unop, Var]),
    ('((2.1 + x0) * exp((x2 - sqrt(x3))))', PreorderIterator,   [Binop, Binop, Const, Var, Unop, Binop, Var, Unop, Var]),
    ('((2.1 + x0) * exp((x2 - sqrt(x3))))', PostorderIterator,  [Const, Var, Binop, Var, Var, Unop, Binop, Unop, Binop]),
    ('((2.1 + x0) * exp((x2 - sqrt(x3))))', SingleNodeIterator, [Binop]),

    ('(x0 + 1.2)', InorderIterator,    [Var, Binop, Const]),
    ('(x0 + 1.2)', PreorderIterator,   [Binop, Var, Const]),
    ('(x0 + 1.2)', PostorderIterator,  [Var, Const, Binop]),
    ('(x0 + 1.2)', SingleNodeIterator, [Binop]),

    ('x0', InorderIterator,    [Var]),
    ('x0', PreorderIterator,   [Var]),
    ('x0', PostorderIterator,  [Var]),
    ('x0', SingleNodeIterator, [Var])
])
def test_iterator(expr, iterator_class, node_order):
    syntree = parse_syntax_tree(expr)
    
    for i, node in enumerate(iterator_class(syntree)):
        assert type(node) is node_order[i]
    
    if issubclass(DefaultIterator, iterator_class):
        for i, node in enumerate(syntree):
            assert type(node) is node_order[i]