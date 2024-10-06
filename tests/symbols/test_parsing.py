import pytest
from symbols.parsing import parse_syntax_tree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.unaop import UnaryOperatorSyntaxTree
from symbols.const import ConstantSyntaxTree
from symbols.var import VariableSyntaxTree


@pytest.mark.parametrize("expr,expected_stree",
[
    (
    '((-1.79 / x0) * exp((log(x0) - square((x0 + x0)))))',
    BinaryOperatorSyntaxTree('*',
        BinaryOperatorSyntaxTree('/',
            ConstantSyntaxTree(-1.79),
            VariableSyntaxTree()
        ),

        UnaryOperatorSyntaxTree('exp',
            BinaryOperatorSyntaxTree('-',
                UnaryOperatorSyntaxTree('log',
                    VariableSyntaxTree()
                ),
                UnaryOperatorSyntaxTree('square',
                    BinaryOperatorSyntaxTree('+',
                        VariableSyntaxTree(),
                        VariableSyntaxTree()
                    )
                )
            )
        )
    )
    ),

    ('log(x0)', UnaryOperatorSyntaxTree('log', VariableSyntaxTree())),

    ('(-1.2 / 4.1)',
        BinaryOperatorSyntaxTree('/',
            ConstantSyntaxTree(-1.2), ConstantSyntaxTree(4.1))),

    ('x2', VariableSyntaxTree(2)),
    ('-1.2', ConstantSyntaxTree(-1.2)),
    ('unaopt(x0)', None),
    ('1.2 @ x0', None),
    ('@!  2', None),
    ('x2.0', None),
    ('', None),
    ('   ', None)
])
def test_parse_syntax_tree(expr, expected_stree):
    
    if expected_stree is None:
        with pytest.raises(Exception):
            parse_syntax_tree(expr)
    
    else:
        actual_stree = parse_syntax_tree(expr)
        assert actual_stree == expected_stree