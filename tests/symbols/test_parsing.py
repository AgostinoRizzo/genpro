import pytest
from symbols.parsing import parse_syntax_tree
from backprop import backprop


@pytest.mark.parametrize("expr,expected_stree",
[
    (
    '((-1.79 / x0) * exp((log(x0) - square((x0 + x0)))))',
    backprop.BinaryOperatorSyntaxTree('*',
        backprop.BinaryOperatorSyntaxTree('/',
            backprop.ConstantSyntaxTree(-1.79),
            backprop.VariableSyntaxTree()
        ),

        backprop.UnaryOperatorSyntaxTree('exp',
            backprop.BinaryOperatorSyntaxTree('-',
                backprop.UnaryOperatorSyntaxTree('log',
                    backprop.VariableSyntaxTree()
                ),
                backprop.UnaryOperatorSyntaxTree('square',
                    backprop.BinaryOperatorSyntaxTree('+',
                        backprop.VariableSyntaxTree(),
                        backprop.VariableSyntaxTree()
                    )
                )
            )
        )
    )
    ),

    ('log(x0)', backprop.UnaryOperatorSyntaxTree('log', backprop.VariableSyntaxTree())),

    ('(-1.2 / 4.1)',
        backprop.BinaryOperatorSyntaxTree('/',
            backprop.ConstantSyntaxTree(-1.2), backprop.ConstantSyntaxTree(4.1))),

    ('x2', backprop.VariableSyntaxTree(2)),
    ('-1.2', backprop.ConstantSyntaxTree(-1.2)),
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