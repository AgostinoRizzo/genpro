import pytest
import math
from symbols.parsing import parse_syntax_tree


@pytest.mark.parametrize("expr,simple_expr",
[
    # None means simple_expr coincides with expr (no simplification is applied).
    ('(x0 * x0)', 'square(x0)'),
    ('(x0 * (x0 * x0))', 'cube(x0)'), ('(x0 * (x1 * x1))', '(x0 * square(x1))'), ('(x0 * (x1 * x2))', None),
    ('((x0 * x0) * x0)', 'cube(x0)'), ('((x1 * x1) * x0)', '(square(x1) * x0)'), ('((x2 * x1) * x0)', None),

    ('cube(exp((0.0*x0)))', '1.0'),
    ('square(exp((0.3 * 0.1)))', str(math.exp(0.3*0.1)** 2)),
    ('cube(log((0.0*x0)))', 'ninf'),
    ('cube(log(-1.0))', 'nan'),

    ('(x0 / x0)', '1.0'),  # TODO: consider x/x when x=0.
    ('(x0 / 1.0)', 'x0'),
    ('(0.0/x0)', '0.0'),
    
    ('(x0 + 0.0)', 'x0'),
    ('(0.0 + square(x0))', 'square(x0)'),

    ('(x0 * 1.0)', 'x0'),
    ('(1.0 * square(x0))', 'square(x0)'),
    ('(x0 * 0.0)', '0.0'),
    ('(0.0 * square(x0))', '0.0'),

    ('log(sqrt(exp(sqrt((x1 * x0)))))', '(0.5 * sqrt((x1 * x0)))'),
    ('(0.11 * (0.11 * (x0 * x1)))', '(0.0121 * (x0 * x1))'),
])
def test_simplifier(expr, simple_expr):
    stree = parse_syntax_tree(expr)
    simple_stree = stree.clone() if simple_expr is None else parse_syntax_tree(simple_expr)
    assert stree.simplify() == simple_stree