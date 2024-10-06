import lrparsing
from lrparsing import Ref, Token

from symbols.syntax_tree import SyntaxTree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.unaop import UnaryOperatorSyntaxTree
from symbols.const import ConstantSyntaxTree
from symbols.var import VariableSyntaxTree


class SyntaxTreeParser(lrparsing.Grammar):
    
    #
    # tokens
    #
    class T(lrparsing.TokenRegistry):
        binopt_re = '\\' + '|\\'.join(BinaryOperatorSyntaxTree.OPERATORS)
        unaopt_re = '|'.join(UnaryOperatorSyntaxTree.OPERATORS)

        constant = Token(re="[\+\-]?[0-9]+\.[0-9]+")
        variable = Token(re="x[0-9]+")
        binopt_iden = Token(re=binopt_re)
        unaopt_iden = Token(re=unaopt_re)
    
    #
    # grammar rules.
    #
    expr = Ref("expr")
    binopt = '(' + expr + T.binopt_iden + expr + ')'
    unaopt = T.unaopt_iden + '(' + expr + ')'
    expr = T.constant | T.variable | binopt | unaopt
    START = expr


def build_expr(parse_tree) -> SyntaxTree:
    rule = parse_tree[0]
    symbols = parse_tree[1:]

    if rule.name == 'expr':
        return build_expr(symbols[0])
    
    if rule.name == 'binopt':
        opt   = symbols[2][1]
        left  = symbols[1]
        right = symbols[3]
        return BinaryOperatorSyntaxTree(opt, build_expr(left), build_expr(right))
    
    if rule.name == 'unaopt':
        opt = symbols[0][1]
        inner = symbols[2]
        return UnaryOperatorSyntaxTree(opt, build_expr(inner))

    if rule.name == 'T.variable':
        varidx = int(symbols[0][1:])
        return VariableSyntaxTree(varidx)
    
    if rule.name == 'T.constant':
        val = float(symbols[0])
        return ConstantSyntaxTree(val)
    
    raise RuntimeError('Invalid rule.')


def parse_syntax_tree(expr:str) -> SyntaxTree:
    parse_tree = SyntaxTreeParser.parse(expr)
    stree = build_expr(parse_tree[1])
    return build_expr(parse_tree[1])