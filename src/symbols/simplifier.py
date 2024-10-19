import symbols.unaop as unaop
import symbols.binop as binop
from symbols.const import ConstantSyntaxTree


def simplify_unary_stree(stree):

    if type(stree.inner) is ConstantSyntaxTree:
        return ConstantSyntaxTree( stree.operate(stree.inner.val) )
    
    if stree.operator == 'exp' and type(stree.inner) is unaop.UnaryOperatorSyntaxTree and stree.inner.operator == 'log':
        return stree.inner.inner
    
    if stree.operator == 'log' and type(stree.inner) is unaop.UnaryOperatorSyntaxTree and stree.inner.operator == 'exp':
        return stree.inner.inner
    
    if stree.operator == 'sqrt' and type(stree.inner) is binop.BinaryOperatorSyntaxTree and \
        stree.inner.operator == '^' and type(stree.inner.right) is ConstantSyntaxTree and stree.inner.right.val == 2:
        return stree.inner.left
    
    if stree.operator == 'sqrt' and type(stree.inner) is unaop.UnaryOperatorSyntaxTree and stree.inner.operator == 'square':
        return stree.inner.inner
    
    if stree.operator == 'square' and type(stree.inner) is unaop.UnaryOperatorSyntaxTree and stree.inner.operator == 'sqrt':
        return stree.inner.inner
    
    return stree


def simplify_binary_stree(stree):
    
    is_left_const  = type(stree.left)  is ConstantSyntaxTree
    is_right_const = type(stree.right) is ConstantSyntaxTree

    if is_left_const and is_right_const and (stree.operator != '/' or stree.right.val != 0.):
        return ConstantSyntaxTree( stree.operate(stree.left.val, stree.right.val) )

    if stree.operator == '*':
        if (is_left_const and stree.left.val == 0) or (is_right_const and stree.right.val == 0):
            return ConstantSyntaxTree(0.0)
        if (is_left_const and stree.left.val == 1):
            return stree.right
        if (is_right_const and stree.right.val == 1):
            return stree.left
        if stree.left == stree.right:
            return unaop.UnaryOperatorSyntaxTree('square', stree.left)
        if type(stree.left) is unaop.UnaryOperatorSyntaxTree and stree.left.operator == 'square' and stree.left.inner == stree.right:
            return unaop.UnaryOperatorSyntaxTree('cube', stree.right)
        if type(stree.right) is unaop.UnaryOperatorSyntaxTree and stree.right.operator == 'square' and stree.right.inner == stree.left:
            return unaop.UnaryOperatorSyntaxTree('cube', stree.left)

    if stree.operator == '/':
        if stree.left == stree.right: return ConstantSyntaxTree(1.0)  # TODO: consider x/x when x=0.
        if is_left_const and stree.left.val == 0.0: return ConstantSyntaxTree(0.0)
        if is_right_const and stree.right.val == 1.0: return stree.left

    if stree.operator == '+':
        if is_left_const  and stree.left.val  == 0: return stree.right
        if is_right_const and stree.right.val == 0: return stree.left
    if stree.operator == '-':
        if is_right_const and stree.right.val == 0: return stree.left
        if stree.left == stree.right: return ConstantSyntaxTree(0.0)

    if stree.operator == '^' and is_right_const:
        if stree.right.val == 0:
            return ConstantSyntaxTree(1.0)
        
        if stree.right.val == 1:
            return stree.left
        
        if stree.right.val == 2 and type(stree.left) is unaop.UnaryOperatorSyntaxTree and stree.left.operator == 'sqrt':
            return stree.left.inner
        
        if type(stree.right) is ConstantSyntaxTree:
            if stree.right.val == 2: return unaop.UnaryOperatorSyntaxTree('square', stree.left)
            if stree.right.val == 3: return unaop.UnaryOperatorSyntaxTree('cube', stree.left)
        
        if type(stree.left) is binop.BinaryOperatorSyntaxTree and stree.left.operator == '^' and \
            type(stree.left.right) is ConstantSyntaxTree:
            stree.right = ConstantSyntaxTree(stree.right.val * stree.left.right.val)
            stree.left = stree.left.left
            return stree

    return stree
