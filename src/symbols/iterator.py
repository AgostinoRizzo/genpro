from collections import deque

from symbols.visitor import SyntaxTreeVisitor
from symbols.syntax_tree import SyntaxTree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.unaop import UnaryOperatorSyntaxTree
from symbols.const import ConstantSyntaxTree
from symbols.var   import VariableSyntaxTree
from symbols.misc  import FunctionSyntaxTree
from symbols.misc  import UnknownSyntaxTree
from symbols.misc  import SemanticSyntaxTree


class SyntaxTreeIterator(SyntaxTreeVisitor):
    def __init__(self, syntree:SyntaxTree):
        self.syntree = syntree
        self.queue = None
    
    def __iter__(self):
        self.queue = deque()
        self.syntree.accept(self)
        return self

    def __next__(self):
        if len(self.queue) == 0:
            raise StopIteration()
        return self.queue.popleft()
    
    def visitConstant(self, syntree:ConstantSyntaxTree): self.queue.append(syntree)
    def visitVariable(self, syntree:VariableSyntaxTree): self.queue.append(syntree)
    def visitFunction(self, syntree:FunctionSyntaxTree): self.queue.append(syntree)
    def visitUnknown (self, syntree:UnknownSyntaxTree):  self.queue.append(syntree)
    def visitSemantic(self, syntree:SemanticSyntaxTree): self.queue.append(syntree)


class InorderIterator(SyntaxTreeIterator):
    """
    Nodes are visited in the order: left -> root -> right;
    In case of unary node, the order is: root -> inner.
    """
    def __init__(self, syntree:SyntaxTree):
        super().__init__(syntree)
    
    def visitUnaryOperator (self, syntree:UnaryOperatorSyntaxTree):
        self.queue.append(syntree)
        syntree.inner.accept(self)
    
    def visitBinaryOperator(self, syntree:BinaryOperatorSyntaxTree):
        syntree.left.accept(self)
        self.queue.append(syntree)
        syntree.right.accept(self)


class PreorderIterator(SyntaxTreeIterator):
    """
    Nodes are visited in the order: root -> left -> right;
    In case of unary node, the order is: root -> inner.
    """
    def __init__(self, syntree:SyntaxTree):
        super().__init__(syntree)
    
    def visitUnaryOperator (self, syntree:UnaryOperatorSyntaxTree):
        self.queue.append(syntree)
        syntree.inner.accept(self)
    
    def visitBinaryOperator(self, syntree:BinaryOperatorSyntaxTree):
        self.queue.append(syntree)
        syntree.left.accept(self)
        syntree.right.accept(self)


class PostorderIterator(SyntaxTreeIterator):
    """
    Nodes are visited in the order: left -> right -> root;
    In case of unary node, the order is: inner -> root.
    """
    def __init__(self, syntree:SyntaxTree):
        super().__init__(syntree)
    
    def visitUnaryOperator (self, syntree:UnaryOperatorSyntaxTree):
        syntree.inner.accept(self)
        self.queue.append(syntree)
    
    def visitBinaryOperator(self, syntree:BinaryOperatorSyntaxTree):
        syntree.left.accept(self)
        syntree.right.accept(self)
        self.queue.append(syntree)


class SingleNodeIterator(SyntaxTreeIterator):
    """
    Only the root node is visited.
    """
    def __init__(self, syntree:SyntaxTree):
        super().__init__(syntree)
    
    def visitUnaryOperator (self, syntree:UnaryOperatorSyntaxTree):
        self.queue.append(syntree)
    
    def visitBinaryOperator(self, syntree:BinaryOperatorSyntaxTree):
        self.queue.append(syntree)


class DefaultIterator(PreorderIterator):
    pass
