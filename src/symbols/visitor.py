from symbols.binop import BinaryOperatorSyntaxTree
from symbols.unaop import UnaryOperatorSyntaxTree
from symbols.const import ConstantSyntaxTree
from symbols.var   import VariableSyntaxTree
from symbols.misc  import FunctionSyntaxTree
from symbols.misc  import UnknownSyntaxTree
from symbols.misc  import SemanticSyntaxTree


class SyntaxTreeVisitor:
    def visitUnaryOperator (self, stree:UnaryOperatorSyntaxTree):  pass
    def visitBinaryOperator(self, stree:BinaryOperatorSyntaxTree): pass
    def visitConstant      (self, stree:ConstantSyntaxTree):       pass
    def visitVariable      (self, stree:VariableSyntaxTree):       pass
    def visitFunction      (self, stree:FunctionSyntaxTree):       pass
    def visitUnknown       (self, stree:UnknownSyntaxTree):        pass
    def visitSemantic      (self, stree:SemanticSyntaxTree):       pass


class SyntaxTreeNodeCollector(SyntaxTreeVisitor):
    def __init__(self):
        self.nodes = []
    def visitUnaryOperator (self, stree:UnaryOperatorSyntaxTree):  self.nodes.append(stree)
    def visitBinaryOperator(self, stree:BinaryOperatorSyntaxTree): self.nodes.append(stree)
    def visitConstant      (self, stree:ConstantSyntaxTree):       self.nodes.append(stree)
    def visitVariable      (self, stree:VariableSyntaxTree):       self.nodes.append(stree)
    def visitFunction      (self, stree:FunctionSyntaxTree):       self.nodes.append(stree)
    def visitUnknown       (self, stree:UnknownSyntaxTree):        self.nodes.append(stree)
    def visitSemantic      (self, stree:SemanticSyntaxTree):       self.nodes.append(stree)

class UnknownSyntaxTreeCollector(SyntaxTreeVisitor):
    def __init__(self):
        self.unknown_labels = set()
        self.unknowns = []
    def visitUnknown(self, stree:UnknownSyntaxTree):
        self.unknown_labels.add(stree.label)
        self.unknowns.append(stree)

class ConstantSyntaxTreeCollector(SyntaxTreeVisitor):
    def __init__(self):
        self.constants = []
    def visitConstant(self, stree:ConstantSyntaxTree):
        self.constants.append(stree)


class SyntaxTreeNodeCounter(SyntaxTreeVisitor):
    def __init__(self):
        self.nnodes = 0
    def visitUnaryOperator (self, stree:UnaryOperatorSyntaxTree):  self.nnodes += 1
    def visitBinaryOperator(self, stree:BinaryOperatorSyntaxTree): self.nnodes += 1
    def visitConstant      (self, stree:ConstantSyntaxTree):       self.nnodes += 1
    def visitVariable      (self, stree:VariableSyntaxTree):       self.nnodes += 1
    def visitFunction      (self, stree:FunctionSyntaxTree):       self.nnodes += 1
    def visitUnknown       (self, stree:UnknownSyntaxTree):        self.nnodes += 1
    def visitSemantic      (self, stree:SemanticSyntaxTree):       self.nnodes += 1


class SyntaxTreeOperatorCollector(SyntaxTreeVisitor):
    def __init__(self):
        self.opts = set()
    def visitUnaryOperator (self, stree:UnaryOperatorSyntaxTree):  self.opts.add(stree.operator)
    def visitBinaryOperator(self, stree:BinaryOperatorSyntaxTree): self.opts.add(stree.operator)

class SyntaxTreeIneqOperatorCollector(SyntaxTreeVisitor):
    def __init__(self):
        self.opts = set()
    
    def visitUnaryOperator (self, stree:UnaryOperatorSyntaxTree):
        self.opts.add(stree.operator)
    
    def visitBinaryOperator(self, stree:BinaryOperatorSyntaxTree):
        if stree.operator in ['+', '-', '*'] and \
           (type(stree.left) is ConstantSyntaxTree or type(stree.right) is ConstantSyntaxTree):
            return
        
        if stree.operator == '/' and type(stree.right) is ConstantSyntaxTree:
            return

        if stree.operator == '-': self.opts.add('+')
        else: self.opts.add(stree.operator)


class SyntaxTreeNodeSelector(SyntaxTreeVisitor):
    def __init__(self, ith:int):
        self.ith = ith
        self.i = 0
        self.node = None
    def visitUnaryOperator(self, stree:UnaryOperatorSyntaxTree):
        if self.i == self.ith: self.node = stree
        self.i += 1
    def visitBinaryOperator(self, stree:BinaryOperatorSyntaxTree):
        if self.i == self.ith: self.node = stree
        self.i += 1
    def visitConstant(self, stree:ConstantSyntaxTree):
        if self.i == self.ith: self.node = stree
        self.i += 1
    def visitVariable(self, stree:VariableSyntaxTree):
        if self.i == self.ith: self.node = stree
        self.i += 1
    def visitFunction(self, stree:FunctionSyntaxTree):
        if self.i == self.ith: self.node = stree
        self.i += 1
    def visitUnknown(self, stree:UnknownSyntaxTree):
        if self.i == self.ith: self.node = stree
        self.i += 1
    def visitSemantic(self, stree:SemanticSyntaxTree):
        if self.i == self.ith: self.node = stree
        self.i += 1