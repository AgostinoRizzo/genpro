from backprop import backprop


class SyntaxTreeInfo:
    X_data = None

    @staticmethod
    def set_problem(data):
        SyntaxTreeInfo.X_data = data.X

    def __init__(self, stree):
        self.stree = stree
        
        self.__nnodes = None
        self.__nodes = None
        self.__backprop_nodes = None
        self.__y = None

    @property
    def nnodes(self):
        if self.__nnodes is None:
            self.__nnodes = self.stree.get_nnodes()
        return self.__nnodes
        
    @property
    def nodes(self):
        if self.__nodes is None:
            nodesCollector = backprop.SyntaxTreeNodeCollector()
            self.stree.accept(nodesCollector)

            self.__nodes = nodesCollector.nodes
            self.__nnodes = len(self.__nodes)
        return self.__nodes

    @property
    def backprop_nodes(self):
        if self.__backprop_nodes is None:
            self.__backprop_nodes = []
            for n in self.nodes:
                if backprop.SyntaxTree.is_invertible_path(n):
                    self.__backprop_nodes.append(n)
        return self.__backprop_nodes

    @property
    def y(self):
        if self.__y is None:
            self.__y = self.stree(SyntaxTreeInfo.X_data)
        return self.__y
    
    def clear(self):
        self.__nnodes = None
        self.__nodes = None
        self.__backprop_nodes = None
        self.__y = None