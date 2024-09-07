import random
from backprop import backprop, gp, pareto_front


class SymbolicDiversifier:
    def __init__(self, data, lib):
        self.data = data
        self.lib = lib
    
    def diversify(self, stree):
        if stree is None: return None
        nodesCollector = backprop.SyntaxTreeNodeCollector()
        stree.accept(nodesCollector)
        node = random.choice(nodesCollector.nodes)

        node_sem = node(self.data.X)
        new_node = self.lib.query(node_sem)

        if new_node is None: return stree
        
        nnodes = node.get_nnodes()
        new_nnodes = new_node.get_nnodes()
        if new_nnodes > nnodes: return stree

        symbset = pareto_front.SymbolicFrequencies.get_symbset(node)
        new_symbset = pareto_front.SymbolicFrequencies.get_symbset(new_node)
        if new_nnodes == nnodes and new_symbset == symbset:
            return stree

        stree = gp.replace_subtree(stree, node, new_node)
        stree.set_parent()
        if new_node.has_parent():
            new_node.parent.invalidate_output()
        return stree