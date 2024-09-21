import random
from backprop import library
from gp import gp


class Corrector:
    def __init__(self, S_data, S_know):
        self.S_data = S_data
        self.S_know = S_know

        X_mesh = S_data.spsampler.meshspace(S_data.xl, S_data.xu, 100)
        self.lib = library.ConstrainedLibrary(2000, 3, S_data, X_mesh)
    
    def correct(self, stree):
        for _ in range(1):
            stree.cache.clear()
            stree.set_parent()

            backprop_nodes = stree.cache.backprop_nodes
            if len(backprop_nodes) == 0:
                return stree
            
            backprop_node = random.choice(backprop_nodes)
            
            # backprop knowledge...
            stree.clear_output()
            stree[(self.S_know.X, ())]  # needed for 'pull_know'.
            k_pulled, noroot_pulled = backprop_node.pull_know(self.S_know.y)
            if k_pulled is None or noroot_pulled is None:
                return stree
            K_pulled = (k_pulled.tobytes(), noroot_pulled)

            # backprop data...
            y = stree(self.S_data.X)  # needed for 'pull_output'.
            y_pulled = backprop_node.pull_output(self.S_data.y)

            y_backprop_node = backprop_node(self.S_data.X)
            max_dist = library.compute_distance(y_backprop_node, y_pulled)
            new_node = self.lib.cquery(y_pulled, K_pulled, max_dist=max_dist)
            if new_node is None:
                return stree

            # correct stree...
            new_stree = gp.replace_subtree(stree, backprop_node, new_node)
            if new_stree.get_max_depth() > 5:  # TODO: lookup based on max admissible depth.
                gp.replace_subtree(new_stree, new_node, backprop_node)
                continue
            
            new_stree.cache.clear()
            new_stree.set_parent()
            if new_node.has_parent():
                new_node.parent.invalidate_output()
            
            stree = new_stree

        return new_stree
        