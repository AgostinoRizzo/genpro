import random
from backprop import library, constraints
from backprop.bperrors import KnowBackpropError
from gp import utils


class Corrector:
    def __init__(self, S_data, know, max_depth:int, X_mesh, libsize:int, lib_maxdepth:int):
        self.S_data = S_data
        self.max_depth = max_depth
        
        self.S_know = know.synth_dataset()
        self.S_know_derivs = {}
        for deriv in know.sign.keys():
            if len(deriv) == 1:
                self.S_know_derivs[deriv] = know.synth_dataset(deriv=deriv)

        derivs = [()] + list(self.S_know_derivs.keys())
        self.lib = library.HierarchicalConstrainedLibrary(libsize, lib_maxdepth, S_data, know, X_mesh, derivs)
    
    def correct(self, stree, backprop_node=None):
        for _ in range(1):
            stree.cache.clear()
            stree.set_parent()

            if backprop_node is None:
                backprop_nodes = stree.cache.backprop_nodes
                if len(backprop_nodes) == 0:
                    return stree, None, None, None
                
                backprop_node = random.choice(backprop_nodes)
            max_nesting_depth = self.max_depth - backprop_node.get_depth()
            
            # backprop knowledge...
            C_pulled = self.backprop_know(stree, backprop_node, max_nesting_depth)
            if C_pulled is None:
                return stree, None, None, None

            # backprop data...
            y = stree(self.S_data.X)  # needed for 'pull_output'.
            y_pulled = backprop_node.pull_output(self.S_data.y)

            y_backprop_node = backprop_node(self.S_data.X)
            max_dist = library.compute_distance(y_backprop_node, y_pulled)
            new_node = self.lib.cquery(y_pulled, C_pulled, max_dist=max_dist)
            if new_node is None:
                return stree, new_node, C_pulled, y_pulled

            # correct stree...
            new_stree = utils.replace_subtree(stree, backprop_node, new_node)
            new_stree.cache.clear()
            new_stree.set_parent()
            if new_node.has_parent():
                new_node.parent.invalidate_output()
            
            stree = new_stree

        return new_stree, new_node, C_pulled, y_pulled
    
    def backprop_know(self, stree, backprop_node, max_nesting_depth) -> constraints.BackpropConstraints:
        k_pulled = {}
        stree.clear_output()
        
        try:
            # backprop image knowledge.
            stree[(self.S_know.X, ())]  # needed for 'pull_know'.
            image_track = {}
            k_image_pulled, noroot_pulled = backprop_node.pull_know(self.S_know.y, track=image_track)
            k_pulled[()] = k_image_pulled

            # backprop derivative knowledge.
            for deriv, S_know_deriv in self.S_know_derivs.items():
                stree[(S_know_deriv.X, deriv)]  # needed for 'pull_know_deriv'.
                k_deriv_pulled = backprop_node.pull_know_deriv(image_track, deriv[0], S_know_deriv.y)
                k_pulled[deriv] = k_deriv_pulled
        
        except KnowBackpropError:
            return None

        return constraints.BackpropConstraints(max_nesting_depth, k_pulled, noroot_pulled)
        