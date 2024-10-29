import random
import numpy as np
from backprop import library, constraints
from backprop import bperrors
from gp import utils
from symbols.unaop import UnaryOperatorSyntaxTree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.grammar import can_nest


class Corrector:
    def __init__(self, S_data, know, max_depth:int, max_length:int, mesh, libsize:int, lib_maxdepth:int, lib_maxlength:int, solutionCreator):
        self.S_data = S_data
        self.max_depth = max_depth
        self.max_length = max_length
        self.mesh = mesh
        
        self.S_know = know.synth_dataset(mesh.X)
        self.S_know_derivs = {}
        for deriv in know.sign.keys():
            if len(deriv) == 1:
                self.S_know_derivs[deriv] = know.synth_dataset(mesh.X, deriv=deriv)

        derivs = [()] + list(self.S_know_derivs.keys())
        if know.has_symmvars():
            self.symm_lib  = library.HierarchicalConstrainedLibrary(libsize, lib_maxdepth, lib_maxlength, S_data, know, mesh, derivs, solutionCreator, True)
            self.asymm_lib = library.HierarchicalConstrainedLibrary(libsize, lib_maxdepth, lib_maxlength, S_data, know, mesh, derivs, solutionCreator, False)
        self.lib = library.HierarchicalConstrainedLibrary(libsize, lib_maxdepth, lib_maxlength, S_data, know, mesh, derivs, solutionCreator)
    
    def correct(self, stree, backprop_node=None, relax:bool=False):
        for _ in range(1):
            stree.cache.clear()
            stree.set_parent()

            if backprop_node is None:
                backprop_nodes = stree.cache.backprop_nodes
                if len(backprop_nodes) == 0:
                    raise bperrors.NoBackpropPathError()
                backprop_node = random.choice(backprop_nodes)
            
            max_nesting_depth = self.max_depth - backprop_node.get_depth()
            max_nesting_length = self.max_length - (stree.get_nnodes() - backprop_node.get_nnodes())
            
            # backprop knowledge...
            C_pulled = None
            if relax:
                C_pulled = self.__get_relaxed_constraints(max_nesting_depth, max_nesting_length)
            else:
                C_pulled = self.backprop_know(stree, backprop_node, max_nesting_depth, max_nesting_length)

            # backprop data...
            y = stree(self.S_data.X)  # needed for 'pull_output'.
            y_pulled = backprop_node.pull_output(self.S_data.y)
            y_pulled_origin = np.copy(y_pulled)
            C_pulled.project(y_pulled)

            y_backprop_node = backprop_node(self.S_data.X)
            max_dist = library.compute_distance(y_backprop_node, y_pulled)
            new_node = None
            if C_pulled.symm is None: new_node = self.lib.      cquery(y_pulled, C_pulled, max_dist=max_dist)
            elif C_pulled.symm[0]:    new_node = self.symm_lib. cquery(y_pulled, C_pulled, max_dist=max_dist)
            else:                     new_node = self.asymm_lib.cquery(y_pulled, C_pulled, max_dist=max_dist)
            
            if new_node.get_nnodes() > max_nesting_length:
                raise bperrors.BackpropMaxLengthError()
            
            parent_opt = None if not backprop_node.has_parent() else backprop_node.parent.operator
            if (type(new_node) is UnaryOperatorSyntaxTree or type(new_node) is BinaryOperatorSyntaxTree) and \
               not can_nest(parent_opt, new_node.operator):
                raise bperrors.BackpropGrammarError()

            # correct stree...
            new_stree = utils.replace_subtree(stree, backprop_node, new_node)
            new_stree.cache.clear()
            new_stree.set_parent()
            if new_node.has_parent():
                new_node.parent.invalidate_output()
            
            stree = new_stree

        return new_stree, new_node, C_pulled, y_pulled_origin
    
    def backprop_know(self, stree, backprop_node, max_nesting_depth, max_nesting_length) -> constraints.BackpropConstraints:
        k_pulled = {}
        stree.clear_output()
        
        # backprop image knowledge.
        stree[(self.S_know.X, ())]  # needed for 'pull_know'.
        image_track = {}
        symm = None if self.mesh.symm_Y_Ids is None else (True, self.mesh.symm_Y_Ids)
        k_image_pulled, noroot_pulled, symm_pulled = backprop_node.pull_know(self.S_know.y, symm_target=symm, track=image_track)
        k_pulled[()] = k_image_pulled

        # backprop derivative knowledge.
        for deriv, S_know_deriv in self.S_know_derivs.items():
            stree[(S_know_deriv.X, deriv)]  # needed for 'pull_know_deriv'.
            k_deriv_pulled = backprop_node.pull_know_deriv(image_track, deriv[0], S_know_deriv.y)
            k_pulled[deriv] = k_deriv_pulled

        return constraints.BackpropConstraints(max_nesting_depth, max_nesting_length, k_pulled, noroot_pulled, symm_pulled)
    
    def __get_relaxed_constraints(self, max_nesting_depth, max_nesting_length) -> constraints.BackpropConstraints:
        k_none = {}
        noroot_none = False
        symm_none = None

        k_none[()] = np.full(self.mesh.X.size, np.nan)
        for deriv, S_know_deriv in self.S_know_derivs.items():
            k_none[deriv] = np.full(self.mesh.X.size, np.nan)

        return constraints.BackpropConstraints(max_nesting_depth, max_nesting_length, k_none, noroot_none, symm_none)
        