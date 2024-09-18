from scipy.special import softmax as scipy_softmax
import random
import numpy as np
from scipy.spatial.distance import pdist as scipy_pdist
from scipy.spatial.distance import squareform as scipy_squareform

from backprop import backprop, gp, models, library


class ConstrBackpropCrossover(gp.Crossover):
    def __init__(self, max_depth:int, S_train):
        self.max_depth = max_depth
        self.evaluator = R2Evaluator(S_train)
    
    def cross(self, parent1:backprop.SyntaxTree, parent2:backprop.SyntaxTree) -> backprop.SyntaxTree:
        nnodes1 = parent1.get_nnodes()
        nnodes2 = parent2.get_nnodes()
        
        child = parent1.clone()
        nodeSelector = backprop.SyntaxTreeNodeSelector(random.randrange(nnodes1))
        child.accept(nodeSelector)
        cross_point1 = nodeSelector.node
        child.set_parent()
        cross_point1_depth = cross_point1.get_depth()
        max_nesting_depth = self.max_depth - cross_point1_depth
        invertible_path = backprop.SyntaxTree.is_invertible_path(cross_point1)
        
        nodesCollector = backprop.SyntaxTreeNodeCollector()
        parent2.accept(nodesCollector)
        allowedNodes = []
        for node in nodesCollector.nodes:
            if node.get_max_depth() <= max_nesting_depth: allowedNodes.append(node)
        
        if len(allowedNodes) == 0: return child
        cross_point2 = random.choice(allowedNodes)

        got = replace_subtree(child, cross_point1, cross_point2.clone())
        got.clear_cache()

        p1_eval = self.evaluator.evaluate(parent1)
        got_eval = self.evaluator.evaluate(got)
        if got_eval.better_than(p1_eval):
            print(f"{cross_point2.simplify()}")

        return got

"""
class SoftmaxSubTreeCrossover:
    def __init__(self, max_depth:int):
        self.max_depth = max_depth
    
    def cross(self, parent1:backprop.SyntaxTree, parent2:backprop.SyntaxTree) -> backprop.SyntaxTree:
        nnodes1 = parent1.get_nnodes()
        nnodes2 = parent2.get_nnodes()
        
        child = parent1.clone()
        nodeSelector = backprop.SyntaxTreeNodeSelector(random.randrange(nnodes1))
        child.accept(nodeSelector)
        cross_point1 = nodeSelector.node
        child.set_parent()
        if cross_point1 is None:
            parent1.get_nnodes()
        cross_point1_depth = cross_point1.get_depth()
        max_nesting_depth = self.max_depth - cross_point1_depth
        
        nodesCollector = backprop.SyntaxTreeNodeCollector()
        parent2.accept(nodesCollector)
        allowedNodes = []
        for node in nodesCollector.nodes:
            if node.get_max_depth() <= max_nesting_depth: allowedNodes.append(node)
        
        if len(allowedNodes) == 0: return child

        # apply softmax on allowed nodes.
        allowerNodesProb = np.array( [n.K_sat for n in allowedNodes] )
        cross_point2 = None
        if np.isinf(allowerNodesProb).all():
            cross_point2 = random.choice(allowedNodes)
        else:
            #print(f"\n{parent2}")
            #for i, n in enumerate(allowedNodes):
            #    print(f"{n} {allowerNodesProb[i]}")

            allowerNodesProb = scipy_softmax(-allowerNodesProb)
            cross_point2 = np.random.choice(allowedNodes, size=1, p=allowerNodesProb)[0]
        
        got = gp.replace_subtree(child, cross_point1, cross_point2.clone())
        got.clear_cache()

        return got
"""

class SoftmaxSubTreeCrossover:
    def __init__(self, max_depth:int):
        self.max_depth = max_depth
    
    def cross(self, parent1:backprop.SyntaxTree, parent2:backprop.SyntaxTree) -> backprop.SyntaxTree:
        
        child = parent1.clone()
        child.set_parent()

        parent1NodeCollector = backprop.SyntaxTreeNodeCollector()
        parent1.accept(parent1NodeCollector)

        childNodeCollector = backprop.SyntaxTreeNodeCollector()
        child.accept(childNodeCollector)

        # apply softmax on child nodes.
        childNodeProbs  = np.array( [(n.match_r2) for n in parent1NodeCollector.nodes] )
        childNodeDepths = np.array( [n.get_depth() for n in parent1NodeCollector.nodes] )
        #childNodeProbs = childNodeProbs ** (childNodeDepths + 1)
        #childNodeProbs = scipy_softmax(childNodeProbs)
        childNodeProbsSum = np.sum(childNodeProbs)
        childNodeProbs = (childNodeProbs / childNodeProbsSum) if childNodeProbsSum > 0 else None
        cross_point1 = np.random.choice(childNodeCollector.nodes, size=1, p=childNodeProbs)[0]

        #if childNodeProbs is not None and (childNodeProbs < 0.001).any():
        can_print = childNodeProbs is not None and type(child) is backprop.BinaryOperatorSyntaxTree and child.operator == '/' and \
            type(child.left) is backprop.VariableSyntaxTree and \
            type(child.right) is backprop.UnaryOperatorSyntaxTree and child.right.operator == 'cube'
        
        if can_print:
            print(f"\nCrossing {child}")
            for i, n in enumerate(childNodeCollector.nodes):
                print(f"{childNodeProbs[i]} {n} -> {n.parent}")
            print(f"Choosen cross point {cross_point1}")
        
        cross_point1_depth = cross_point1.get_depth()
        max_nesting_depth = self.max_depth - cross_point1_depth
        
        nodesCollector = backprop.SyntaxTreeNodeCollector()
        parent2.accept(nodesCollector)
        allowedNodes = []
        for node in nodesCollector.nodes:
            if node.get_max_depth() <= max_nesting_depth: allowedNodes.append(node)
        
        if len(allowedNodes) == 0: return child
        cross_point2 = random.choice(allowedNodes)
        
        got = gp.replace_subtree(child, cross_point1, cross_point2.clone())
        got.clear_cache()

        if can_print:
            print(f"Parent 2 {parent2}")
            print(f"Choosen second cross point {cross_point2}")
            print(f"Offspring {got}")
            print(f"Offspring max depth {got.get_max_depth()}")

        return got


class MatchSubTreeCrossover:
    def __init__(self, max_depth:int):
        self.max_depth = max_depth
    
    def cross(self, parent1:backprop.SyntaxTree, parent2:backprop.SyntaxTree) -> backprop.SyntaxTree:
        nnodes1 = parent1.get_nnodes()
        nnodes2 = parent2.get_nnodes()
        
        child = parent1.clone()

        nodesCollector = backprop.SyntaxTreeNodeCollector()
        parent1.accept(nodesCollector)
        parent1_nodes = nodesCollector.nodes
        matchableNodesIdx = []
        for i, n in enumerate(parent1_nodes):
            if n.y is not None:
                matchableNodesIdx.append(i)
        
        nodesCollector = backprop.SyntaxTreeNodeCollector()
        child.accept(nodesCollector)
        child_nodes = nodesCollector.nodes

        matchableNodes = []
        for i in matchableNodesIdx:
            mn = child_nodes[i]
            mn.X = parent1_nodes[i].X
            mn.y = parent1_nodes[i].y
            matchableNodes.append(mn)
        
        if len(matchableNodes) > 0:
            best_n = None
            best_mn = None
            best_mse = None
            parent2(matchableNodes[0].X)
            nodesCollector = backprop.SyntaxTreeNodeCollector()
            parent2.accept(nodesCollector)
            
            for mn in matchableNodes:
                for n in nodesCollector.nodes:

                    if mn.get_depth() + n.get_max_depth() + 1 > self.max_depth:
                        continue
                    
                    mse = np.sum( (n.output - mn.y) ** 2 ) / mn.y.size
                    if best_mse is None or mse < best_mse:
                        best_n = n
                        best_mn = mn
                        best_mse = mse
            
            if best_n is not None:
                cross_point1 = best_mn
                cross_point2 = best_n
                got = gp.replace_subtree(child, cross_point1, cross_point2.clone())
                got.clear_cache()
                return got


        nodeSelector = backprop.SyntaxTreeNodeSelector(random.randrange(nnodes1))
        child.accept(nodeSelector)
        cross_point1 = nodeSelector.node
        child.set_parent()
        if cross_point1 is None:
            parent1.get_nnodes()
        cross_point1_depth = cross_point1.get_depth()
        max_nesting_depth = self.max_depth - cross_point1_depth
        
        nodesCollector = backprop.SyntaxTreeNodeCollector()
        parent2.accept(nodesCollector)
        allowedNodes = []
        for node in nodesCollector.nodes:
            if node.get_max_depth() <= max_nesting_depth: allowedNodes.append(node)
        
        if len(allowedNodes) == 0: return child
        cross_point2 = random.choice(allowedNodes)

        got = gp.replace_subtree(child, cross_point1, cross_point2.clone())
        got.clear_cache()
        
        return got


class OptimalGeometricCrossover(gp.Crossover):
    def __init__(self, maxdepth, data, know):
        self.data = data
        self.know = know
        self.sx = gp.SubTreeCrossover(maxdepth)
    
    def cross(self, parent1:backprop.SyntaxTree, parent2:backprop.SyntaxTree) -> backprop.SyntaxTree:
        parent1_sem = parent1(self.data.X)
        parent2_sem = parent2(self.data.X)

        if np.isnan(parent1_sem).any() or np.isnan(parent2_sem).any() or \
           np.isinf(parent1_sem).any() or np.isinf(parent2_sem).any():

            if type(parent1) is not backprop.SemanticSyntaxTree and type(parent2) is not backprop.SemanticSyntaxTree:
                return self.sx.cross(parent1, parent2)
            return parent1

        # A @ x = b with x = [alpha, beta]
        A = np.column_stack((parent1_sem, parent2_sem))
        b = self.data.y

        x = np.linalg.lstsq(A, b)
        alpha = x[0][0]
        beta  = x[0][1]

        child_sem = parent1_sem * alpha + parent2_sem * beta
        child = backprop.SemanticSyntaxTree(child_sem)

        return child


class EclipseCrossover(gp.Crossover):
    def __init__(self, selector, data):
        self.selector = selector
        self.data = data
    
    def cross(self, parent1:backprop.SyntaxTree, parent2:backprop.SyntaxTree) -> backprop.SyntaxTree:
        l1, l2 = self.selector.cross_coeffs[(id(parent1), id(parent2))]

        if l1 == 0 and l2 == 0: return backprop.ConstantSyntaxTree(0.0)
        if l1 == 0: return parent2.clone().scale(l2)
        if l2 == 0: return parent1.clone().scale(l1)
        
        #return backprop.BinaryOperatorSyntaxTree('+', parent1.clone().scale(l1), parent2.clone().scale(l2))

        child = parent1.clone() if abs(l1) >= abs(l2) else parent2.clone()

        s1 = parent1.clone().scale(l1)(self.data.X)
        s2 = parent2.clone().scale(l2)(self.data.X)
        sT = s1 + s2

        nodesCollector = backprop.SyntaxTreeNodeCollector()
        child.accept(nodesCollector)

        backprop_nodes = []
        for node in nodesCollector.nodes:
            if id(node) != id(child) and backprop.SyntaxTree.is_invertible_path(node):
                backprop_nodes.append(node)
        
        if len(backprop_nodes) == 0:
            return child

        for bp_node in backprop_nodes:

            child.set_parent()        
            y = child(self.data.X)
            
            pulled_y, _ = bp_node.pull_output(sT)
            if (pulled_y == 0).all(): continue

            sat_pulled_y = not np.isnan(pulled_y).any() and not np.isinf(pulled_y).any()
            if not sat_pulled_y: continue

            coeffs, residual, _, _, _ = np.polyfit(self.data.X.ravel(), pulled_y, 6, full=True)  # TODO: fix max degree.
            P = models.ModelFactory.create_poly(6)
            P.set_coeffs(coeffs)

            if residual < 1e-10:
                print(f"Best found!")
                return gp.replace_subtree(child, bp_node, P.to_stree().simplify())

        return child


class ApproxGeometricCrossover(gp.Crossover):

    def __init__(self, lib, max_depth, diversifier=None):
        self.lib = lib
        self.fallback_crossover = gp.SubTreeCrossover(max_depth)
        self.diversifier = diversifier
    
    def cross(self, parent1:backprop.SyntaxTree, parent2:backprop.SyntaxTree) -> backprop.SyntaxTree:
        
        child = parent1.clone()  # TODO: avoid cloning...
        nodesCollector = backprop.SyntaxTreeNodeCollector()
        child.accept(nodesCollector)

        cross_node = random.choice(nodesCollector.nodes)

        if not backprop.SyntaxTree.is_invertible_path(cross_node):
            return self.fallback_crossover.cross(parent1, parent2)

        """backprop_nodes = []
        for node in nodesCollector.nodes:
            if backprop.SyntaxTree.is_invertible_path(node):
                backprop_nodes.append(node)
        
        if len(backprop_nodes) == 0:
            return self.fallback_crossover.cross(parent1, parent2)"""

        child.set_parent()        
        y = child(self.lib.data.X)

        """best_bp_node = None
        best_bp_dist = None
        for bp_node in backprop_nodes:
            pulled_y, _ = bp_node.pull_output(self.lib.data.y)
            if (pulled_y == 0).all(): continue

            sat_pulled_y = not np.isnan(pulled_y).any() and not np.isinf(pulled_y).any()
            if not sat_pulled_y: continue

            d = self.lib.query_dist(pulled_y)
            if best_bp_node is None or d < best_bp_dist:
                best_bp_node = bp_node
                best_bp_dist = d

        if best_bp_node is None:
            return self.fallback_crossover.cross(parent1, parent2)

        bp_node = random.choice(backprop_nodes)"""
        
        target_cross = True

        """sem1 = parent1(self.lib.data.X)
        sem2 = parent2(self.lib.data.X)
        undef_sem1 = np.isnan(sem1).any() or np.isinf(sem1).any()
        undef_sem2 = np.isnan(sem2).any() or np.isinf(sem2).any()
        if undef_sem1 and undef_sem2: target_cross = False
        if self.diversifier is not None and not undef_sem1 and not undef_sem2:
            dist = scipy_squareform( scipy_pdist( np.array([sem1, sem2]) ) )[0].max()
            if dist > self.diversifier.mean_dist:
                target_cross = False"""

        pulled_y = None
        if target_cross: pulled_y, _ = cross_node.pull_output(self.lib.data.y)
        else: pulled_y, _ = cross_node.pull_output(parent1(self.lib.data.X) * 0.5 + parent2(self.lib.data.X) * 0.5)
        if (pulled_y == 0).all(): return self.fallback_crossover.cross(parent1, parent2)

        sat_pulled_y = not np.isnan(pulled_y).any() and not np.isinf(pulled_y).any()
        if not sat_pulled_y: return self.fallback_crossover.cross(parent1, parent2)

        new_sub_stree = self.lib.query(pulled_y)
        if new_sub_stree is None: return self.fallback_crossover.cross(parent1, parent2)

        offspring = gp.replace_subtree(child, cross_node, new_sub_stree)
        if offspring.get_max_depth() > 5:  # TODO: lookup based on max admissible depth.
            return self.fallback_crossover.cross(parent1, parent2)
        
        return offspring


class CrossNPushCrossover(gp.Crossover):

    def __init__(self, lib, max_depth):
        self.lib = lib
        self.main_crossover = gp.SubTreeCrossover(max_depth)
    
    def cross(self, parent1:backprop.SyntaxTree, parent2:backprop.SyntaxTree) -> backprop.SyntaxTree:
        
        child = self.main_crossover.cross(parent1, parent2)
        
        backprop_nodes = child.cache.backprop_nodes
        if len(backprop_nodes) == 0:
            return child
        cross_node = random.choice(backprop_nodes)

        child.set_parent()
        y = child(self.lib.data.X)  # needed for 'pull_output'.

        pulled_y, _ = cross_node.pull_output(self.lib.data.y)
        if (pulled_y == 0).all(): return child

        sat_pulled_y = not np.isnan(pulled_y).any() and not np.isinf(pulled_y).any()
        if not sat_pulled_y: return child

        cross_node_sem = cross_node(self.lib.data.X)
        cross_node_dist = np.linalg.norm(cross_node_sem - pulled_y)

        new_sub_stree = self.lib.query(pulled_y, max_dist=cross_node_dist)
        if new_sub_stree is None: return child
        
        origin_child = child.clone()
        offspring = gp.replace_subtree(child, cross_node, new_sub_stree)
        if offspring.get_max_depth() > 5:  # TODO: lookup based on max admissible depth.
            return origin_child
        
        offspring.set_parent()
        if new_sub_stree.has_parent():
            new_sub_stree.parent.invalidate_output()
        return offspring


class ConstrainedCrossNPushCrossover(CrossNPushCrossover):

    def __init__(self, lib, max_depth, know_evaluator):
        super().__init__(lib, max_depth)
        self.know_evaluator = know_evaluator
    
    def cross(self, parent1:backprop.SyntaxTree, parent2:backprop.SyntaxTree) -> backprop.SyntaxTree:
        
        child = self.main_crossover.cross(parent1, parent2)
        
        backprop_nodes = child.cache.backprop_nodes
        if len(backprop_nodes) == 0:
            return child
        
        cross_node = random.choice(backprop_nodes)

        child.set_parent()
        y = child(self.lib.data.X)  # needed for 'pull_output'.

        pulled_y, _ = cross_node.pull_output(self.lib.data.y)
        if (pulled_y == 0).all(): return child

        sat_pulled_y = not np.isnan(pulled_y).any() and not np.isinf(pulled_y).any()
        if not sat_pulled_y: return child

        new_sub_strees = self.lib.multiquery(pulled_y)
        if new_sub_strees is None: return child

        best_offspring = None
        best_know_nv = None

        for new_sub_sem, new_sub_stree in new_sub_strees:

            child = gp.replace_subtree(child, cross_node, new_sub_stree)

            if child.get_max_depth() > 5:  # TODO: lookup based on max admissible depth.
                gp.replace_subtree(child, new_sub_stree, cross_node)
                continue
            
            child.set_parent()
            if new_sub_stree.has_parent():
                new_sub_stree.parent.stash_output()
            
            know_nv = self.evaluate_offspring(child)

            if best_offspring is None or know_nv < best_know_nv:
                best_offspring = child.clone()
                best_know_nv = know_nv
            
            if new_sub_stree.has_parent():
                new_sub_stree.parent.backup_output()
            
            gp.replace_subtree(child, new_sub_stree, cross_node)
            child.set_parent()
        
        if best_offspring is None:
            return child
        
        return best_offspring
    
    def evaluate_offspring(self, offspring) -> int:
        know_eval = self.know_evaluator.evaluate(offspring)
        know_nv   =  know_eval['nv0'] + know_eval['nv1'] + know_eval['nv2']
        return know_nv


class ConstrainedSubTreeCrossover(gp.Crossover):
    def __init__(self, population:list, eval_map:dict, selector:gp.Selector, S_data, S_know, X_mesh):
        self.lib = library.DynamicConstrainedLibrary(population, eval_map, selector, S_data, X_mesh)
        self.max_depth = 5
        self.S_data = S_data
        self.S_know = S_know
    
    def cross(self, parent1:backprop.SyntaxTree, parent2:backprop.SyntaxTree) -> backprop.SyntaxTree:
        child = parent1.clone()
        child.set_parent()

        cross_point1 = random.choice(child.cache.nodes)
        cross_point1_depth = cross_point1.get_depth()
        max_nesting_depth = self.max_depth - cross_point1_depth
        max_nesting_depth = max(max_nesting_depth, 0)  # TODO
        
        cross_point2 = None

        if backprop.SyntaxTree.is_invertible_path(cross_point1):
            child.clear_output()
            child[(self.S_know.X, ())]  # needed for 'pull_know'.
            k_pulled, noroot_pulled = cross_point1.pull_know(self.S_know.y)
            K_pulled = (k_pulled, noroot_pulled)

            y = child(self.S_data.X)  # needed for 'pull_output'.
            pulled_y, _ = cross_point1.pull_output(self.S_data.y)

            if k_pulled is None or noroot_pulled is None:
                cross_point2 = self.lib.query(max_nesting_depth, pulled_y)  # TODO: unsat stree.
            else:
                cross_point2 = self.lib.cquery(K_pulled, max_nesting_depth, pulled_y)
        
        else:
            cross_point2 = self.lib.query(max_nesting_depth)
            
        child = gp.replace_subtree(child, cross_point1, cross_point2)
        child.cache.clear()
        child.set_parent()
        if cross_point2.has_parent():
            cross_point2.parent.invalidate_output()
        return child
    