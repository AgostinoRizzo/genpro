from functools import cmp_to_key

class DataLengthFrontTracker:

    def __init__(self):
        self.front = []

    def track(self, stree, evaluation):
        stree_data = evaluation.data_r2
        stree_length = evaluation.nnodes

        if stree_data < 0.5: return

        to_remove = []
        is_dominated = False

        for idx, (_, data, length) in enumerate(self.front):
            if stree_data > data and stree_length < length:
                to_remove.append(idx)
            
            if data >= stree_data and length <= stree_length:
                is_dominated = True
        
        if len(to_remove) > 0 or not is_dominated:
            for idx in sorted(to_remove, reverse=True):
                self.front.pop(idx)
            self.front.append((stree, stree_data, stree_length))

    def get_front(self):
        def fcmp(a, b):
            return (a[1] / a[2]) - (b[1] / b[2])
        return sorted(self.front, key=cmp_to_key(fcmp), reverse=True)
    
    def get_population(self):
        return [stree for stree, _, _ in self.get_front()]