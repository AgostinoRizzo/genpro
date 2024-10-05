class GPStats:
    def __init__(self):
        self.best = None
        self.best_eval = None
        self.qualities = {'currBest': [], 'currAvg': [], 'currWorst': [], 'best': []}
    
    def update(self, population, eval_map):
        currBest  = 0.0
        currAvg   = 0.0
        currWorst = 1.0

        for stree in population:
            stree_eval = eval_map[id(stree)]
            stree_eval_value = stree_eval.get_value()
            
            currBest   = max(currBest, stree_eval_value)
            currAvg   += stree_eval_value
            currWorst  = min(currWorst, stree_eval_value)

            if self.best is None or stree_eval.better_than(self.best_eval):
                self.best = stree
                self.best_eval = stree_eval

        currAvg /= len(population)

        self.qualities['currBest' ].append(currBest)    
        self.qualities['currAvg'  ].append(currAvg)
        self.qualities['currWorst'].append(currWorst)
        #self.qualities['best'     ].append(self.bests_eval_map[id(self.bests[0])].get_value())
        

class LayeredGPStats(GPStats):
    def __init__(self,):
        super().__init__()
        self.buckets = {}
        self.fea_ratio = {'currBest': [], 'currAvg': [], 'currWorst': [], 'best': []}
    
    def update(self, population, eval_map):
        super().update(population, eval_map)

        fea_ratio_best  = 0.0
        fea_ratio_avg   = 0.0
        fea_ratio_worst = 1.0
        
        for stree in population:
            fea_ratio = eval_map[id(stree)].fea_ratio

            fea_ratio_best   = max(fea_ratio_best, fea_ratio)
            fea_ratio_avg   += fea_ratio
            fea_ratio_worst  = min(fea_ratio_worst, fea_ratio)
        
        fea_ratio_avg /= len(population)

        self.fea_ratio['currBest' ].append(fea_ratio_best)    
        self.fea_ratio['currAvg'  ].append(fea_ratio_avg)
        self.fea_ratio['currWorst'].append(fea_ratio_worst)
        #self.fea_ratio['best'     ].append(self.bests_eval_map[id(self.bests[0])].fea_ratio)