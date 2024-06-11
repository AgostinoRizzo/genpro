import itertools
import backprop

def _merge_pair(c1, c2) -> list:
    dp_y_1, relopt_1 = c1
    dp_y_2, relopt_2 = c2
    opts = f"{relopt_1}{relopt_2}"
    
    merge_map = {  # opts: (cond, keep)
        '==': (dp_y_1 == dp_y_2, [c1]),
        '=>': (dp_y_1 > dp_y_2, [c1]),
        '=<': (dp_y_1 < dp_y_2, [c1]),
        '>=': (dp_y_2 > dp_y_1, [c2]),
        '<=': (dp_y_2 < dp_y_1, [c2]),
        '>>': (True, [c1] if max(dp_y_1, dp_y_2) == dp_y_1 else [c2]),
        '<<': (True, [c1] if min(dp_y_1, dp_y_2) == dp_y_1 else [c2]),
        '><': (dp_y_1 < dp_y_2, [c1, c2]),
        '<>': (dp_y_1 > dp_y_2, [c1, c2])
    }
    
    if opts not in merge_map.keys():
        raise RuntimeError(f"Merge between operators {opts} not supported.")
    
    cond, ret = merge_map[opts]
    if cond: return ret
    return [c1, c2]


class ConstraintMap:
    def __init__(self):
        self.map = {}

    def add(self, model_name:str, dp_x:float, dp_y:float, relopt:backprop.Relopt):
        if model_name not in self.map.keys():
            self.map[model_name] = {}
        if dp_x not in self.map[model_name].keys():
            self.map[model_name][dp_x] = set()
        self.map[model_name][dp_x].add( (dp_y, relopt.opt) )
    
    def compute_fitting(self, model_name:str, model:callable) -> float:
        fit = 0.
        for dp_x in self.map[model_name].keys():
            for (dp_y, relopt) in self.map[model_name][dp_x]:
                resid = model(dp_x) - dp_y
                if   relopt == '=': fit += abs(resid)
                elif relopt == '>': fit += abs(min(0, resid))
                elif relopt == '<': fit += abs(max(0, resid))
                else: raise RuntimeError(f"Operator {relopt} not supported.")
        return fit
    
    def merge(self):
        for model_name in self.map.keys():
            for dp_x in self.map[model_name].keys():

                constrs = self.map[model_name][dp_x]
                if len(constrs) < 2: continue

                keep_constrs = None

                """for (dp_y, relopt) in self.map[model_name][dp_x]:
                    dp_x_str = "{:.2f}".format(dp_x)
                    dp_y_str = "{:.2f}".format(dp_y)
                    print(f"{model_name}({dp_x_str}) {relopt} {dp_y_str}\n")"""

                while True:
                    keep_constrs  = set()
                    throw_constrs = set()

                    for (c1, c2) in itertools.combinations(constrs, 2):
                        if c1 in throw_constrs or c2 in throw_constrs:
                            continue
                        merged_pair = _merge_pair(c1, c2)
                        for c in [c1, c2]:
                            if c in merged_pair and c not in throw_constrs: keep_constrs.add(c)
                            else:
                                throw_constrs.add(c)
                                if c in keep_constrs: keep_constrs.remove(c)

                    #print(f"{len(keep_constrs)} {len(throw_constrs)}\n")
                    if len(throw_constrs) > 0 and len(keep_constrs) >= 2: constrs = keep_constrs
                    else: break
                
                self.map[model_name][dp_x] = keep_constrs    

    def __str__(self) -> str:
        out = ''
        for model_name in self.map.keys():
            out += f"--- Constraints for {model_name} ---\n"
            for dp_x in self.map[model_name].keys():
                for (dp_y, relopt) in self.map[model_name][dp_x]:
                    dp_x_str = "{:.2f}".format(dp_x)
                    dp_y_str = "{:.2f}".format(dp_y)
                    out += f"{model_name}({dp_x_str}) {relopt} {dp_y_str}\n"
            out += "\n"
        return out