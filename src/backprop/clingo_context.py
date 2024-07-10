import clingo

class ClingoContext:
    def add(self, t1, t2):
        if t1.type is clingo.SymbolType.Number and t2.type is clingo.SymbolType.Number:
            return clingo.Number(t1.number + t2.number)
        if t1.type is clingo.SymbolType.Function and t2.type is clingo.SymbolType.Function:
            sum_args = [0] * len(t1.arguments)
            for i in range(len(sum_args)):
                sum_args[i] = clingo.Number(t1.arguments[i].number + t2.arguments[i].number)
            return clingo.Function(t1.name, sum_args)
        raise RuntimeError(f"Add operation not defined for types {t1.type} and {t2.type}.")
    
    def sub(self, t1, t2):
        if t1.type is clingo.SymbolType.Number and t2.type is clingo.SymbolType.Number:
            return clingo.Number(t1.number - t2.number)
        if t1.type is clingo.SymbolType.Function and t2.type is clingo.SymbolType.Function:
            sum_args = [0] * len(t1.arguments)
            for i in range(len(sum_args)):
                sum_args[i] = clingo.Number(t1.arguments[i].number - t2.arguments[i].number)
            return clingo.Function(t1.name, sum_args)
        raise RuntimeError(f"Sub operation not defined for types {t1.type} and {t2.type}.")
    
    def concat(self, t1, t2):
        if t1.type is clingo.SymbolType.String and t2.type is clingo.SymbolType.String:
            return clingo.String(t1.string + t2.string)
        raise RuntimeError(f"Concat operation not defined for types {t1.type} and {t2.type}.")
    
    def contains(self, t1, t2):
        if t1.type is clingo.SymbolType.Number:
            if t1 == t2: return clingo.Number(1)
            return clingo.Number(0)
        if t1.type is clingo.SymbolType.Function:
            for arg in t1.arguments:
                if t2 == arg: return clingo.Number(1)
            return clingo.Number(0)
        raise RuntimeError(f"Contains operation not defined for types {t1.type} and {t2.type}.")