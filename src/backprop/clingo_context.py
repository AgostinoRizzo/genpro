import clingo

class ClingoArith:
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