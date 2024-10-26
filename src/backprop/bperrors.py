class BackpropError(RuntimeError):
    pass

class KnowBackpropError(BackpropError):
    pass

class NoBackpropPathError(BackpropError):
    pass

class BackpropGrammarError(BackpropError):
    pass

class BackpropMaxLengthError(BackpropError):
    pass