from symbols.syntax_tree import SyntaxTree
from symbols.generator import SyntaxTreeGenerator


INVALID_NESTING = set([
    ('log', 'log'), ('exp', 'exp'), ('sqrt', 'sqrt'),
    ('log', 'exp'), ('exp', 'log'), ('square', 'sqrt'), ('sqrt', 'square'),
])

def can_nest(parent_opt:str, child_opt:str) -> bool:
    global INVALID_NESTING
    if parent_opt is None: return True
    return (parent_opt, child_opt) not in INVALID_NESTING

NESTING_OPERATORS     = {}
UNA_NESTING_OPERATORS = {}
BIN_NESTING_OPERATORS = {}

NESTING_OPERATORS    [None] = SyntaxTreeGenerator.OPERATORS
UNA_NESTING_OPERATORS[None] = SyntaxTreeGenerator.UNA_OPERATORS
BIN_NESTING_OPERATORS[None] = SyntaxTreeGenerator.BIN_OPERATORS
for parent_opt in SyntaxTreeGenerator.OPERATORS:
    opts     = []
    una_opts = []
    bin_opts = []
    
    for child_opt in SyntaxTreeGenerator.OPERATORS:
        if can_nest(parent_opt, child_opt):
            opts.append(child_opt)
    
    for child_opt in SyntaxTreeGenerator.UNA_OPERATORS:
        if can_nest(parent_opt, child_opt):
            una_opts.append(child_opt)
    
    for child_opt in SyntaxTreeGenerator.BIN_OPERATORS:
        if can_nest(parent_opt, child_opt):
            bin_opts.append(child_opt)
    
    NESTING_OPERATORS    [parent_opt] = opts
    UNA_NESTING_OPERATORS[parent_opt] = una_opts
    BIN_NESTING_OPERATORS[parent_opt] = bin_opts


def get_nesting_operators(parent_opt:str) -> list:
    global NESTING_OPERATORS
    return NESTING_OPERATORS[parent_opt]

def get_una_nesting_operators(parent_opt:str) -> list:
    global UNA_NESTING_OPERATORS
    return UNA_NESTING_OPERATORS[parent_opt]

def get_bin_nesting_operators(parent_opt:str) -> list:
    global BIN_NESTING_OPERATORS
    return BIN_NESTING_OPERATORS[parent_opt]
