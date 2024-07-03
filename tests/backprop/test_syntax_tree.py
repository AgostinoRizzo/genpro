import pytest
import sympy
import numpy as np
import string

from backprop.backprop import \
    SyntaxTree, BinaryOperatorSyntaxTree, UnaryOperatorSyntaxTree, ConstantSyntaxTree, \
    UnknownSyntaxTree, UnknownSyntaxTreeCollector
from backprop import models
import space


@pytest.mark.parametrize("stree", [
    UnaryOperatorSyntaxTree('log',  UnknownSyntaxTree('A')),
    UnaryOperatorSyntaxTree('exp',  UnknownSyntaxTree('A')),
    UnaryOperatorSyntaxTree('sqrt', UnknownSyntaxTree('A')),
    BinaryOperatorSyntaxTree('+',   UnknownSyntaxTree('A'), UnknownSyntaxTree('B')),
    BinaryOperatorSyntaxTree('-',   UnknownSyntaxTree('A'), UnknownSyntaxTree('B')),
    BinaryOperatorSyntaxTree('*',   UnknownSyntaxTree('A'), UnknownSyntaxTree('B')),
    BinaryOperatorSyntaxTree('/',   UnknownSyntaxTree('A'), UnknownSyntaxTree('B')),
    BinaryOperatorSyntaxTree('^',   UnknownSyntaxTree('A'), ConstantSyntaxTree(2)),
    BinaryOperatorSyntaxTree('*',
        UnaryOperatorSyntaxTree('log', UnknownSyntaxTree('A')),
        BinaryOperatorSyntaxTree('/', UnknownSyntaxTree('A'),
            UnaryOperatorSyntaxTree('sqrt', UnknownSyntaxTree('B')))),
    ConstantSyntaxTree(5),
    BinaryOperatorSyntaxTree('+',   ConstantSyntaxTree(2), ConstantSyntaxTree(3))
])
@pytest.mark.parametrize("nvars", [1, 2])
def test_diff(stree, nvars):

    unkn_collector = UnknownSyntaxTreeCollector()
    stree.accept(unkn_collector)
    for unkn in unkn_collector.unknowns:
        unkn.nvars = nvars

    all_derivs = space.get_all_derivs(nvars=nvars, max_derivdeg=2)
    derivs_map = SyntaxTree.diff_all(stree, all_derivs, include_zeroth=False)

    unkn_model = models.ModelFactory.create_poly(deg=1, nvars=nvars)
    unkn_model.set_coeffs(1.)
    
    stree_sympy = stree.to_sympy()
    stree_sympy_symbs = [None] * nvars
    for s in stree_sympy.free_symbols:
        varidx = 0 if len(s.name) == 1 else int(s.name[1:])
        stree_sympy_symbs[varidx] = s
    for i in range(nvars):
        if stree_sympy_symbs[i] is None: stree_sympy_symbs[i] = sympy.Symbol(f"x{i}")

    spsampler = space.UnidimSpaceSampler() if nvars == 1 else space.MultidimSpaceSampler()
    xl = -1.
    xu =  1.
    if nvars > 1:
        xl = np.array([xl]*nvars)
        xu = np.array([xu]*nvars)
    X = spsampler.meshspace(xl, xu, 200)

    for deriv, stree_deriv in derivs_map.items():
        unkn_collector = UnknownSyntaxTreeCollector()
        stree_deriv.accept(unkn_collector)

        for unkn_stree in unkn_collector.unknowns:
            unkn_stree.set_unknown_model(unkn_stree.label, unkn_model.get_deriv(unkn_stree.deriv))
        
        # compute derivative from stree_sympy (reference).
        xs = [stree_sympy_symbs[varidx] for varidx in deriv]
        stree_sympy_deriv = stree_sympy if len(xs) == 0 else sympy.diff(stree_sympy, *xs)
        for f in stree_sympy_deriv.atoms(sympy.Function):
            if str(type(f)) not in string.ascii_uppercase: continue
            stree_sympy_deriv = stree_sympy_deriv.subs(f, unkn_model.to_sympy())
        stree_sympy_deriv = stree_sympy_deriv.doit()
        
        # lambdify stree_sympy_deriv
        stree_sympy_deriv_multi = sympy.lambdify(stree_sympy_symbs, stree_sympy_deriv, 'numpy')
        stree_sympy_deriv = stree_sympy_deriv_multi if nvars == 1 else \
                lambda x, stree_sympy_deriv_multi=stree_sympy_deriv_multi: \
                    stree_sympy_deriv_multi(*( ([np.empty(0)]*S.nvars) if x.size == 0 else x.T ))
        
        # compute mse.
        with np.errstate(divide='ignore', invalid='ignore'):
            y = stree_sympy_deriv(X)
            mse = np.nanmean( (stree_deriv(X) - y) ** 2 )
            assert mse == pytest.approx(0.)