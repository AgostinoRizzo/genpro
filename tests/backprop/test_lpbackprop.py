import pytest
import re
import numpy as np

from symbols.syntax_tree import SyntaxTree
from symbols.binop import BinaryOperatorSyntaxTree
from symbols.misc import UnknownSyntaxTree
from backprop.lpbackprop import ASPSpecBuilder, build_knowledge_spec, synthesize_unknown
import dataset
import dataset_misc1d
import dataset_misc2d
import space
from numlims import NumericLimits


@pytest.mark.parametrize("stree,nvars,stree_spec", [
    (BinaryOperatorSyntaxTree('/', UnknownSyntaxTree('A'), UnknownSyntaxTree('B')), 1,
    """
    bin_tree_node("m","/","A","B").
    unkn_tree_node("A").
    unkn_tree_node("B").
    bin_tree_node("d0m","/","m1","m2").
    bin_tree_node("m1","-","m3","m4").
    bin_tree_node("m3","*","d0A","B").
    unkn_tree_node("d0A").
    unkn_tree_node("B").
    bin_tree_node("m4","*","A","d0B").
    unkn_tree_node("A").
    unkn_tree_node("d0B").
    bin_tree_node("m2","^","B","m5").
    unkn_tree_node("B").
    const_tree_node("m5",2).
    const(2).
    bin_tree_node("d0d0m","/","m6","m7").
    bin_tree_node("m6","-","m8","m9").
    bin_tree_node("m8","*","m10","m11").
    bin_tree_node("m10","-","m12","m13").
    bin_tree_node("m12","+","m14","m15").
    bin_tree_node("m14","*","d0d0A","B").
    unkn_tree_node("d0d0A").
    unkn_tree_node("B").
    bin_tree_node("m15","*","d0A","d0B").
    unkn_tree_node("d0A").
    unkn_tree_node("d0B").
    bin_tree_node("m13","+","m16","m17").
    bin_tree_node("m16","*","d0A","d0B").
    unkn_tree_node("d0A").
    unkn_tree_node("d0B").
    bin_tree_node("m17","*","A","d0d0B").
    unkn_tree_node("A").
    unkn_tree_node("d0d0B").
    bin_tree_node("m11","^","B","m18").
    unkn_tree_node("B").
    const_tree_node("m18",2).
    const(2).
    bin_tree_node("m9","*","m19","m20").
    bin_tree_node("m19","-","m21","m22").
    bin_tree_node("m21","*","d0A","B").
    unkn_tree_node("d0A").
    unkn_tree_node("B").
    bin_tree_node("m22","*","A","d0B").
    unkn_tree_node("A").
    unkn_tree_node("d0B").
    bin_tree_node("m20","*","m23","d0B").
    bin_tree_node("m23","*","m24","B").
    const_tree_node("m24",2).
    const(2).
    unkn_tree_node("B").
    unkn_tree_node("d0B").
    bin_tree_node("m7","^","B","m25").
    unkn_tree_node("B").
    const_tree_node("m25",4).
    const(4).
    """)
])
def test_ASPSpecBuilder(stree:SyntaxTree, nvars:int, stree_spec:str):

    all_derivs = space.get_all_derivs(nvars=nvars, max_derivdeg=2)
    stree_map = SyntaxTree.diff_all(stree, all_derivs, include_zeroth=True)

    aspSpecBuilder = ASPSpecBuilder()
    for deriv, stree in stree_map.items(): aspSpecBuilder.map_root(stree, deriv)
    for deriv, stree in stree_map.items(): stree.accept(aspSpecBuilder)
    
    actual   = re.sub(r"[\s\t\n]+", '', aspSpecBuilder.spec)
    expected = re.sub(r"[\s\t\n]+", '', stree_spec)
    assert actual == expected


@pytest.mark.parametrize("K,expected_K_spec,expected_breakpts_coords", [
    (dataset_misc1d.MagmanDatasetScaled().knowledge,
    """
    root("m",0).
    root("d0m",-1).
    root("d0m",1).
    sign("m","+",-4,-3).
    sign("m","+",-3,-2).
    sign("m","+",-2,-1).
    sign("m","+",-1,0).
    sign("m","-",0,1).
    sign("m","-",1,2).
    sign("m","-",2,3).
    sign("m","-",3,4).
    sign("d0m","+",-4,-3).
    sign("d0m","+",-3,-2).
    sign("d0m","+",-2,-1).
    sign("d0m","-",-1,0).
    sign("d0m","-",0,1).
    sign("d0m","+",1,2).
    sign("d0m","+",2,3).
    sign("d0m","+",3,4).
    sign("d0d0m","+",-4,-3).
    sign("d0d0m","+",-3,-2).
    sign("d0d0m","-",-2,-1).
    sign("d0d0m","-",-1,0).
    sign("d0d0m","+",0,1).
    sign("d0d0m","+",1,2).
    sign("d0d0m","-",2,3).
    sign("d0d0m","-",3,4).
    odd_symm("m",0).
    even_symm("d0m",0).
    odd_symm("d0d0m",0).
    :- tree_node(N), undef(N, _).
    infty(4).
    infty(-4).
    varidx("0").
    """,
    {0: 0, -0.2082733245333337: -1, 0.2082733245333337: 1, 2.0: 3, -0.36073997999999996: -2,
     0.36073997999999996: 2, 10.0: 4, -10.0: -4, -2.0: -3}
    ),

    (dataset_misc2d.Resistance2().knowledge,
    """
    root("m",(0, 1)).
    root("m",(1, 0)).
    sign("m","+",(0, 0),(1, 1)).
    sign("m","+",(0, 1),(1, 2)).
    sign("m","+",(1, 0),(2, 1)).
    sign("m","+",(1, 1),(2, 2)).
    sign("d0m","+",(0, 0),(1, 1)).
    sign("d0m","+",(0, 1),(1, 2)).
    sign("d0m","+",(1, 0),(2, 1)).
    sign("d0m","+",(1, 1),(2, 2)).
    sign("d1m","+",(0, 0),(1, 1)).
    sign("d1m","+",(0, 1),(1, 2)).
    sign("d1m","+",(1, 0),(2, 1)).
    sign("d1m","+",(1, 1),(2, 2)).
    sign("d0d0m","-",(0, 0),(1, 1)).
    sign("d0d0m","-",(0, 1),(1, 2)).
    sign("d0d0m","-",(1, 0),(2, 1)).
    sign("d0d0m","-",(1, 1),(2, 2)).
    sign("d1d1m","-",(0, 0),(1, 1)).
    sign("d1d1m","-",(0, 1),(1, 2)).
    sign("d1d1m","-",(1, 0),(2, 1)).
    sign("d1d1m","-",(1, 1),(2, 2)).
    :- tree_node(N), undef(N, _).
    infty(2).
    infty(-2).
    varidx("0").
    varidx("1").
    """,
    {0: 0, -60.0: -2, 60.0: 2, 20.0: 1}
    )
])
def test_build_knowledge_spec(K:dataset.DataKnowledge, expected_K_spec:str, expected_breakpts_coords:dict):

    actual_K_spec, break_points, break_point_coords_map, break_point_coords_invmap = build_knowledge_spec(K, model_name='m')
    
    actual_K_spec   = re.sub(r"[\s\t\n]+", '', actual_K_spec)
    expected_K_spec = re.sub(r"[\s\t\n]+", '', expected_K_spec)
    assert actual_K_spec == expected_K_spec
    
    n_breakpts_coords = len(expected_breakpts_coords)
    assert len(break_point_coords_map) == n_breakpts_coords and \
           len(break_point_coords_map) == len(break_point_coords_invmap)

    for coord_idx, coord_val in expected_breakpts_coords.items():
        assert coord_idx in break_point_coords_map
        assert coord_val in break_point_coords_invmap
        assert coord_val == break_point_coords_map[coord_idx]
        assert coord_idx == break_point_coords_invmap[coord_val]
    
    for p in break_points:
        for coord in p:
            assert coord in break_point_coords_map


@pytest.mark.parametrize("deriv,sign,symm,numlims,breakpts,nvars,expected_coeffs", [
    (
        {0: (0,0), 2: (0,0)},
        {0: [(-10.0, -2.0, '-'), (-2.0, -0.36073997999999996, '-'), (-0.36073997999999996, -0.2082733245333337, '-'), (-0.2082733245333337, 0, '-'),
             (0, 0.2082733245333337, '+'), (0.2082733245333337, 0.36073997999999996, '+'), (0.36073997999999996, 2.0, '+'), (2.0, 10.0, '+') ],
         1: [(0.2082733245333337, 0.36073997999999996, '+'), (0.36073997999999996, 2.0, '+'), (2.0, 10.0, '+'), (-10.0, -2.0, '+'),
             (-2.0, -0.36073997999999996, '+'), (-0.36073997999999996, -0.2082733245333337, '+'), (0, 0.2082733245333337, '+'), (-0.2082733245333337, 0, '+')],
         2: [(0, 0.2082733245333337, '-'), (0.2082733245333337, 0.36073997999999996, '-'), (-10.0, -2.0, '+'), (-2.0, -0.36073997999999996, '+'),
             (0.36073997999999996, 2.0, '-'), (2.0, 10.0, '-'), (-0.36073997999999996, -0.2082733245333337, '+'), (-0.2082733245333337, 0, '+')]
        },
        {0: (0, 'odd'), 1: (0, 'even'), 2: (0, 'odd')},
        NumericLimits(10.0, 1e-12, 1e-50, 1e-10),
        [0, -0.2082733245333337, 2.0, 0.2082733245333337, -0.36073997999999996, 0.36073997999999996, 10.0, -10.0, -2.0],
        1,
        [0., 0., 0., 0., 0., 1., 0.]
    ),

    (
        {0: (0,0), 2: (0,0)},
        {0: [(-10.0, -2.0, '+'), (-2.0, -0.36073997999999996, '+'), (-0.36073997999999996, -0.2082733245333337, '+'), (-0.2082733245333337, 0, '+'),
             (0, 0.2082733245333337, '-'), (0.2082733245333337, 0.36073997999999996, '-'), (0.36073997999999996, 2.0, '-'), (2.0, 10.0, '-') ],
         1: [(0.2082733245333337, 0.36073997999999996, '-'), (0.36073997999999996, 2.0, '-'), (2.0, 10.0, '-'), (-10.0, -2.0, '-'),
             (-2.0, -0.36073997999999996, '-'), (-0.36073997999999996, -0.2082733245333337, '-'), (0, 0.2082733245333337, '-'), (-0.2082733245333337, 0, '-')],
         2: [(0, 0.2082733245333337, '+'), (0.2082733245333337, 0.36073997999999996, '+'), (-10.0, -2.0, '-'), (-2.0, -0.36073997999999996, '-'),
             (0.36073997999999996, 2.0, '+'), (2.0, 10.0, '+'), (-0.36073997999999996, -0.2082733245333337, '-'), (-0.2082733245333337, 0, '-')]
        },
        {0: (0, 'odd'), 1: (0, 'even'), 2: (0, 'odd')},
        NumericLimits(10.0, 1e-12, 1e-50, 1e-10),
        [0, -0.2082733245333337, 2.0, 0.2082733245333337, -0.36073997999999996, 0.36073997999999996, 10.0, -10.0, -2.0],
        1,
        [0., 0., 0., 0., 0., -1., 0.]
    ),

    (
        {0: (0,0), 2: (0,0)},
        {0: [(-10.0, -2.0, '-'), (-2.0, -0.36073997999999996, '-'), (-0.36073997999999996, -0.2082733245333337, '-'), (-0.2082733245333337, 0, '-'),
             (0, 0.2082733245333337, '+'), (0.2082733245333337, 0.36073997999999996, '+'), (0.36073997999999996, 2.0, '+'), (2.0, 10.0, '+') ],
         1: [(0.2082733245333337, 0.36073997999999996, '+'), (0.36073997999999996, 2.0, '+'), (2.0, 10.0, '+'), (-10.0, -2.0, '+'),
             (-2.0, -0.36073997999999996, '+'), (-0.36073997999999996, -0.2082733245333337, '+'), (0, 0.2082733245333337, '+'), (-0.2082733245333337, 0, '+')],
         2: [(0, 0.2082733245333337, '+'), (0.2082733245333337, 0.36073997999999996, '+'), (-10.0, -2.0, '-'), (-2.0, -0.36073997999999996, '-'),
             (0.36073997999999996, 2.0, '+'), (2.0, 10.0, '+'), (-0.36073997999999996, -0.2082733245333337, '-'), (-0.2082733245333337, 0, '-')]
        },
        {0: (0, 'odd'), 1: (0, 'even'), 2: (0, 'odd')},
        NumericLimits(10.0, 1e-12, 1e-50, 1e-10),
        [0, -0.2082733245333337, 2.0, 0.2082733245333337, -0.36073997999999996, 0.36073997999999996, 10.0, -10.0, -2.0],
        1,
        [0., 1., 0., 1., 0., 1., 0.]
    ),

    (
        {0: (0,0), 2: (0,0)},
        {0: [(-10.0, -2.0, '+'), (-2.0, -0.36073997999999996, '+'), (-0.36073997999999996, -0.2082733245333337, '+'), (-0.2082733245333337, 0, '+'),
             (0, 0.2082733245333337, '-'), (0.2082733245333337, 0.36073997999999996, '-'), (0.36073997999999996, 2.0, '-'), (2.0, 10.0, '-') ],
         1: [(0.2082733245333337, 0.36073997999999996, '-'), (0.36073997999999996, 2.0, '-'), (2.0, 10.0, '-'), (-10.0, -2.0, '-'),
             (-2.0, -0.36073997999999996, '-'), (-0.36073997999999996, -0.2082733245333337, '-'), (0, 0.2082733245333337, '-'), (-0.2082733245333337, 0, '-')],
         2: [(0, 0.2082733245333337, '-'), (0.2082733245333337, 0.36073997999999996, '-'), (-10.0, -2.0, '+'), (-2.0, -0.36073997999999996, '+'),
             (0.36073997999999996, 2.0, '-'), (2.0, 10.0, '-'), (-0.36073997999999996, -0.2082733245333337, '+'), (-0.2082733245333337, 0, '+')]
        },
        {0: (0, 'odd'), 1: (0, 'even'), 2: (0, 'odd')},
        NumericLimits(10.0, 1e-12, 1e-50, 1e-10),
        [0, -0.2082733245333337, 2.0, 0.2082733245333337, -0.36073997999999996, 0.36073997999999996, 10.0, -10.0, -2.0],
        1,
        [0., -1., 0., -1., 0., -1., 0.]
    ),


    (
        {1: (0,0), 2: (0,0)},
        {0: [(-10.0, -2.0, '-'), (-2.0, -0.36073997999999996, '-'), (-0.36073997999999996, -0.2082733245333337, '-'), (-0.2082733245333337, 0, '-'),
             (0, 0.2082733245333337, '-'), (0.2082733245333337, 0.36073997999999996, '-'), (0.36073997999999996, 2.0, '-'), (2.0, 10.0, '-') ],
         1: [(0.2082733245333337, 0.36073997999999996, '-'), (0.36073997999999996, 2.0, '-'), (2.0, 10.0, '-'), (-10.0, -2.0, '+'),
             (-2.0, -0.36073997999999996, '+'), (-0.36073997999999996, -0.2082733245333337, '+'), (0, 0.2082733245333337, '-'), (-0.2082733245333337, 0, '+')],
         2: [(0, 0.2082733245333337, '-'), (0.2082733245333337, 0.36073997999999996, '-'), (-10.0, -2.0, '-'), (-2.0, -0.36073997999999996, '-'),
             (0.36073997999999996, 2.0, '-'), (2.0, 10.0, '-'), (-0.36073997999999996, -0.2082733245333337, '-'), (-0.2082733245333337, 0, '-')]
        },
        {0: (0, 'even'), 1: (0, 'odd'), 2: (0, 'even')},
        NumericLimits(10.0, 1e-12, 1e-50, 1e-10),
        [0, -0.2082733245333337, 2.0, 0.2082733245333337, -0.36073997999999996, 0.36073997999999996, 10.0, -10.0, -2.0],
        1,
        [-1., 0., -1., 0., 0., 0., -1.]
    ),

    (
        {1: (0,0), 2: (0,0)},
        {0: [(-10.0, -2.0, '+'), (-2.0, -0.36073997999999996, '+'), (-0.36073997999999996, -0.2082733245333337, '+'), (-0.2082733245333337, 0, '+'),
             (0, 0.2082733245333337, '+'), (0.2082733245333337, 0.36073997999999996, '+'), (0.36073997999999996, 2.0, '+'), (2.0, 10.0, '+') ],
         1: [(0.2082733245333337, 0.36073997999999996, '+'), (0.36073997999999996, 2.0, '+'), (2.0, 10.0, '+'), (-10.0, -2.0, '-'),
             (-2.0, -0.36073997999999996, '-'), (-0.36073997999999996, -0.2082733245333337, '-'), (0, 0.2082733245333337, '+'), (-0.2082733245333337, 0, '-')],
         2: [(0, 0.2082733245333337, '+'), (0.2082733245333337, 0.36073997999999996, '+'), (-10.0, -2.0, '+'), (-2.0, -0.36073997999999996, '+'),
             (0.36073997999999996, 2.0, '+'), (2.0, 10.0, '+'), (-0.36073997999999996, -0.2082733245333337, '+'), (-0.2082733245333337, 0, '+')]
        },
        {0: (0, 'even'), 1: (0, 'odd'), 2: (0, 'even')},
        NumericLimits(10.0, 1e-12, 1e-50, 1e-10),
        [0, -0.2082733245333337, 2.0, 0.2082733245333337, -0.36073997999999996, 0.36073997999999996, 10.0, -10.0, -2.0],
        1,
        [1., 0., 1., 0., 0., 0., 1.]
    ),

])
def test_synthesize_unknown(deriv, sign, symm, numlims, breakpts, nvars, expected_coeffs):
    K = dataset.DataKnowledge(
        limits=numlims,
        spsampler= space.UnidimSpaceSampler() if nvars == 1 else space.MultidimSpaceSampler()
        )

    for d, (x,y) in deriv.items():
        K.add_deriv(d, dataset.DataPoint(x, y))

    for d, constrs in sign.items():
        for l, u, sign in constrs:
            K.add_sign(d, l, u, sign)
    
    for d, (x0, even_odd) in symm.items():
        K.add_symm(d, x0, even_odd == 'even')
    
    breakpts_lst = breakpts
    breakpts = []
    for bp in breakpts_lst:
        breakpts.append(np.array([bp]))
    
    P, _, _ = synthesize_unknown('A', K, breakpts, nvars)
    np.testing.assert_allclose(P.get_coeffs(), expected_coeffs)
