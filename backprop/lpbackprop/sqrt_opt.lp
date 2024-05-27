%
% knowledge backprop for sqrt
%

% sign
sign(G, "+", L, U) :- sign(F, "+", L, U), un_tree_node(F, "sqrt", G).
:- sign(F, "+", L, U), un_tree_node(F, "sqrt", G), not sign(G, "+", L, U).
:- sign(F, "-", L, U), un_tree_node(F, "sqrt", _).

% root
root(G, X) :- root(F, X), un_tree_node(F, "sqrt", G).
:- root(F, X), un_tree_node(F, "sqrt", G), not root(G, X).

% symmetry
even_symm(G, X) :- un_tree_node(F, "sqrt", G), even_symm(F, X).
:- un_tree_node(F, "sqrt", G), even_symm(F, X), not even_symm(G, X).
:- un_tree_node(F, "sqrt", _), odd_symm(F, _).