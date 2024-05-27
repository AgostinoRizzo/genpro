%
% knowledge backprop for exp
%

% sign
:- sign(F, "-", L, U), un_tree_node(F, "exp", _).

% root
:- root(F, X), un_tree_node(F, "exp", _).

% symmetry
even_symm(G, X) :- un_tree_node(F, "exp", G), even_symm(F, X).
:- un_tree_node(F, "exp", G), even_symm(F, X), not even_symm(G, X).
:- un_tree_node(F, "exp", _), odd_symm(F, _).