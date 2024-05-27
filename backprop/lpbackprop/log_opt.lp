%
% knowledge backprop for log (natural)
%

% sign
sign(G, "+", L, U) :- un_tree_node(F, "log", G).  % independently from sign of F.
:- un_tree_node(F, "log", G), not sign(G, "+", L, U).

% root
% nothing here.

% symmetry
even_symm(G, X) :- un_tree_node(F, "log", G), even_symm(F, X).
:- un_tree_node(F, "log", G), even_symm(F, X), not even_symm(G, X).
:- un_tree_node(F, "log", _), odd_symm(F, _).