%
% knowledge backprop for * and /
%

% sign
mulOrDiv("*"). mulOrDiv("/").
sign_and(Num, Den, "+", "+", L, U) |
sign_and(Num, Den, "-", "-", L, U) :- sign(Model, "+", L, U), bin_tree_node(Model, MulDiv, Num, Den), mulOrDiv(MulDiv).
:- sign(Model, "+", L, U), bin_tree_node(Model, MulDiv, Num, Den), mulOrDiv(MulDiv),
   sign(Num, "+", L, U), sign(Den, "-", L, U).
:- sign(Model, "+", L, U), bin_tree_node(Model, MulDiv, Num, Den), mulOrDiv(MulDiv),
   sign(Num, "-", L, U), sign(Den, "+", L, U).

sign_and(Num, Den, "+", "-", L, U) |
sign_and(Num, Den, "-", "+", L, U) :- sign(Model, "-", L, U), bin_tree_node(Model, MulDiv, Num, Den), mulOrDiv(MulDiv).
:- sign(Model, "-", L, U), bin_tree_node(Model, MulDiv, Num, Den), mulOrDiv(MulDiv),
   sign(Num, "+", L, U), sign(Den, "+", L, U).
:- sign(Model, "-", L, U), bin_tree_node(Model, MulDiv, Num, Den), mulOrDiv(MulDiv),
   sign(Num, "-", L, U), sign(Den, "-", L, U).

% root
root(Left, Input) | root(Right, Input) :- root(Model, Input), bin_tree_node(Model, "*", Left, Right).
:- root(Model, Input), bin_tree_node(Model, "*", Left, Right), not root(Left, Input), not root(Right, Input).

root(Num, Input) :- root(Model, Input), bin_tree_node(Model, "/", Num, _).
:- root(Model, Input), bin_tree_node(Model, "/", Num, _), not root(Num, Input).
:- bin_tree_node(_, "/", _, Den), root(Den, _).

% symmetry
even_symm_and(F, G, X) |
odd_symm_and (F, G, X) :- bin_tree_node(Model, OPT, F, G), mulOrDiv(OPT), even_symm(Model, X).
even_odd_symm_and(F, G, X) |
odd_even_symm_and(F, G, X) :- bin_tree_node(Model, OPT, F, G), mulOrDiv(OPT), odd_symm (Model, X).
:- bin_tree_node(Model, OPT, F, G), mulOrDiv(OPT), even_symm(Model, X),
   even_symm(F, X), odd_symm(G, X).
:- bin_tree_node(Model, OPT, F, G), mulOrDiv(OPT), even_symm(Model, X),
   odd_symm(F, X), even_symm(G, X).
:- bin_tree_node(Model, OPT, F, G), mulOrDiv(OPT), odd_symm(Model, X),
   even_symm(F, X), even_symm(G, X).
:- bin_tree_node(Model, OPT, F, G), mulOrDiv(OPT), odd_symm(Model, X),
   odd_symm(F, X), odd_symm(G, X).