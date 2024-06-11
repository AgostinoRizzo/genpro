%
% knowledge backprop for +
%

% sign
sign_and(Right, Left, "+", "+", L, U) |
sign_and(Right, Left, "+", "-", L, U) |
sign_and(Right, Left, "-", "+", L, U) :- sign(Model, "+", L, U), bin_tree_node(Model, "+", Right, Left).
:- sign(Model, "+", L, U), bin_tree_node(Model, "+", Right, Left),
   sign(Right, "-", L, U), sign(Left, "-", L, U).

sign_and(Right, Left, "-", "-", L, U) |
sign_and(Right, Left, "+", "-", L, U) |
sign_and(Right, Left, "-", "+", L, U) :- sign(Model, "-", L, U), bin_tree_node(Model, "+", Right, Left).
:- sign(Model, "-", L, U), bin_tree_node(Model, "+", Right, Left),
   sign(Right, "+", L, U), sign(Left, "+", L, U).

% root
sign_and(Right, Left, "+", "-", Input, Input) |
sign_and(Right, Left, "-", "+", Input, Input) :- root(Model, Input), bin_tree_node(Model, "+", Right, Left).
:- root(Model, Input), bin_tree_node(Model, "+", Right, Left),
   sign(Right, "+", Input, Input), sign(Left, "+", Input, Input).
:- root(Model, Input), bin_tree_node(Model, "+", Right, Left),
   sign(Right, "-", Input, Input), sign(Left, "-", Input, Input).


%
% knowledge backprop for -
%

% sign
sign_and(Right, Left, "+", "+", L, U) |
sign_and(Right, Left, "+", "-", L, U) |
sign_and(Right, Left, "-", "-", L, U) :- sign(Model, "+", L, U), bin_tree_node(Model, "-", Right, Left).
:- sign(Model, "+", L, U), bin_tree_node(Model, "-", Right, Left),
   sign(Right, "-", L, U), sign(Left, "+", L, U).

sign_and(Right, Left, "-", "-", L, U) |
sign_and(Right, Left, "-", "+", L, U) |
sign_and(Right, Left, "+", "+", L, U) :- sign(Model, "-", L, U), bin_tree_node(Model, "-", Right, Left).
:- sign(Model, "-", L, U), bin_tree_node(Model, "-", Right, Left),
   sign(Right, "+", L, U), sign(Left, "-", L, U).

% root
sign_and(Right, Left, "+", "+", Input, Input) |
sign_and(Right, Left, "-", "-", Input, Input) :- root(Model, Input), bin_tree_node(Model, "-", Right, Left).
:- root(Model, Input), bin_tree_node(Model, "-", Right, Left),
   sign(Right, "+", Input, Input), sign(Left, "-", Input, Input).
:- root(Model, Input), bin_tree_node(Model, "-", Right, Left),
   sign(Right, "-", Input, Input), sign(Left, "+", Input, Input).


%
% symmetry backprop for + and -
%
addOrSub("+"). addOrSub("-").
even_symm(F, X) :- bin_tree_node(Model, OPT, F, _), addOrSub(OPT), even_symm(Model, X).
even_symm(G, X) :- bin_tree_node(Model, OPT, _, G), addOrSub(OPT), even_symm(Model, X).
odd_symm (F, X) :- bin_tree_node(Model, OPT, F, _), addOrSub(OPT), odd_symm (Model, X).
odd_symm (G, X) :- bin_tree_node(Model, OPT, _, G), addOrSub(OPT), odd_symm (Model, X).
:- bin_tree_node(Model, OPT, F, _), addOrSub(OPT), even_symm(Model, X), not even_symm(F, X).
:- bin_tree_node(Model, OPT, _, G), addOrSub(OPT), even_symm(Model, X), not even_symm(G, X).
:- bin_tree_node(Model, OPT, F, _), addOrSub(OPT), odd_symm (Model, X), not odd_symm (F, X).
:- bin_tree_node(Model, OPT, _, G), addOrSub(OPT), odd_symm (Model, X), not odd_symm (G, X).