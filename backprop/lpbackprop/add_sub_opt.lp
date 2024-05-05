% knowledge backprop for +
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

sign_and(Right, Left, "+", "-", Input, Input) |
sign_and(Right, Left, "-", "+", Input, Input) :- root(Model, Input), bin_tree_node(Model, "+", Right, Left).
:- root(Model, Input), bin_tree_node(Model, "+", Right, Left),
   sign(Right, "+", Input, Input), sign(Left, "+", Input, Input).
:- root(Model, Input), bin_tree_node(Model, "+", Right, Left),
   sign(Right, "-", Input, Input), sign(Left, "-", Input, Input).


% knowledge backprop for -
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

sign_and(Right, Left, "+", "+", Input, Input) |
sign_and(Right, Left, "-", "-", Input, Input) :- root(Model, Input), bin_tree_node(Model, "-", Right, Left).
:- root(Model, Input), bin_tree_node(Model, "-", Right, Left),
   sign(Right, "+", Input, Input), sign(Left, "-", Input, Input).
:- root(Model, Input), bin_tree_node(Model, "-", Right, Left),
   sign(Right, "-", Input, Input), sign(Left, "+", Input, Input).