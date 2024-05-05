% knowledge backprop for * and /
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


root(Left, Input) | root(Right, Input) :- root(Model, Input), bin_tree_node(Model, "*", Left, Right).
:- root(Model, Input), bin_tree_node(Model, "*", Left, Right), not root(Left, Input), not root(Right, Input).

root(Num, Input) :- root(Model, Input), bin_tree_node(Model, "/", Num, _).
:- root(Model, Input), bin_tree_node(Model, "/", Num, _), not root(Num, Input).
:- bin_tree_node(_, "/", _, Den), root(Den, _).