% knowledge backprop for ^
even(ExpVal) | sign(Base, "+", L, U)  :- sign(Model, "+", L, U), bin_tree_node(Model, "^", Base, Exp), const_tree_node(Exp, ExpVal).
:- sign(Model, "+", L, U), bin_tree_node(Model, "^", Base, Exp), const_tree_node(Exp, ExpVal),
   odd(ExpVal), sign(Base, "-", L, U).

sign_and_odd(Base, "-", L, U, ExpVal) :- sign(Model, "-", L, U), bin_tree_node(Model, "^", Base, Exp), const_tree_node(Exp, ExpVal).
:- sign(Model, "-", L, U), bin_tree_node(Model, "^", Base, Exp), const_tree_node(Exp, _),
   sign(Base, "+", L, U).
:- sign(Model, "-", L, U), bin_tree_node(Model, "^", Base, Exp), const_tree_node(Exp, ExpVal),
   even(ExpVal).

root(Base, Input) :- root(Model, Input), bin_tree_node(Model, "^", Base, _).
:- root(Model, Input), bin_tree_node(Model, "^", Base, _), not root(Base, Input).