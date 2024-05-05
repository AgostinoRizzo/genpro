% knowledge backprop for ^
even(Exp) | sign(Base, "+", L, U)  :- sign(Model, "+", L, U), bin_tree_node(Model, "^", Base, Exp), const(Exp).
:- sign(Model, "+", L, U), bin_tree_node(Model, "^", Base, Exp), const(Exp),
   odd(Exp), sign(Base, "-", L, U).

sign_and_odd(Base, "-", L, U, Exp) :- sign(Model, "-", L, U), bin_tree_node(Model, "^", Base, Exp), const(Exp).
:- sign(Model, "-", L, U), bin_tree_node(Model, "^", Base, Exp), const(Exp),
   sign(Base, "+", L, U).
:- sign(Model, "-", L, U), bin_tree_node(Model, "^", Base, Exp), const(Exp),
   even(Exp).

root(Base, Input) :- root(Model, Input), bin_tree_node(Model, "^", Base, _).
:- root(Model, Input), bin_tree_node(Model, "^", Base, _), not root(Base, Input).