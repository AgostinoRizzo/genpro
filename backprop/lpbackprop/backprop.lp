% bin_tree_node(NodeName, Operator, LeftNode, RightNode)
% unkn_tree_node(NodeName)
% sign(ModelName, PosNeg, LowerBound, UpperBound)
% sign_and(ModelName1, ModelName2, PosNeg1, PosNeg2, LowerBound, UpperBound)
% root(ModelName, Input)
% even_symm(ModelName, Input)
% odd_symm(ModelName, Input)

#const infty=100000.


% syntax tree m(x) = A(x)/B(x)
bin_tree_node("m", "/", "A", "B").

% syntax tree m'(x) = (((A'(x) * B(x)) - (A(x) * B'(x))) / (B(x) ^ 2))
bin_tree_node("m'", "/", "m'_num", "m'_den").
bin_tree_node("m'_num", "-", "m'_num_left", "m'_num_right").
bin_tree_node("m'_num_left",  "*", "A'", "B" ).
bin_tree_node("m'_num_right", "*", "A" , "B'").
bin_tree_node("m'_den", "^", "B", 2).
const(2).  % TODO: factorize.
even(2).
:- even(N), odd(N).

unkn_tree_node("A").
unkn_tree_node("B").
unkn_tree_node("A'").
unkn_tree_node("B'").
deriv("A", "A'").
deriv("B", "B'").
deriv("A'", "A''").
deriv("B'", "B''").

tree_node(N) :- bin_tree_node(N, _, _, _).
tree_node(N) :- unkn_tree_node(N).

% solution (real model).
%sign("A", "+", -infty, -peak_x). sign("A", "+", -peak_x, 0).
%sign("A", "-", 0, peak_x). sign("A", "-", peak_x, infty).
%sign("A'", "-", -infty, -peak_x). sign("A'", "-", -peak_x, 0).
%sign("A'", "-", 0, peak_x). sign("A'", "-", peak_x, infty).

%sign("B", "+", -infty, -peak_x). sign("B", "+", -peak_x, 0).
%sign("B", "+", 0, peak_x). sign("B", "+", peak_x, infty).
%sign("B'", "-", -infty, -peak_x). sign("B'", "-", -peak_x, 0).
%sign("B'", "+", 0, peak_x). sign("B'", "+", peak_x, infty).

% prior knowledge (image).
#const peak_x=1.
sign("m", "+", -infty, -peak_x). sign("m", "+", -peak_x, 0).  % TODO: manage closed intervals (now are open).
sign("m", "-", 0, peak_x). sign("m", "-", peak_x, infty).
root("m", 0).
:- tree_node(N), undef(N, _).

% prior knowledge (first deriv).
sign("m'", "+", -infty, -peak_x).
sign("m'", "-", -peak_x, 0). sign("m'", "-", 0, peak_x).
sign("m'", "+",  peak_x, infty).
root("m'", -peak_x).
root("m'", peak_x).

% prior knowledge (symmetry).
odd_symm ("m"  , 0).
even_symm("m'" , 0).
odd_symm ("m''", 0).

%
% prior knowledge back propagation.
%
% see {...}_opt.lp encodings.
% utility backprop rules.
sign(N1, PN1, L, U) :- sign_and(N1, _, PN1, _, L, U).
sign(N2, PN2, L, U) :- sign_and(_, N2, _, PN2, L, U).
sign(Base, PN, L, U) :- sign_and_odd(Base, PN, L, U, _).
odd(Val) :- sign_and_odd(_, _, _, _, Val).

even_symm(F, X) :- even_symm_and(F, _, X).
even_symm(G, X) :- even_symm_and(_, G, X).
odd_symm (F, X) :- odd_symm_and (F, _, X).
odd_symm (G, X) :- odd_symm_and (_, G, X).
even_symm(F, X) :- even_odd_symm_and(F, _, X).
odd_symm (G, X) :- even_odd_symm_and(_, G, X).
odd_symm (F, X) :- odd_even_symm_and(F, _, X).
even_symm(G, X) :- odd_even_symm_and(_, G, X).

%
% hyper knowledge back propagation (search space reduction).
%
% see {...}_opt.lp encodings.
% root
root(N, Mid) | undef(N, Mid) :- sign(N, PN1, _, Mid), sign(N, PN2, Mid, _), PN1 != PN2.
sign_change(N, Input) :- sign(N, PN1, _, Input), sign(N, PN2, Input, _), PN1!=PN2.
:- root(N, Input), not sign_change(N, Input), not sign_change(N_pr, Input), deriv(N, N_pr).
% symmetry
:- even_symm(F, X1), even_symm(F, X2), X1!=X2.
:- odd_symm(F, X1), odd_symm(F, X2), X1!=X2.
:- even_symm(F, _), odd_symm(F, _).
% symmetry + sign (TODO: no generation here)
:- even_symm(F, X), sign(F, PN1, L1, U1), L1<U1, U1<=X, sign(F, PN2, L2, U2), L2=X+(X-U1), U2=X+(X-L1), PN1!=PN2.
:- odd_symm (F, X), sign(F, PN1, L1, U1), L1<U1, U1<=X, sign(F, PN2, L2, U2), L2=X+(X-U1), U2=X+(X-L1), PN1=PN2.
% symmetry + root (TODO: no generation here)
:- even_symm(F, X), root(F, X1), X1<X, not root(F, X2), X2=X+X-X1.
:- even_symm(F, X), root(F, X1), X1>X, not root(F, X2), X2=X-(X1-X).
:- odd_symm (F, X), root(F, X1), X1<X, not root(F, X2), X2=X+X-X1.
:- odd_symm (F, X), root(F, X1), X1>X, not root(F, X2), X2=X-(X1-X).

%
% prefer simple (unknown) models.
%
sign_changes(N, C) :- #count{Mid : sign(N, PN1, _, Mid), sign(N, PN2, Mid, _), PN1!=PN2}=C, tree_node(N).
root_count(N, C) :- #count{Input : root(N, Input)}=C, tree_node(N).
:~ W=#sum{C,N: sign_changes(N, C), unkn_tree_node(N)}. [W@2]
:~ W=#sum{C,N: root_count(N, C), unkn_tree_node(N)}. [W@1]

%
% join sign intervals.
%
%sign(N, PN, L, U) :- sign(N, PN, L, Mid), sign(N, PN, Mid, U).  % TODO: inclusion of Mid only in case of closed interval.
% interval overlapping + constraints for opposite signs.
:- sign(N, PN1, L1, U1), sign(N, PN2, L2, U2), PN1!=PN2, L2<U1, U2>L1.
:- sign(N, PN1, L,  U ), sign(N, PN2, L , U ), PN1!=PN2.

%
% get knowledge abound unknown models
%
sign_unkn(N, PN, L, U) :- sign(N, PN, L, U), unkn_tree_node(N), N="A'".
root_unkn(N, Input) :- root(N, Input), unkn_tree_node(N).
undef_unkn(N, Input) :- undef(N, Input), unkn_tree_node(N).
#show sign_unkn/4.
#show root_unkn/2.
#show undef_unkn/2.
%#show sign_changes/2.
%#show even_symm/2.