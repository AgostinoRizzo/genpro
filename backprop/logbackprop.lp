% bin_tree_node(NodeName, Operator, LeftNode, RightNode)
% unkn_tree_node(NodeName)
% sign(ModelName, PosNeg, LowerBound, UpperBound)
% sign_and(ModelName1, ModelName2, PosNeg1, PosNeg2, LowerBound, UpperBound)
% root(ModelName, Input)

#const infty=100000.


% syntax tree m(x) = A(x)/B(x)
bin_tree_node("m", "/", "A", "B").
unkn_tree_node("A").
unkn_tree_node("B").

tree_node(N) :- bin_tree_node(N, _, _, _).
tree_node(N) :- unkn_tree_node(N).

% prior knowledge.
sign("m", "+", -infty, 0).  % TODO: manage closed intervals (now are open).
sign("m", "-", 0, infty).
root("m", 0).
:- tree_node(N), undef(N, _).
% TODO: add simmetry + backpropagate?.

%
% prior knowledge back propagation.
%
sign_and(Num, Den, "+", "+", L, U) |
sign_and(Num, Den, "-", "-", L, U) :- sign(Model, "+", L, U), bin_tree_node(Model, "/", Num, Den).

sign_and(Num, Den, "+", "-", L, U) |
sign_and(Num, Den, "-", "+", L, U) :- sign(Model, "-", L, U), bin_tree_node(Model, "/", Num, Den).

root(Num, Input) :- root(Model, Input), bin_tree_node(Model, "/", Num, _).

sign(N1, PN1, L, U) :- sign_and(N1, _, PN1, _, L, U).
sign(N2, PN2, L, U) :- sign_and(_, N2, _, PN2, L, U).

%
% hyper knowledge back propagation (search space reduction).
%
root(N, Mid) | undef(N, Mid) :- sign(N, PN1, _, Mid), sign(N, PN2, Mid, _), PN1 != PN2.
:- bin_tree_node(_, "/", _, Den), root(Den, _).

%
% prefer simple (unknown) models.
%
sign_changes(N, C) :- #count{Mid : sign(N, PN1, _, Mid), sign(N, PN2, Mid, _), PN1!=PN2}=C, tree_node(N).
root_count(N, C) :- #count{Input : root(N, Input)}=C, tree_node(N).
:~ sign_changes(N, C), unkn_tree_node(N). [C@2]
:~ root_count(N, C), unkn_tree_node(N). [C@1]

%
% join sign intervals.
%
sign(N, PN, L, U) :- sign(N, PN, L, Mid), sign(N, PN, Mid, U).  % TODO: inclusion of Mid only in case of closed interval.
% TODO interval overlapping + add constraints for opposite signs.

%
% get knowledge abound unknown models
%
sign_unkn(N, PN, L, U) :- sign(N, PN, L, U), unkn_tree_node(N).
root_unkn(N, Input) :- root(N, Input), unkn_tree_node(N).
undef_unkn(N, Input) :- undef(N, Input), unkn_tree_node(N).
#show sign_unkn/4.
#show root_unkn/2.
#show undef_unkn/2.
#show sign_changes/2.
