projection := module()
option package:
export PCAD_McCallumProj, PCAD_SetFactors:
local PCAD_MakeMonic, PCAD_MakeInteger, PCAD_RemoveConstantMultiples, PCAD_SFBasis, PCAD_ContSet, PCAD_PrimSet, PCAD_McCcoeffset:

PCAD_MakeMonic:=proc( poly::polynom, RorVar, $ ) :: polynom:
local i, vars, R, pol:

if RorVar::'table' and RorVar['type']='polynomial_ring' then
	vars:=RorVar['variables']:
	R:=RorVar:
elif RorVar::'list'('symbol') then
	vars:=RorVar:
	R:=RegularChains:-PolynomialRing(vars):
else
	error("the second argument should either be a polynomial ring or a list of variables. This is an unexpected error message - please report to M.England@bath.ac.uk"):
fi:
if indets(poly,'name') minus convert(vars,set) <> {} then
	error("the polynomial has variables other than those specified. This is an unexpected error message - please report to M.England@bath.ac.uk"):
fi:
if poly::'constant' then
	return(poly)
else
	pol:=poly:
	i:=1:
fi:
while i<nops(vars)+1 do
	pol:=RegularChains:-Initial(pol,R):
	if pol::'constant' then
		return( expand(poly/pol) ):
	else
		i:=i+1:
	fi:
od:
error("leading coefficient could not be identified. This is an unexpected error message - please report to M.England@bath.ac.uk"):
end proc:

PCAD_MakeInteger:=proc( poly::polynom, $ ) :: polynom:
local bottom:

bottom:=denom(poly):
if bottom::'constant' then
	return( expand(bottom*poly) ):
else
	return(poly)
fi:
end proc:

PCAD_RemoveConstantMultiples:=proc( pols, $ ) :: set;
local polys, vars, RET:

if pols::list then
	polys:=convert(pols,'set'):
elif pols::set then
	polys:=pols:
else
	error("first input should be a list or set of polynomials. This is an unexpected error message - please report to M.England@bath.ac.uk"):
fi:
if polys={} then
	return({}):
fi:
vars:=convert(indets(polys,'name'),list):
RET := map(X->PCAD_MakeMonic(X, vars), polys):
RET := map(X->PCAD_MakeInteger(X), RET):
end proc:

PCAD_SFBasis:=proc( pols, var::symbol, $ ) :: set;
local polys, i, j, decomp, head, body, RET:

if pols::list then
	polys:=convert(pols,set):
elif pols::set
	then polys:=pols:
else
	error("first input should be a list or set of polynomials. This is an unexpected error message - please report to M.England@bath.ac.uk"):
fi:
RET:={}:
for i from 1 to nops(polys) do
	decomp:=sqrfree(op(i,polys),var):
	head:=op(1,decomp):
	body:=op(2,decomp):
	if type(head,constant)=false then
		RET:={op(RET),op(1,decomp)}:
	fi:
	for j from 1 to nops(body) do
		RET:={op(RET), op(1,op(j,body))}:
	od:
od:
return(RET):
end proc:

PCAD_ContSet:=proc( pols, var::symbol, $ ) :: set:
local S, i, polys:

if pols::list then
	polys:=convert(pols,set):
elif pols::set then
	polys:=pols:
else
	error("first input should be a list or set of polynomials. This is an unexpected error message - please report to M.England@bath.ac.uk"):
fi:
S:={seq( content(op(i,polys),var), i=1..nops(polys))}:
remove(X->X::'constant',S):
end proc:

PCAD_PrimSet:=proc( pols, var::symbol, $ ) :: set:
local S, i, polys:

if pols::list then
	polys:=convert(pols,set):
elif pols::set then
	polys:=pols:
else
	error("first input should be a list or set of polynomials. This is an unexpected error message - please report to M.England@bath.ac.uk"):
fi:
S:={seq( primpart(op(i,polys),var), i=1..nops(polys))}:
remove(X->X::'constant',S):
end proc:

PCAD_SetFactors := proc(inpols, $) :: set:
local S, F, i;

if inpols::list then
	S:=convert(inpols,set):
elif inpols::set then
	S:=inpols:
else
	error("first input should be a list or set of polynomials. This is an unexpected error message - please report to M.England@bath.ac.uk"):
fi:
F:=table():
for i from 1 to nops(S) do
	F[i]:=op(map(X->op(1,X),op(2,factors(op(i,S))))):
od:
F:={entries(F,'nolist')}:
end proc:

PCAD_McCallumProj:=proc( polys::set, var::symbol, lvars::list, $) :: set:
local Polys, cont, Pol, clist, Pset1, Pset2, Pset, i, j:

Polys:=PCAD_PrimSet(polys,var):
cont:=PCAD_ContSet(polys,var);
Polys:=PCAD_SFBasis(Polys,var):
userinfo(3,{'PCAD_McCallumProj', 'CADProjection'},"now using McCallum projection with respect to", var):
userinfo(3,{'PCAD_McCallumProj', 'CADProjection'},"square free basis of primitive part consisting of ", nops(Polys), "polynomials"):
userinfo(4,{'PCAD_McCallumProj', 'CADProjection'},"the polynomials are", Polys):
userinfo(4,{'PCAD_McCallumProj', 'CADProjection'},"content ", cont, " was added to output."):
Pset1:=table():
for i from 1 to nops(Polys) do
	Pol:=op(i,Polys):
	clist:=PCAD_McCcoeffset(Pol,var, convert(lvars,'set')):
	Pset1[i]:=discrim(Pol,var),op(clist):
	userinfo(4,{'PCAD_McCallumProj', 'CADProjection'},"added discriminant of polynomial", i, "and first", nops(clist), "coeffs"):
od:
Pset2:=table():
for i from 1 to nops(Polys) do
	for j from i+1 to nops(Polys) do
		Pset2[i,j]:=resultant( op(i,Polys), op(j,Polys), var):
		userinfo(4,{'PCAD_McCallumProj', 'CADProjection'},"added resultant of polynomials ", i, "and ", j):
	od:
od:
Pset:={op(cont), entries(Pset1,'nolist'), entries(Pset2,'nolist')}:
Pset:=remove(X->type(X,constant),Pset):
userinfo(3,{'PCAD_McCallumProj', 'CADProjection'}, nops(Pset), "non constant projection polynomials found"):
Pset:=PCAD_RemoveConstantMultiples(Pset):
userinfo(3,{'PCAD_McCallumProj', 'CADProjection'},"removed constants and constant multiples to leave", nops(Pset)):
Pset:
end proc:

PCAD_McCcoeffset:=proc( poly::polynom, var::symbol, lvars::set, $ ) :: list:
local i, crw, deg, cflist, tvar, res:

deg:=degree(poly,var):
cflist:=ListTools:-Reverse( [seq( coeff(poly,var,i), i=0..deg)] ):
crw:=op(1,cflist):
if crw::'constant' then
	return([]):
elif nops(lvars)=1 then
	return([crw]):
fi:
for i from 2 to nops(cflist) do
	tvar:=indets(crw,'name')[1]:
	res:=resultant(crw,op(i,cflist), tvar):
	if res=0 then
		crw:=crw:
	elif res::'constant' then
		return([op(1..i,cflist)]):
	else
		crw:=res:
	fi:
od:
cflist:
end proc:

end module:
