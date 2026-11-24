from __future__ import annotations

import ast
import itertools
from dataclasses import dataclass

import sympy as sp
import torch


@dataclass
class Graph:
    variables: tuple[str, ...]
    x: torch.Tensor
    edge_index: torch.Tensor


def _expression(text: str) -> sp.Expr:
    tree = ast.parse(text.strip().replace("^", "**"), mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name) and not node.id.startswith("_"):
            return sp.Symbol(node.id)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return sp.Rational(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div) and not right.free_symbols and right != 0:
                return left / right
            if isinstance(node.op, ast.Pow) and right.is_Integer and 0 <= right <= 10000:
                return left ** right
        raise ValueError("Only polynomial arithmetic with numeric coefficients is supported")

    return visit(tree)


def polynomials(text: str) -> list[sp.Poly]:
    text = text.strip()
    if text.startswith(("[", "{")):
        if not text.endswith("]" if text[0] == "[" else "}"):
            raise ValueError("Unclosed polynomial list")
        text = text[1:-1]
    if not text.strip():
        return []
    expressions = {}
    for item in text.split(","):
        expression = sp.expand(_expression(item))
        if expression.free_symbols:
            poly = sp.Poly(expression)
            expressions[str(poly.as_expr())] = poly
    return [expressions[key] for key in sorted(expressions)]


def graph_from_text(text: str, *, normalize: bool = True) -> Graph:
    polys = polynomials(text)
    names = tuple(sorted({str(v) for p in polys for v in p.gens}))
    index = {name: i for i, name in enumerate(names)}
    rows = []
    terms_by_poly = []
    vars_by_poly = []
    for poly in polys:
        terms = []
        for powers, _coefficient in poly.terms():
            terms.append({str(v): degree for v, degree in zip(poly.gens, powers) if degree})
        terms_by_poly.append(terms)
        vars_by_poly.append(sorted({v for term in terms for v in term}))
    all_terms = sum(map(len, terms_by_poly))
    for variable in names:
        f = [0.0] * 14
        neighbours = set()
        for variables, terms in zip(vars_by_poly, terms_by_poly):
            present = [term for term in terms if variable in term]
            if not present:
                continue
            neighbours.update(set(variables) - {variable})
            degree = max(term[variable] for term in present)
            leading_degree = max(sum(term.values()) - degree for term in present if term[variable] == degree)
            totals = [sum(term.values()) for term in present]
            f[1] += 1
            f[2] = max(f[2], degree)
            f[3] += degree
            f[4] = max(f[4], leading_degree)
            f[5] = max(f[5], len(present))
            f[6] = max(f[6], max(totals))
            f[7] += sum(totals)
            f[8] += leading_degree
            f[9] += len(present)
            f[12] = max(f[12], max(len(term) - 1 for term in present))
            f[13] = max(f[13], len(variables) - 1)
        f[0] = len(neighbours)
        f[10] = f[1] / len(polys)
        f[11] = f[9] / all_terms
        rows.append(f)
    x = torch.tensor(rows, dtype=torch.float32).reshape(-1, 14)
    if normalize and len(names):
        if len(names) == 1:
            x = x / (x.abs().amax(dim=0) + 1e-7)
        else:
            x = (x - x.mean(dim=0)) / (x.std(dim=0, correction=1) + 1e-7)
    edges, seen = [], set()
    for variables in vars_by_poly:
        for a, b in itertools.combinations(variables, 2):
            if (a, b) not in seen:
                seen.add((a, b))
                edges.extend(((index[a], index[b]), (index[b], index[a])))
    edge_index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()
    return Graph(names, x, edge_index)
