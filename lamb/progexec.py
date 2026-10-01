"""Run and verify compiled LAMb programs with nothing but the standard library.

A compiled LAMb copy (``lamb.selfcompile``, ``lamb.engine``) is a library of straight-line
register programs, one per problem structure. Executing one needs no neural network, no
torch and no GPU: load the operands into registers, apply each ``(op, ptr_a, ptr_b)``
instruction exactly with Python ``Fraction`` arithmetic, read the output register. That is
what this file does, deliberately dependency-free, so that anything that only has to *check*
work -- a verifier node, a test harness, another language binding -- can do so cheaply and
deterministically. The residue algebra the model trains through is exact within its ring,
so it and this executor agree; ``tests/test_engine.py`` holds them to it.

Library format (JSON), written by ``python -m lamb.engine compile``::

    {"format": 1,
     "spec": {"ops": ["+", "-", "*"], "n_operands": 9, "n_const": 1,
              "constants": [1], "n_instr": 7},
     "programs": {"<structure key>": {"depth": 2, "program": [[op, a, b], ...]}}}

Register layout, identical to ``RegMachineTrainer._prepare``: the constants first, then the
expression's literal operands left to right, zero-padded to ``n_operands``; result ``t`` of the
program is written to register ``n_operands + t``; the answer is the last instruction's result.
"""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple, Union

Tree = tuple
Program = Sequence[Sequence[int]]


# -- the grammar's string form ------------------------------------------------
def parse(expr: str) -> Tree:
    """Parse ``a<op>b`` / ``(L)<op>(R)`` into ``(op, a, b)`` / ``(op, L, R)``.

    The same rules as :func:`lamb.alu.parse_expr`; restated here so this file needs nothing
    else, and pinned to it by a test, because a condition restated in two places will
    otherwise disagree in one of them.
    """
    expr = expr.strip()
    if expr.startswith("("):
        depth = 0
        for i, c in enumerate(expr):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
        return (expr[i + 1], parse(expr[1:i]), parse(expr[i + 3:-1]))
    for j, c in enumerate(expr):
        if c in "+-*/" and j > 0:
            return (c, int(expr[:j]), int(expr[j + 1:]))
    raise ValueError(f"cannot parse {expr!r}")


def is_leaf(t: Tree) -> bool:
    return isinstance(t[1], int)


def depth(t: Tree) -> int:
    return 1 if is_leaf(t) else 1 + max(depth(t[1]), depth(t[2]))


def leaves(t: Tree) -> List[int]:
    """Literal operands left to right -- the initial register file."""
    return [t[1], t[2]] if is_leaf(t) else leaves(t[1]) + leaves(t[2])


def structure(t: Tree):
    """Operators and shape, no operands: the key a compiled program is filed under."""
    return (t[0],) if is_leaf(t) else (t[0], structure(t[1]), structure(t[2]))


def key(t: Tree) -> str:
    """A string form of :func:`structure`, stable across processes and languages."""
    return json.dumps(structure(t), separators=(",", ":"))


def render(t: Tree) -> str:
    """The grammar's string form of a tree (inverse of :func:`parse`)."""
    if is_leaf(t):
        return f"{t[1]}{t[0]}{t[2]}"
    return f"({render(t[1])}){t[0]}({render(t[2])})"


def evaluate(t: Tree) -> Fraction:
    """The exact value of an expression -- the reference a verifier checks against."""
    if is_leaf(t):
        a, b = Fraction(t[1]), Fraction(t[2])
    else:
        a, b = evaluate(t[1]), evaluate(t[2])
    return _apply(t[0], a, b)


# -- execution ----------------------------------------------------------------
def _apply(op: str, a: Fraction, b: Fraction) -> Fraction:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            raise ZeroDivisionError("program divides by zero")
        return a / b
    raise ValueError(f"unknown operator {op!r}")


def run(program: Program, operands: Sequence[Union[int, Fraction]], spec: Dict) -> Fraction:
    """Execute a program exactly; returns the last instruction's result.

    Raises ``ValueError`` on a pointer that reads a register not yet written -- the
    machine's causal mask makes that unrepresentable in a model's output, so seeing one
    here means the program was tampered with or built for a different layout.
    """
    n_op = spec["n_operands"]
    consts = [Fraction(c) for c in spec.get("constants", [1] * spec.get("n_const", 0))]
    regs = consts + [Fraction(v) for v in operands]
    if len(regs) > n_op:
        raise ValueError(f"{len(regs)} operands do not fit {n_op} registers")
    regs += [Fraction(0)] * (n_op - len(regs))
    ops = spec["ops"]
    for t, (o, a, b) in enumerate(program):
        if not (0 <= a < len(regs) and 0 <= b < len(regs)):
            raise ValueError(f"instruction {t} reads an unwritten register ({a}, {b})")
        regs.append(_apply(ops[o], regs[a], regs[b]))
    return regs[-1]


# -- library use ----------------------------------------------------------------
def load(path: str) -> Dict:
    with open(path) as f:
        lib = json.load(f)
    if lib.get("format") != 1:
        raise ValueError(f"unknown library format {lib.get('format')!r}")
    return lib


def lookup(lib: Dict, expr: str) -> Tuple[Optional[Program], Tree]:
    t = parse(expr)
    entry = lib["programs"].get(key(t))
    return (entry["program"] if entry else None), t


def solve(lib: Dict, expr: str) -> Optional[Fraction]:
    """Answer with the compiled program for this structure, or ``None`` if there is none.

    ``None`` is a refusal, never a guess: a structure the copy was not compiled for is
    work it cannot do, and returning anything else would be a number nobody computed.
    """
    prog, t = lookup(lib, expr)
    if prog is None:
        return None
    return run(prog, leaves(t), lib["spec"])


def verify(lib_or_spec: Dict, expr: str, program: Program,
           claimed: Union[int, str, Fraction]) -> bool:
    """Check a claimed answer for ``expr`` produced by ``program``.

    True iff the program runs, its result equals ``claimed``, and both equal the exact value
    of the expression. Costs one straight-line execution and one exact evaluation -- no
    model -- which is what makes checking LAMb's work cheap relative to producing it.
    """
    spec = lib_or_spec.get("spec", lib_or_spec)
    t = parse(expr)
    try:
        got = run(program, leaves(t), spec)
    except (ValueError, ZeroDivisionError, IndexError):
        return False
    claim = Fraction(claimed)
    return got == claim and claim == evaluate(t)


def main(argv: Optional[Sequence[str]] = None) -> int:          # pragma: no cover
    """``lamb-verify LIBRARY.json EXPR [EXPR ...]`` -- solve with compiled programs only;
    ``lamb-verify LIBRARY.json --check EXPR PROGRAM_JSON ANSWER`` -- verify a claimed result."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) >= 5 and args[1] == "--check":
        ok = verify(load(args[0]), args[2], json.loads(args[3]), args[4])
        print("valid" if ok else "INVALID")
        return 0 if ok else 1
    if len(args) < 2:
        print("usage: lamb-verify LIBRARY.json EXPR [EXPR ...]\n"
              "       lamb-verify LIBRARY.json --check EXPR PROGRAM_JSON ANSWER")
        return 2
    library = load(args[0])
    for e in args[1:]:
        v = solve(library, e)
        print(f"{e} = {v}" if v is not None else f"{e}: no compiled program (refused)")
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
