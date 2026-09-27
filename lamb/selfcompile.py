"""Self-compilation: the model reads its own emitted program off itself and runs the
executor-only version as an exact, cheaper copy of its neural core. See ROADMAP 3g.

The register machine's pointers index register *slots*, not values, so the program the
neural core emits for one problem is the program for *every* problem of that structure
(CLAUDE.md: one pointer pattern across 300 depth-3 problems, only the operators varying).
The emitted program is therefore an operand-invariant, exact specification of the core's
behaviour on the whole task class -- a 1:1 copy that costs one algebra execution (~6.5% of
a training step) instead of a transformer forward. Compiling the core is the model reading
its own algorithm off itself: run it once per structure, record the argmax program, and
the library of programs plus the exact executor reproduces the core with the neural net
removed. The copy's *content* -- the algorithm -- is the model's; this module is the
caching harness that runs it, and is careful never to substitute the gold program for the
model's own (the point is to copy what the model does, not what it should do).

Nothing here is assumed. :func:`compile_model` records how many *distinct* programs each
structure needed (1 == fully compilable), and :func:`verify` reports the fraction of
held-out problems where the compiled copy reproduces the core's *own* output exactly, not
just the fraction that are correct -- a compiled copy that is right where the core is right
and wrong where the core is wrong is a faithful copy, which is the claim being tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .alu import parse_expr
from .regmachine import run_program

Instr = Tuple[int, int, int]
Pattern = Tuple[str, ...]


def op_pattern(tree) -> Pattern:
    """Post-order operator signature of an expression tree.

    The structural key the compiled copy looks a program up by, and it is readable from
    the *input* by parsing -- no neural net. A leaf ``(op, a, b)`` contributes its
    operator; an internal node contributes its children's then its own. For a balanced
    depth-2 tree ``(a o1 b) o0 (c o2 d)`` this is ``(o1, o2, o0)``, which with the fixed
    slot-indexed pointer structure determines the whole program.
    """
    if isinstance(tree[1], int):
        return (tree[0],)
    return op_pattern(tree[1]) + op_pattern(tree[2]) + (tree[0],)


@dataclass
class CompiledModel:
    """A neural core compiled to a program library: an exact, executor-only copy.

    ``machine`` is the exact executor (:class:`RegisterMachine`), reused for its algebra
    and register file -- *not* the neural core, which is exactly what compilation removes.
    ``prepare`` is the teacher's ``_prepare`` bound method, reused only for the fixed
    digit->residue operand loading (a known non-neural map, ROADMAP 3a-vi), never for its
    gold programs.
    """

    library: Dict[Pattern, List[Instr]]
    n_operands: int
    n_instr: int
    out_reg: int
    machine: object
    prepare: object
    depth: int = 0
    rational: bool = False
    variants: Dict[Pattern, int] = field(default_factory=dict)

    def covers(self, expr: str) -> bool:
        return op_pattern(parse_expr(expr)) in self.library

    def _decode(self, regs, index: int) -> List[Optional[Fraction]]:
        if self.rational:
            n = self.machine.sys.decode(self.machine.alg.unpack(regs.num[:, index]))
            d = self.machine.sys.decode(self.machine.alg.unpack(regs.den[:, index]))
            return [None if dd == 0 else Fraction(nn, dd) for nn, dd in zip(n, d)]
        return [Fraction(v) for v in regs.decode(index)]

    @torch.no_grad()
    def solve(self, tasks: Sequence[Tuple[str, str, list]]
              ) -> Tuple[List[Optional[Fraction]], List[bool]]:
        """Answer each task with the library program for its structure, no neural core.

        Returns ``(answers, covered)``. An uncovered structure (never seen at compile
        time) has no program; it is marked ``covered=False`` and its answer is left as a
        no-op program's output rather than silently borrowing the core -- a copy that
        quietly falls back to the original is not a measurement of the copy.
        """
        prep = self.prepare(list(tasks))
        vals = prep[2]
        progs, covered = [], []
        noop = [(0, 0, 0)] * self.n_instr
        for t in tasks:
            p = self.library.get(op_pattern(parse_expr(t[0])))
            covered.append(p is not None)
            progs.append(p if p is not None else noop)
        regs = run_program(self.machine, vals, progs, "cpu")
        return self._decode(regs, self.out_reg), covered


@torch.no_grad()
def _argmax_programs(rmt, tasks) -> Tuple[List[List[Instr]], list, list, list]:
    """The core's own argmax program per task (slot-indexed), with vals/answers/keep."""
    _, _, vals, answers, keep = rmt._prepare(list(tasks))
    latent_h = rmt._latents(list(tasks))
    cnt = torch.tensor(rmt._counts, device=rmt.device)
    op_l, a_l, b_l = rmt.machine.logits(latent_h, cnt)
    oi, ai, bi = op_l.argmax(-1), a_l.argmax(-1), b_l.argmax(-1)
    progs = [[(int(oi[i, t]), int(ai[i, t]), int(bi[i, t]))
              for t in range(rmt.n_instr)] for i in range(len(tasks))]
    return progs, vals, answers, keep


@torch.no_grad()
def compile_model(rmt, n_samples: int = 2048, batch: int = 256,
                  seed: int = 0) -> CompiledModel:
    """Read the core's algorithm off itself into a program library.

    Runs the core on ``n_samples`` problems from its own training distribution and records
    its argmax program per operator pattern. ``variants`` counts distinct programs seen per
    pattern; a pattern with more than one is *not* fully compiled by a single program and
    is a signal the emission is not operand-invariant there. The first program seen wins,
    which is exact wherever ``variants == 1``.
    """
    from collections import defaultdict

    library: Dict[Pattern, List[Instr]] = {}
    variants: Dict[Pattern, set] = defaultdict(set)
    seen = 0
    while seen < n_samples:
        tasks = rmt.inner._sample_batch(min(batch, n_samples - seen))
        seen += len(tasks)
        progs, _, _, _ = _argmax_programs(rmt, tasks)
        for task, prog in zip(tasks, progs):
            pat = op_pattern(parse_expr(task[0]))
            variants[pat].add(tuple(prog))
            library.setdefault(pat, prog)
    return CompiledModel(
        library=library, n_operands=rmt.n_operands, n_instr=rmt.n_instr,
        out_reg=rmt.n_operands + rmt.n_instr - 1, machine=rmt.machine,
        prepare=rmt._prepare, depth=int(rmt.cfg.depth),
        rational=bool(getattr(rmt, "rational", False)),
        variants={p: len(v) for p, v in variants.items()})


def tree_depth(t) -> int:
    """Nesting depth of an expression tree; a leaf ``(op, a, b)`` is depth 1."""
    if isinstance(t[1], int):
        return 1
    return 1 + max(tree_depth(t[1]), tree_depth(t[2]))


def tree_to_expr(t) -> str:
    """Serialise a subtree back to the grammar's own string form, so a compiled copy
    can parse it and load its operands with the same fixed map the core uses."""
    if isinstance(t[1], int):
        return f"{t[1]}{t[0]}{t[2]}"
    return f"({tree_to_expr(t[1])}){t[0]}({tree_to_expr(t[2])})"


def solve_hierarchical(copies: Dict[int, CompiledModel], expr: str
                       ) -> Tuple[Optional[Fraction], int]:
    """Solve a task deeper than any single copy, using only cheaper copies of the model.

    This is the multi-agent step (ROADMAP 3g): a task too deep for one copy is decomposed,
    each subtree within a copy's depth is solved by that copy, and two sub-results are
    combined by a *depth-1 copy* solving ``"v_left op v_right"`` -- not by Python
    arithmetic, so every step of the reasoning is a cheap 1:1 copy of the model. It works
    for values of any magnitude because the copies' programs are operand-invariant (their
    pointers are structural), which is the same property that made compilation exact.

    Returns ``(value, n_calls)`` where ``n_calls`` counts copy invocations. The routing
    here is structural, not learned; the model emitting its own dispatch program is 3g's
    pre-registered next step. Requires a depth-1 copy for composition and, for efficiency,
    the deepest copy available at or below each subtree's depth.
    """
    from .alu import parse_expr

    one = copies.get(1)
    if one is None:
        raise ValueError("hierarchical solving needs a depth-1 copy for composition")
    depths = sorted(copies)
    calls = [0]

    def best_copy_for(d: int) -> Optional[CompiledModel]:
        usable = [k for k in depths if k <= d]
        return copies[max(usable)] if usable else None

    def rec(t) -> Optional[Fraction]:
        d = tree_depth(t)
        c = best_copy_for(d)
        if c is not None and c.depth >= d and c.covers(tree_to_expr(t)):
            calls[0] += 1
            vals, cov = c.solve([(tree_to_expr(t), "0", [])])
            if cov[0] and vals[0] is not None:
                return vals[0]
        if isinstance(t[1], int):                 # a leaf no copy covered: unsolved
            return None
        lv, rv = rec(t[1]), rec(t[2])
        if lv is None or rv is None:
            return None
        calls[0] += 1                              # compose via a depth-1 copy
        vals, cov = one.solve([(f"{lv}{t[0]}{rv}", "0", [])])
        return vals[0] if cov[0] else None

    return rec(parse_expr(expr)), calls[0]


@torch.no_grad()
def verify(rmt, compiled: CompiledModel, n: int = 512, seed: int = 0) -> Dict[str, float]:
    """1:1 fidelity of the compiled copy against the core, on held-out problems.

    ``fidelity`` is the fraction where the compiled copy reproduces the core's *own*
    decoded output exactly -- the copy claim. ``core_acc``/``compiled_acc`` are each
    against the true answer, and ``coverage`` is the fraction of held-out structures the
    library knew. Reported together because a faithful copy of a fallible core should
    track the core, not the truth.
    """
    held = rmt.inner._eval_set(n)
    core_progs, vals, answers, keep = _argmax_programs(rmt, held)
    core_regs = run_program(rmt.machine, vals, core_progs, str(rmt.device))
    out_reg = rmt.n_operands + rmt.n_instr - 1
    core_out = compiled._decode(core_regs, out_reg)
    comp_out, covered = compiled.solve(held)

    n_h = len(held)
    fidelity = sum(1 for c, m in zip(comp_out, core_out) if c == m) / n_h
    core_acc = sum(1 for o, a in zip(core_out, answers) if o == Fraction(int(a))) / n_h
    comp_acc = sum(1 for o, a in zip(comp_out, answers) if o == Fraction(int(a))) / n_h
    max_variant = max(compiled.variants.values()) if compiled.variants else 0
    return {"fidelity": fidelity, "core_acc": core_acc, "compiled_acc": comp_acc,
            "coverage": sum(covered) / n_h, "n_patterns": float(len(compiled.library)),
            "max_variants": float(max_variant)}
