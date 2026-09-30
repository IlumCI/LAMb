"""The model decides when to compile a new copy of itself. ROADMAP 3g, Phase C.

The pieces so far: a core compiles itself into exact copies (:mod:`lamb.selfcompile`), and a
core learns from its own hand-offs what those copies accept and routes work across them
(:mod:`lamb.dispatch`, ``experience`` mode). What remained a harness step was *which* copies
exist. Here the model grows them while it works:

1. It routes each task with its learned self-model. A hand-off a copy refuses falls back to
   splitting, so misrouting costs calls, never correctness, and exploring is safe.
2. Every subtree it had to do the slow way, that its core could have done in one shot, is
   logged by structure key together with the value it obtained through its trusted copies.
3. When a key has recurred enough that a copy would pay for itself, it compiles one from its
   own neural core -- runs the core on the logged instances -- and adopts the program only if
   the core emitted the *same* program on every instance (operand-invariance, the property
   that makes a copy exact) and that program reproduces the values it already had. The new
   copy is checked against the model's own decomposition; nothing external is consulted.
4. Routing keeps learning online, and has to discover that the copy now takes that work.

The trigger is a fixed expected-value rule, stated plainly because it is the one part that is
not learned: compile when ``seen * saved_per_occurrence >= k * forward_cost``, i.e. once the
recorded history alone would have paid for the ``k`` core forwards the check costs. Everything
the rule reads is the model's own experience.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .dispatch import (DispatchPolicy, Node, covering_depth, emit_program, internal_nodes,
                       parse_nodes)
from .regmachine import run_program
from .selfcompile import (CompiledModel, _argmax_programs, execute_dispatch_program,
                          op_pattern, tree_depth, tree_to_expr)


def run_with_fallback(root: Node, decide: Callable[[Node], bool],
                      copies: Dict[int, CompiledModel]
                      ) -> Tuple[int, List[Tuple[Node, bool]], List[Node]]:
    """Deploy the votes; a refused hand-off is paid for and then split.

    Returns ``(calls, handoffs, splits)``: copy calls spent (a refused attempt costs one),
    every attempted hand-off with whether it was accepted, and every node that ended up
    split. Leaves always go to the depth-1 copy.
    """
    calls, handoffs, splits = 0, [], []

    def rec(n: Node) -> None:
        nonlocal calls
        if n.is_leaf():
            calls += 1
            return
        if decide(n):
            calls += 1
            ok = covering_depth(n, copies) is not None
            handoffs.append((n, ok))
            if ok:
                return
        splits.append(n)
        rec(n.left)
        rec(n.right)
        calls += 1

    rec(root)
    return calls, handoffs, splits


def effective(decide: Callable[[Node], bool],
              copies: Dict[int, CompiledModel]) -> Callable[[Node], bool]:
    """The program that actually ran: a vote to dispatch that was refused became a split."""
    return lambda n: decide(n) and covering_depth(n, copies) is not None


def ideal_calls(root: Node, max_depth: int) -> int:
    """Calls if every subtree within the core's reach had a copy -- the floor compilation
    can approach."""
    if root.is_leaf() or tree_depth(root.tree()) <= max_depth:
        return 1
    return ideal_calls(root.left, max_depth) + ideal_calls(root.right, max_depth) + 1


class CoreCompiler:
    """Decides, from logged experience, when to compile a copy from the core, and checks it.

    The copy it grows is registered at ``copies[depth]`` and starts empty, so it refuses
    everything until something is compiled into it. ``forwards`` counts core forward passes
    spent compiling, the cost side of the trigger.
    """

    def __init__(self, core, copies: Dict[int, CompiledModel], k: int = 3,
                 forward_cost: float = 4.0):
        self.core = core
        self.copies = copies
        self.k = k
        self.forward_cost = forward_cost
        self.depth = int(core.cfg.depth)
        self.copy = CompiledModel(
            library={}, n_operands=core.n_operands, n_instr=core.n_instr,
            out_reg=core.n_operands + core.n_instr - 1, machine=core.machine,
            prepare=core._prepare, depth=self.depth)
        copies[self.depth] = self.copy
        self.count: Dict[tuple, int] = {}
        self.saving: Dict[tuple, int] = {}
        self.buffer: Dict[tuple, List[Tuple[str, object]]] = {}
        self.rejected: set = set()
        self.forwards = 0
        self.adopted = 0

    def observe(self, split_nodes: Sequence[Node]) -> None:
        """Log subtrees the model had to split but its core could have done in one shot."""
        slow = effective(lambda _: True, self.copies)       # its trusted slow path
        for n in split_nodes:
            t = n.tree()
            if tree_depth(t) != self.depth:
                continue
            key = op_pattern(t)
            if key in self.copy.library or key in self.rejected:
                continue
            self.count[key] = self.count.get(key, 0) + 1
            buf = self.buffer.setdefault(key, [])
            if len(buf) < self.k:
                val, calls = execute_dispatch_program(emit_program(n, slow), self.copies)
                self.saving.setdefault(key, calls - 1)
                buf.append((tree_to_expr(t), val))

    @torch.no_grad()
    def maybe_compile(self) -> List[Tuple[tuple, str]]:
        """Compile every key whose logged history pays for its check; verify before adopting."""
        events = []
        for key, seen in list(self.count.items()):
            if key in self.copy.library or key in self.rejected:
                continue
            buf = self.buffer.get(key, [])
            if len(buf) < self.k or seen * self.saving[key] < self.k * self.forward_cost:
                continue
            tasks = [(e, "0", []) for e, _ in buf]
            progs, vals, _, _ = _argmax_programs(self.core, tasks)
            self.forwards += len(tasks)
            consistent = all(p == progs[0] for p in progs)
            regs = run_program(self.core.machine, vals, [progs[0]] * len(tasks),
                               str(self.core.device))
            got = self.copy._decode(regs, self.copy.out_reg)
            agrees = all(g == v for g, (_, v) in zip(got, buf))
            if consistent and agrees:
                self.copy.library[key] = progs[0]
                self.adopted += 1
                events.append((key, "adopted"))
            else:
                self.rejected.add(key)
                events.append((key, "rejected"))
        return events


class SelfImprovingAgent:
    """Routes a task stream with a learned self-model, growing copies as it goes.

    ``compiler=None`` never compiles (the static baseline); a compiler whose copy was filled
    in advance is the up-front baseline. Routing learns online in experience mode from the
    hand-offs it actually makes, with an epsilon floor on its votes.
    """

    def __init__(self, copies: Dict[int, CompiledModel], model_cfg, tokenizer,
                 compiler: Optional[CoreCompiler] = None, depth: int = 5, digits: int = 2,
                 explore: float = 0.2, lr: float = 1e-3, batch_size: int = 32,
                 core_depth: int = 3, seed: int = 0, device: str = "cpu"):
        from .selfplay.grammar import Descriptor, TaskGrammar

        torch.manual_seed(seed)
        self.copies = copies
        self.compiler = compiler
        self.explore = explore
        self.batch_size = batch_size
        self.core_depth = core_depth
        self.device = device
        self.grammar = TaskGrammar()
        self.desc = Descriptor(depth, digits, 0, 1)
        self.policy = DispatchPolicy(model_cfg, tokenizer).to(device)
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=0.01)
        self._seed = seed * 1_000_003 + 17
        self.stream_calls = 0
        self.stream_tasks = 0

    def _batch(self, n: int) -> List[str]:
        out = []
        while len(out) < n:
            self._seed += 1
            e, _ = self.grammar.sample(self.desc, self._seed, exclude_heldout=True)
            if not parse_nodes(e).is_leaf():
                out.append(e)
        return out

    def round(self) -> Dict[str, float]:
        """One batch of real work: route, pay, learn from the hand-offs, maybe compile."""
        self.policy.train()
        exprs = self._batch(self.batch_size)
        roots = [parse_nodes(e) for e in exprs]
        logits = self.policy.node_logits(exprs, roots, self.device)
        feats, targets, splits_all, calls = [], [], [], 0
        for root, lg in zip(roots, logits):
            index = {id(n): k for k, n in enumerate(internal_nodes(root))}
            probs = ((1 - self.explore) * torch.sigmoid(lg) + self.explore * 0.5).detach()
            votes = torch.bernoulli(probs)
            c, handoffs, splits = run_with_fallback(
                root, lambda n: bool(votes[index[id(n)]]), self.copies)
            calls += c
            for n, ok in handoffs:
                feats.append(lg[index[id(n)]])
                targets.append(1.0 if ok else 0.0)
            splits_all.extend(splits)
        if feats:
            loss = F.binary_cross_entropy_with_logits(
                torch.stack(feats), torch.tensor(targets, device=feats[0].device))
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
        events = []
        if self.compiler is not None:
            self.compiler.observe(splits_all)
            events = self.compiler.maybe_compile()
        self.stream_calls += calls
        self.stream_tasks += len(exprs)
        return {"calls": calls / len(exprs),
                "adopted": float(sum(1 for _, s in events if s == "adopted")),
                "rejected": float(sum(1 for _, s in events if s == "rejected"))}

    @torch.no_grad()
    def evaluate(self, n: int = 100, seed: int = 4242) -> Dict[str, float]:
        """Greedy routing on held-out tasks; every program is executed and checked."""
        from fractions import Fraction

        self.policy.eval()
        rng = random.Random(seed)
        exprs, answers = [], []
        while len(exprs) < n:
            e, a = self.grammar.sample_heldout(self.desc, rng.randint(0, 2 ** 31 - 1))
            if not parse_nodes(e).is_leaf():
                exprs.append(e)
                answers.append(a)
        roots = [parse_nodes(e) for e in exprs]
        logits = self.policy.node_logits(exprs, roots, self.device)
        calls = correct = best = ideal = d3_hit = d3_tot = 0
        for root, lg, ans in zip(roots, logits, answers):
            index = {id(nd): k for k, nd in enumerate(internal_nodes(root))}
            decide = lambda nd, lg=lg, index=index: bool(lg[index[id(nd)]] > 0)
            c, _, _ = run_with_fallback(root, decide, self.copies)
            got, _ = execute_dispatch_program(
                emit_program(root, effective(decide, self.copies)), self.copies)
            calls += c
            correct += got is not None and got == Fraction(int(ans))
            best += run_with_fallback(root, lambda nd: covering_depth(nd, self.copies)
                                      is not None, self.copies)[0]
            ideal += ideal_calls(root, self.core_depth)
            for nd in internal_nodes(root):
                if tree_depth(nd.tree()) == self.core_depth and covering_depth(nd, self.copies):
                    d3_tot += 1
                    d3_hit += decide(nd)
        return {"calls": calls / n, "accuracy": correct / n, "best_now": best / n,
                "ideal": ideal / n,
                "routes_to_compiled": d3_hit / max(1, d3_tot)}
