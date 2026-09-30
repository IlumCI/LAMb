"""The core emits its own dispatch program. ROADMAP 3g, Phase B.

Phase A (:mod:`lamb.selfcompile`) made the decomposition of a deep task an explicit
program, a list of ``dispatch`` and ``combine`` steps run entirely by cheap compiled
copies of the model. There the program was written by :func:`gold_dispatch_program`.
Here the *core* writes it.

**The decision.** At every internal node of the expression tree the model chooses:
dispatch this whole subtree to one of its copies, or split it and combine the children.
Leaves are always dispatched (a depth-1 copy always exists). Applied top down, those
votes determine a unique dispatch program, which the exact executor runs. So the program
is the model's; the executor only walks the tree applying its votes, the way the residue
algebra only executes the ``(op, ptr, ptr)`` the register machine emits.

**Why the choice is real.** Dispatching high is cheap but only valid if a copy covers the
subtree; splitting is always valid but costs a call per node. The copies here cover every
subtree of depth <= 2 and only *balanced* subtrees of depth 3 (the depth-3 copy was
compiled on balanced trees). So the cost-optimal program needs the model to know what its
own copies can do: an unbalanced depth-3 subtree must be split, a balanced one should not
be. That is the meta-understanding the self-copy idea turns on, made into a measurable
decision rather than asserted.

**Reading a node.** The core is causal, so a node's operator token cannot see its right
subtree. Each node is read at the *end* of its span, where the whole subtree has been seen,
concatenated with the state at the span's start.

**Four training signals, none an external label.** ``probe``: the model asks its copies
about every subtree -- "would you accept this?" -- and fits the answers (binary
cross-entropy). The target is the copies' own competence, read from their libraries, not a
teacher's program; applied top down it is also the cost-optimal rule. ``outcome``: decisions
are sampled, the whole program is scored by success and call count, and a group-relative
REINFORCE step is taken. ``feedback``: each dispatch actually attempted while solving is
rewarded by whether its copy accepted it. ``experience``: the same accept/refuse
observations from its own hand-offs, used as prediction targets instead of rewards -- the
one that learns the self-model from experience alone (ROADMAP 3g). Coverage stands in for executing during training,
which is exact because the copies are (3g fidelity 1.000); :meth:`DispatchTrainer.evaluate`
checks it by executing every program (``oracle_agree``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .selfcompile import (CompiledModel, execute_dispatch_program, op_pattern,
                          tree_depth, tree_to_expr)


@dataclass
class Node:
    """An expression-tree node with the character span it occupies in the source."""

    op: str
    start: int
    end: int
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    a: Optional[int] = None
    b: Optional[int] = None

    def is_leaf(self) -> bool:
        return self.left is None

    def tree(self):
        if self.is_leaf():
            return (self.op, self.a, self.b)
        return (self.op, self.left.tree(), self.right.tree())


def parse_nodes(expr: str, off: int = 0) -> Node:
    """Parse the grammar's string form, keeping each node's character span.

    Mirrors :func:`lamb.alu.parse_expr` exactly, so the tree is the same one every other
    component sees. ``start``/``end`` index into the full source string.
    """
    end = off + len(expr) - 1
    if expr.startswith("("):
        depth = 0
        for i, c in enumerate(expr):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
        return Node(op=expr[i + 1], start=off, end=end,
                    left=parse_nodes(expr[1:i], off + 1),
                    right=parse_nodes(expr[i + 3:-1], off + i + 3))
    for j, c in enumerate(expr):
        if c in "+-*/" and j > 0:
            return Node(op=c, start=off, end=end, a=int(expr[:j]), b=int(expr[j + 1:]))
    raise ValueError(f"cannot parse {expr!r}")


def internal_nodes(root: Node) -> List[Node]:
    """Internal nodes in pre-order: the nodes where a decision is actually made."""
    if root.is_leaf():
        return []
    return [root] + internal_nodes(root.left) + internal_nodes(root.right)


def covering_depth(node: Node, copies: Dict[int, CompiledModel]) -> Optional[int]:
    """The copy depth that can solve this subtree whole, or ``None`` if none can."""
    t = node.tree()
    d = tree_depth(t)
    c = copies.get(d)
    return d if (c is not None and op_pattern(t) in c.library) else None


def emit_program(root: Node, decide: Callable[[Node], bool]) -> list:
    """Turn per-node dispatch votes into a dispatch program, top down."""
    prog: list = []

    def rec(n: Node) -> int:
        if n.is_leaf() or decide(n):
            t = n.tree()
            prog.append(("dispatch", tree_to_expr(t), tree_depth(t)))
            return len(prog) - 1
        li, ri = rec(n.left), rec(n.right)
        prog.append(("combine", n.op, li, ri))
        return len(prog) - 1

    rec(root)
    return prog


def program_outcome(root: Node, decide: Callable[[Node], bool],
                    copies: Dict[int, CompiledModel]) -> Tuple[bool, int, List[Node]]:
    """``(succeeds, calls, decided_nodes)`` without executing.

    Succeeds iff every dispatched subtree is covered by a copy; exact because the copies
    are. ``decided_nodes`` are the internal nodes whose vote the program actually used,
    which are the only ones a policy gradient should credit.
    """
    calls, ok, used = 0, True, []

    def rec(n: Node) -> None:
        nonlocal calls, ok
        if not n.is_leaf():
            used.append(n)
        if n.is_leaf() or decide(n):
            calls += 1
            if covering_depth(n, copies) is None:
                ok = False
            return
        rec(n.left)
        rec(n.right)
        calls += 1

    rec(root)
    return ok, calls, used


def gold_decide(copies: Dict[int, CompiledModel]) -> Callable[[Node], bool]:
    """The cost-optimal rule: dispatch wherever a copy covers the subtree."""
    return lambda n: covering_depth(n, copies) is not None


class DispatchPolicy(nn.Module):
    """A LAMb core with a dispatch head read at each node's span.

    The head sees ``[h_end, h_start]``: the state after the whole subtree has been read,
    and the state where the subtree began, so a causal core can in principle recover the
    subtree's extent and shape from the pair.
    """

    def __init__(self, model_cfg, tokenizer):
        super().__init__()
        from .model.lamb import build_model

        self.tok = tokenizer
        self.model = build_model(model_cfg, tokenizer)
        d = model_cfg.d_model
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def hidden(self, exprs: Sequence[str], device: str) -> torch.Tensor:
        encs = [self.tok.encode_prompt(e) for e in exprs]
        t = max(len(e.ids) for e in encs)
        b = len(encs)
        ids = torch.full((b, t), self.tok.PAD, dtype=torch.long)
        ab = torch.zeros((b, t), dtype=torch.long)
        val = torch.zeros((b, t))
        vm = torch.zeros((b, t))
        pad = torch.ones((b, t), dtype=torch.bool)
        for i, e in enumerate(encs):
            n = len(e.ids)
            ids[i, :n] = torch.tensor(e.ids)
            ab[i, :n] = torch.tensor(e.abacus)
            val[i, :n] = torch.tensor(e.value)
            vm[i, :n] = torch.tensor(e.value_mask)
            pad[i, :n] = False
        dev = torch.device(device)
        x = self.model.embed(ids.to(dev), ab.to(dev), val.to(dev), vm.to(dev))
        h, _ = self.model.core(x, pad.to(dev))
        return h

    def node_logits(self, exprs: Sequence[str], roots: Sequence[Node],
                    device: str) -> List[torch.Tensor]:
        """Per-row tensor of dispatch logits, aligned with :func:`internal_nodes`."""
        h = self.hidden(exprs, device)
        out = []
        for i, root in enumerate(roots):
            nodes = internal_nodes(root)
            if not nodes:
                out.append(torch.zeros(0, device=h.device))
                continue
            # +1: token 0 is BOS, then one token per source character.
            ends = torch.tensor([n.end + 1 for n in nodes], device=h.device)
            starts = torch.tensor([n.start + 1 for n in nodes], device=h.device)
            feat = torch.cat([h[i, ends], h[i, starts]], dim=-1)
            out.append(self.head(feat).squeeze(-1))
        return out


class DispatchTrainer:
    """Train the core to emit dispatch programs, supervised or from outcome alone."""

    def __init__(self, copies: Dict[int, CompiledModel], model_cfg, tokenizer,
                 depth: int = 4, digits: int = 2, mode: str = "probe",
                 lr: float = 1e-3, batch_size: int = 32, group: int = 4,
                 call_cost: float = 0.5, explore: float = 0.2, seed: int = 0,
                 device: str = "cpu"):
        from .selfplay.grammar import Descriptor, TaskGrammar

        if mode not in ("probe", "outcome", "feedback", "experience"):
            raise ValueError(f"unknown mode {mode!r}")
        torch.manual_seed(seed)
        self.copies = copies
        self.mode = mode
        self.device = device
        self.batch_size = batch_size
        self.group = group
        self.call_cost = call_cost
        self.explore = explore
        self.grammar = TaskGrammar()
        # shape=1: depth is a ceiling, so training sees a mix of shapes and depths, which
        # is what makes "is this subtree one my copies cover" a question with two answers.
        self.desc = Descriptor(depth, digits, 0, 1)
        self.digits = digits
        self.policy = DispatchPolicy(model_cfg, tokenizer).to(device)
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=0.01)
        self._seed = seed * 1_000_003 + 11

    def _batch(self, n: int) -> List[str]:
        out = []
        while len(out) < n:
            self._seed += 1
            e, _ = self.grammar.sample(self.desc, self._seed, exclude_heldout=True)
            if not parse_nodes(e).is_leaf():
                out.append(e)
        return out

    def train_step(self, step: int) -> Dict[str, float]:
        self.policy.train()
        exprs = self._batch(self.batch_size)
        roots = [parse_nodes(e) for e in exprs]
        logits = self.policy.node_logits(exprs, roots, self.device)
        if self.mode == "probe":
            ys = [torch.tensor([1.0 if covering_depth(n, self.copies) is not None else 0.0
                                for n in internal_nodes(r)], device=self.device)
                  for r in roots]
            lg, y = torch.cat(logits), torch.cat(ys)
            loss = F.binary_cross_entropy_with_logits(lg, y)
            stats = {"acc": float(((lg > 0).float() == y).float().mean())}
        elif self.mode == "outcome":
            loss, stats = self._outcome_loss(roots, logits)
        elif self.mode == "feedback":
            loss, stats = self._feedback_loss(roots, logits)
        else:
            loss, stats = self._experience_loss(roots, logits)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        stats["loss"] = float(loss.detach())
        return stats

    def _outcome_loss(self, roots, logits):
        """Group-relative REINFORCE on sampled votes; reward = success minus call cost."""
        terms, rewards_all, succ_all = [], [], []
        for root, lg in zip(roots, logits):
            nodes = internal_nodes(root)
            index = {id(n): k for k, n in enumerate(nodes)}
            n_nodes = 2 * len(nodes) + 1                      # calls of always-split
            # Epsilon floor on the vote distribution, scored under that same mixture.
            # Without it the policy collapses to always-split: splitting is always valid,
            # early high dispatches often fail, so the dispatch probability runs to zero,
            # every sample in a group becomes identical, and the group-relative advantage
            # vanishes (measured: reward pinned at exactly 0.500, i.e. always-split calls,
            # by step 200). The same collapse, and the same fix, as program_grpo.
            probs = (1 - self.explore) * torch.sigmoid(lg) + self.explore * 0.5
            group_r, group_lp = [], []
            for _ in range(self.group):
                votes = torch.bernoulli(probs.detach())
                ok, calls, used = program_outcome(
                    root, lambda n: bool(votes[index[id(n)]]), self.copies)
                r = (1.0 - self.call_cost * calls / n_nodes) if ok else 0.0
                idx = torch.tensor([index[id(n)] for n in used], device=lg.device)
                v, p = votes[idx], probs[idx]
                lp = (v * torch.log(p.clamp_min(1e-6))
                      + (1 - v) * torch.log((1 - p).clamp_min(1e-6))).sum()
                group_r.append(r)
                group_lp.append(lp)
                succ_all.append(float(ok))
            base = sum(group_r) / len(group_r)
            for r, lp in zip(group_r, group_lp):
                terms.append(-(r - base) * lp)
            rewards_all.extend(group_r)
        loss = torch.stack(terms).mean()
        return loss, {"reward": sum(rewards_all) / len(rewards_all),
                      "success": sum(succ_all) / len(succ_all)}

    def _feedback_loss(self, roots, logits):
        """REINFORCE on each dispatch the model actually attempts, rewarded by whether the
        copy it sent the work to accepted it.

        Program-level reward failed (outcome mode): one bad vote zeroes the whole program,
        so a small group cannot tell which vote was the mistake, and the policy either
        stays at always-split or over-reaches. But a dispatch has an observable consequence
        of its own -- the copy either solves the subtree or refuses (``covered=False`` from
        :meth:`CompiledModel.solve`) -- so each attempt carries its own credit: +1 accepted,
        -1 refused. Splits earn 0, so a node's dispatch probability moves only on evidence
        from trying it. No labels: the model learns what its copies can do by sending them
        work, which is the meta-knowledge the dispatch decision needs. Acceptance is read
        from the copy's library, which is exactly what the executor consults
        (``oracle_agree`` in :meth:`evaluate` checks it by executing).
        """
        terms, accepts, attempts = [], 0, 0
        for root, lg in zip(roots, logits):
            nodes = internal_nodes(root)
            index = {id(n): k for k, n in enumerate(nodes)}
            probs = (1 - self.explore) * torch.sigmoid(lg) + self.explore * 0.5
            for _ in range(self.group):
                votes = torch.bernoulli(probs.detach())
                _, _, used = program_outcome(
                    root, lambda n: bool(votes[index[id(n)]]), self.copies)
                for n in used:
                    k = index[id(n)]
                    if votes[k] < 0.5:
                        continue
                    ok = covering_depth(n, self.copies) is not None
                    attempts += 1
                    accepts += ok
                    terms.append(-(1.0 if ok else -1.0) * torch.log(probs[k].clamp_min(1e-6)))
        loss = (torch.stack(terms).mean() if terms
                else sum(lg.sum() for lg in logits) * 0.0)
        return loss, {"accept_rate": accepts / max(1, attempts),
                      "attempts": attempts / max(1, len(roots) * self.group)}

    def _experience_loss(self, roots, logits):
        """Learn a self-model from the hand-offs actually made while solving.

        The accept/refuse a copy returns is an *observation*, not just a reward. Feedback
        mode pushed it through REINFORCE, which keeps only its sign relative to a zero
        baseline and pays policy-gradient variance for it. Here it is a prediction target
        instead: the dispatch head is trained, by binary cross-entropy, to predict whether
        its copy will accept a subtree, on exactly the subtrees it chose to send while
        solving (sampled with an epsilon floor so uncertain ones keep getting tried). The
        policy is then "dispatch where acceptance is predicted". Still experience alone:
        a node the model never tried contributes nothing, unlike ``probe``, which asks the
        copies about every subtree of every problem.
        """
        feats, targets, attempts, accepts = [], [], 0, 0
        for root, lg in zip(roots, logits):
            nodes = internal_nodes(root)
            index = {id(n): k for k, n in enumerate(nodes)}
            probs = ((1 - self.explore) * torch.sigmoid(lg) + self.explore * 0.5).detach()
            tried: Dict[int, float] = {}
            for _ in range(self.group):
                votes = torch.bernoulli(probs)
                _, _, used = program_outcome(
                    root, lambda n: bool(votes[index[id(n)]]), self.copies)
                for n in used:
                    k = index[id(n)]
                    if votes[k] > 0.5 and k not in tried:
                        tried[k] = 1.0 if covering_depth(n, self.copies) is not None else 0.0
            for k, y in tried.items():
                feats.append(lg[k])
                targets.append(y)
            attempts += len(tried)
            accepts += sum(tried.values())
        if not feats:
            return sum(lg.sum() for lg in logits) * 0.0, {"accept_rate": 0.0, "tried": 0.0}
        lg_all = torch.stack(feats)
        y_all = torch.tensor(targets, device=lg_all.device)
        loss = F.binary_cross_entropy_with_logits(lg_all, y_all)
        return loss, {"accept_rate": accepts / max(1, attempts),
                      "tried": attempts / max(1, len(roots))}

    @torch.no_grad()
    def category_rates(self, n: int = 200, depth: Optional[int] = None,
                       seed: int = 999) -> Dict[str, Tuple[float, int]]:
        """Dispatch-vote rate per node category on held-out trees, ``(rate, count)``.

        Categories are what the copies can and cannot take: depth 2 (always covered),
        depth 3 balanced (covered), depth 3 unbalanced (refused), depth >= 4 (refused).
        The correct policy is 1, 1, 0, 0; this shows *where* a policy is wrong, which the
        single vote accuracy hides.
        """
        from .selfplay.grammar import Descriptor

        self.policy.eval()
        desc = Descriptor(depth or self.desc.depth, self.digits, 0, 1)
        rng = random.Random(seed)
        exprs = []
        while len(exprs) < n:
            e, _ = self.grammar.sample_heldout(desc, rng.randint(0, 2 ** 31 - 1))
            if not parse_nodes(e).is_leaf():
                exprs.append(e)
        roots = [parse_nodes(e) for e in exprs]
        logits = self.policy.node_logits(exprs, roots, self.device)
        tally: Dict[str, List[int]] = {}
        for root, lg in zip(roots, logits):
            for k, nd in enumerate(internal_nodes(root)):
                d = tree_depth(nd.tree())
                if d == 3:
                    cat = "d3_bal" if covering_depth(nd, self.copies) else "d3_unbal"
                else:
                    cat = f"d{min(d, 4)}{'+' if d >= 4 else ''}"
                tally.setdefault(cat, []).append(int(lg[k] > 0))
        return {c: (sum(v) / len(v), len(v)) for c, v in sorted(tally.items())}

    @torch.no_grad()
    def evaluate(self, n: int = 200, depth: Optional[int] = None,
                 seed: int = 12345) -> Dict[str, float]:
        """Emit, *execute*, and score programs on held-out problems.

        Every program is run by :func:`execute_dispatch_program` with the real copies, so
        the success figure is measured, not inferred from coverage. Reported beside it:
        the calls the gold (cost-optimal) program and the always-split program would make,
        and ``oracle_agree``, whether coverage predicted execution correctly on each row.
        """
        from .selfplay.grammar import Descriptor

        self.policy.eval()
        desc = Descriptor(depth or self.desc.depth, self.digits, 0, 1)
        rng = random.Random(seed)
        exprs, answers = [], []
        while len(exprs) < n:
            e, a = self.grammar.sample_heldout(desc, rng.randint(0, 2 ** 31 - 1))
            if not parse_nodes(e).is_leaf():
                exprs.append(e)
                answers.append(a)
        roots = [parse_nodes(e) for e in exprs]
        logits = self.policy.node_logits(exprs, roots, self.device)
        gold = gold_decide(self.copies)
        correct = calls = gold_calls = split_calls = agree = vote_ok = votes = 0
        for root, lg, ans in zip(roots, logits, answers):
            nodes = internal_nodes(root)
            index = {id(nd): k for k, nd in enumerate(nodes)}
            decide = lambda nd, lg=lg, index=index: bool(lg[index[id(nd)]] > 0)
            prog = emit_program(root, decide)
            got, c = execute_dispatch_program(prog, self.copies)
            ok_exec = got is not None and got == Fraction(int(ans))
            ok_pred, c_pred, _ = program_outcome(root, decide, self.copies)
            correct += ok_exec
            calls += c
            agree += (ok_exec == ok_pred)
            gold_calls += program_outcome(root, gold, self.copies)[1]
            split_calls += program_outcome(root, lambda nd: False, self.copies)[1]
            for nd in nodes:
                votes += 1
                vote_ok += (decide(nd) == gold(nd))
        return {"accuracy": correct / n, "calls": calls / n, "gold_calls": gold_calls / n,
                "split_calls": split_calls / n, "vote_acc": vote_ok / max(1, votes),
                "oracle_agree": agree / n}


def build_gold_copies(depths_shapes: Sequence[Tuple[int, int]], n: int = 3000,
                      digits: int = 2) -> Dict[int, CompiledModel]:
    """Copies whose libraries hold the gold program per structure.

    Equivalent to compiling a trained core wherever the core is competent (3g measured
    fidelity 1.000 at depths 2 and 3), and far cheaper than training one per depth here,
    where the question is the dispatch policy, not the copies. ``(depth, shape)`` pairs
    let a copy cover only balanced trees, which is what makes the policy's choice real.
    """
    from . import ArithmeticTokenizer, LotusConfig
    from .alu import parse_expr
    from .config import ModelConfig
    from .regmachine import RegMachineTrainer

    copies: Dict[int, CompiledModel] = {}
    for depth, shape in depths_shapes:
        cfg = LotusConfig(steps=1, batch_size=64, n_latent=max(8, 2 ** depth), loops=1,
                          depth=depth, digits=digits, shape=shape, trace_coef=0.0,
                          alu_coef=0.0, use_boundaries=False, switch_coef=0.0,
                          device="cpu", alu_moduli=(16, 25, 27, 11, 37))
        t = RegMachineTrainer(cfg, ArithmeticTokenizer(),
                              ModelConfig(d_model=16, n_heads=2, d_ff=32,
                                          recurrent_steps=1))
        tasks = t.inner._sample_batch(n)
        _, golds, _, _, _ = t._prepare(tasks)
        lib = {}
        for task, g in zip(tasks, golds):
            lib.setdefault(op_pattern(parse_expr(task[0])), g)
        copies[depth] = CompiledModel(
            library=lib, n_operands=t.n_operands, n_instr=t.n_instr,
            out_reg=t.n_operands + t.n_instr - 1, machine=t.machine, prepare=t._prepare,
            depth=depth)
    return copies
