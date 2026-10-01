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

    Core-backed copies are created per depth as needed and registered at ``copies[depth]``
    (never replacing a copy that already exists there); each starts empty and refuses
    everything until something is compiled into it. ``forwards`` counts core forward passes
    spent compiling, the cost side of every decision.

    ``told_reach=True`` (the default, and how Phase C ran): only work of exactly the core's
    depth is considered -- the model is *told* its core's reach. ``False`` tells it nothing:
    any work no copy takes is a candidate, and the reach has to be found out by trying. An instance the core
    cannot even load (more operands or instructions than its machine holds) is a failed
    attempt at full cost, the same as one it loads and gets wrong -- the compiler does not
    get to consult the core's architecture for free. ``charge_rejections=False`` makes
    failed attempts free, i.e. perfect foreknowledge of reach: a reference, not a policy.
    """

    def __init__(self, core, copies: Dict[int, CompiledModel], k: int = 3,
                 forward_cost: float = 4.0, told_reach: bool = True,
                 charge_rejections: bool = True):
        self.core = core
        self.copies = copies
        self.k = k
        self.forward_cost = forward_cost
        self.depth = int(core.cfg.depth)
        self.scope: Optional[int] = self.depth if told_reach else None
        self.charge_rejections = charge_rejections
        self.own: Dict[int, CompiledModel] = {}
        self.copy = self._copy_for(self.depth)             # the copy Phase C grows
        self.count: Dict[tuple, int] = {}
        self.saving: Dict[tuple, int] = {}
        self.buffer: Dict[tuple, List[Tuple[str, object]]] = {}
        self.depth_of: Dict[tuple, int] = {}
        self.first_seen: Dict[tuple, int] = {}
        self.rejected: set = set()
        self.forwards = 0
        self.wasted = 0
        self.adopted = 0
        self.now = 0

    def _copy_for(self, depth: int) -> CompiledModel:
        if depth not in self.own:
            c = CompiledModel(
                library={}, n_operands=self.core.n_operands, n_instr=self.core.n_instr,
                out_reg=self.core.n_operands + self.core.n_instr - 1,
                machine=self.core.machine, prepare=self.core._prepare, depth=depth)
            self.own[depth] = c
            self.copies.setdefault(depth, c)
        return self.own[depth]

    def compiled(self, key) -> bool:
        return any(key in c.library for c in self.own.values())

    def observe(self, split_nodes: Sequence[Node]) -> None:
        """Log work the model did the slow way that no copy takes."""
        slow = effective(lambda _: True, self.copies)       # its trusted slow path
        for n in split_nodes:
            t = n.tree()
            d = tree_depth(t)
            if (self.scope is not None and d != self.scope) or \
                    covering_depth(n, self.copies) is not None:
                continue
            key = op_pattern(t)
            if self.compiled(key) or key in self.rejected:
                continue
            self.count[key] = self.count.get(key, 0) + 1
            self.depth_of[key] = d
            self.first_seen.setdefault(key, self.now)
            buf = self.buffer.setdefault(key, [])
            if len(buf) < self.k:
                val, calls = execute_dispatch_program(emit_program(n, slow), self.copies)
                self.saving.setdefault(key, calls - 1)
                buf.append((tree_to_expr(t), val))

    def _loadable(self, expr: str) -> bool:
        from .alu import parse_expr
        from .regmachine import gold_program, operands

        t = parse_expr(expr)
        return (len(operands(t)) + self.core.n_const <= self.core.n_operands
                and len(gold_program(t, ops=self.core.machine.ops)[0]) <= self.core.n_instr)

    @torch.no_grad()
    def attempt(self, key) -> str:
        """Run the core on the logged instances; adopt only a consistent, correct program."""
        buf = self.buffer[key]
        cost = len(buf)
        ok = False
        if all(self._loadable(e) for e, _ in buf):
            tasks = [(e, "0", []) for e, _ in buf]
            progs, vals, _, _ = _argmax_programs(self.core, tasks)
            regs = run_program(self.core.machine, vals, [progs[0]] * len(tasks),
                               str(self.core.device))
            got = self.copy._decode(regs, self.copy.out_reg)
            ok = (all(p == progs[0] for p in progs)
                  and all(g == v for g, (_, v) in zip(got, buf)))
        if ok:
            self.forwards += cost
            self._copy_for(self.depth_of[key]).library[key] = progs[0]
            self.adopted += 1
            return "adopted"
        if self.charge_rejections:
            self.forwards += cost
            self.wasted += cost
        self.rejected.add(key)
        return "rejected"

    def candidates(self) -> List[tuple]:
        return [k for k in self.count if not self.compiled(k) and k not in self.rejected
                and len(self.buffer.get(k, [])) >= self.k]

    def maybe_compile(self, now: int = 0, total: int = 0) -> List[Tuple[tuple, str]]:
        """The fixed rule: compile once the logged history alone pays for the check."""
        self.now = now
        return [(key, self.attempt(key)) for key in self.candidates()
                if self.count[key] * self.saving[key] >= self.k * self.forward_cost]


class LearnedCompiler(CoreCompiler):
    """The compile decision from learned predictions instead of fixed assumptions.

    The fixed rule assumes every compile succeeds and waits until past history alone pays
    for the check. Here both are predicted from the model's own representation of the
    structure (``featurize``: the routing core's state for the subtree, detached):

    * a **reach head** -- will my core back this? -- trained by cross-entropy on the
      verified outcome of every attempt it has made, generalising to structures never tried;
    * a **rate head** -- how often will this recur per round? -- trained by Poisson
      likelihood on the occurrences it has logged, with each key's exposure since first seen.

    It attempts when ``p_backed * rate * saving * rounds_remaining > k * forward_cost``, plus
    an ``explore`` chance of trying a candidate the prediction rejects so the reach head keeps
    getting evidence. The form is expected value; the inputs are learned. The horizon
    (rounds remaining) is a known operational quantity, not learned.
    """

    def __init__(self, core, copies, featurize, feat_dim: int, k: int = 3,
                 forward_cost: float = 4.0, explore: float = 0.05, lr: float = 3e-3,
                 fit_sample: int = 256, seed: int = 0):
        super().__init__(core, copies, k=k, forward_cost=forward_cost, told_reach=False)
        import torch.nn as nn

        self.featurize = featurize
        self.explore = explore
        self.fit_sample = fit_sample
        self.rng = random.Random(seed)
        self.reach = nn.Sequential(nn.Linear(feat_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.rate = nn.Sequential(nn.Linear(feat_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.opt = torch.optim.Adam(list(self.reach.parameters())
                                    + list(self.rate.parameters()), lr=lr)
        self.outcome: Dict[tuple, float] = {}
        self.example: Dict[tuple, str] = {}

    def observe(self, split_nodes: Sequence[Node]) -> None:
        super().observe(split_nodes)
        for key, buf in self.buffer.items():
            if buf:
                self.example.setdefault(key, buf[0][0])

    def _fit(self, keys: List[tuple], feats: torch.Tensor, steps: int = 3) -> None:
        idx = {k: i for i, k in enumerate(keys)}
        exposure = torch.tensor([max(1, self.now - self.first_seen[k] + 1) for k in keys],
                                dtype=torch.float32)
        counts = torch.tensor([float(self.count.get(k, 0)) for k in keys])
        tried = [k for k in keys if k in self.outcome]
        for _ in range(steps):
            log_rate = self.rate(feats).squeeze(-1)
            mu = torch.exp(log_rate) * exposure
            loss = (mu - counts * torch.log(mu.clamp_min(1e-8))).mean()
            if tried:
                ti = torch.tensor([idx[k] for k in tried])
                y = torch.tensor([self.outcome[k] for k in tried])
                loss = loss + F.binary_cross_entropy_with_logits(
                    self.reach(feats[ti]).squeeze(-1), y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

    def predict(self, keys: List[tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.featurize([self.example[k] for k in keys]).detach()
        with torch.no_grad():
            p = torch.sigmoid(self.reach(feats).squeeze(-1))
            rate = torch.exp(self.rate(feats).squeeze(-1))
        return p, rate

    def maybe_compile(self, now: int = 0, total: int = 0) -> List[Tuple[tuple, str]]:
        self.now = now
        cands = self.candidates()
        # Train on what matters each round -- every tried key (reach), every candidate, and a
        # capped random sample of the rest (rate). Re-featurising every key ever logged made
        # the cost and memory grow with the log (measured: ~1.9 GB and slowing, when blind
        # logging admits thousands of one-off deep structures).
        keys = [k for k in self.count if k in self.example]
        if not keys:
            return []
        must = [k for k in keys if k in self.outcome or k in set(cands)]
        rest = [k for k in keys if k not in self.outcome and k not in set(cands)]
        keys = must + self.rng.sample(rest, min(len(rest), self.fit_sample))
        feats = self.featurize([self.example[k] for k in keys]).detach()
        self._fit(keys, feats)
        if not cands:
            return []
        p, rate = self.predict(cands)
        remaining = max(0, total - now)
        events = []
        for key, pk, rk in zip(cands, p.tolist(), rate.tolist()):
            ev = pk * rk * self.saving[key] * remaining - self.k * self.forward_cost
            if ev > 0 or self.rng.random() < self.explore:
                status = self.attempt(key)
                self.outcome[key] = 1.0 if status == "adopted" else 0.0
                events.append((key, status))
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
        self.rounds_done = 0
        self.total_rounds = 0           # set by the caller: the known length of the work

    @torch.no_grad()
    def featurize(self, exprs: List[str]) -> torch.Tensor:
        """The routing core's own state for a standalone subtree: ``[h_end, h_start]`` of its
        root, the same readout the dispatch head uses. Handed to a learned compiler so its
        predictions are about the model's representation, not hand-built features."""
        self.policy.eval()
        h = self.policy.hidden(exprs, self.device)
        ends = torch.tensor([len(e) for e in exprs])            # +1 BOS, -1 last char
        return torch.cat([h[torch.arange(len(exprs)), ends], h[:, 1]], dim=-1).cpu()

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
            events = self.compiler.maybe_compile(self.rounds_done, self.total_rounds)
        self.stream_calls += calls
        self.stream_tasks += len(exprs)
        self.rounds_done += 1
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
                if tree_depth(nd.tree()) >= 3 and covering_depth(nd, self.copies):
                    d3_tot += 1
                    d3_hit += decide(nd)
        return {"calls": calls / n, "accuracy": correct / n, "best_now": best / n,
                "ideal": ideal / n,
                "routes_to_compiled": d3_hit / max(1, d3_tot)}
