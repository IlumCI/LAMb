"""GRPO over sampled register *programs*, with a curriculum bandit. See ROADMAP 3e.

Distinct from :mod:`lamb.selfplay.grpo`, which runs GRPO over *answer tokens* for the
autoregressive solver (and the hypernetwork proposer). This module runs GRPO over the
register machine's emitted program -- the ``(op, ptr, ptr)`` heads -- so its action space
is a program, not a token sequence. The group-relative advantage and DAPO dynamic-sampling
ideas are the same; :mod:`lamb.selfplay.grpo` holds the scalar ``group_advantages`` and
``dynamic_keep_mask`` primitives, and the ``(G, B)``-matrix forms here are their vectorised
equivalents for the per-instruction program case.

The answer-only arm of 3a-xii learns a program by backpropagating a dense answer
loss through a *soft* program: a mixture over operations and a mixture over
registers. 3a-x measured that such a mixture decodes to a value in neither register
~30% of the time, so that gradient is aimed at a soft object with no executable
meaning -- biased, not merely noisy. This module asks the same question a different
way: sample *hard, valid* programs, execute each exactly, and learn by policy
gradient on the reward. Nothing differentiable flows through the algebra, so the
soft-read pathology cannot arise; only real programs are ever scored.

The learning rule is GRPO (arXiv:2402.03300): for one problem, sample a group of ``G``
programs, score each by the exact executor, and use the within-group
``(r - mean) / std`` as the advantage. No value network, no reward model -- the
verifier is the arithmetic itself. Its known failure is advantage collapse
(arXiv:2605.21125): when every program in a group is wrong the advantage is zero and
there is no gradient, which at depth 3 is every group. Three things answer that, and
each is free here in a way it is not for a text model:

* **DAPO dynamic sampling** (arXiv:2503.14476): a group whose reward has no variance
  contributes nothing, so it is dropped and its rate reported (ACR).
* **Process reward from the gold trace**: the grammar hands over every sub-expression
  value, so a program that computes a right intermediate is credited before it ever
  gets the final answer right. Order-free, because conformity is not competence
  (3a-xii). A shaping term, not the objective -- an intermediate can be hit by luck.
* **A curriculum bandit**: arms are difficulties, the per-arm signal is learnability
  ``p*(1-p)`` which peaks at the frontier (arXiv:2502.12272), tracked as a
  non-stationary EMA and sampled Boltzmann (Self-Evolving Curriculum,
  arXiv:2505.14970). The bandit keeps training where groups are still informative.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .holdout import is_heldout
from .regmachine import RegMachineTrainer, run_program
from .selfplay.grammar import Descriptor, TaskGrammar


class CurriculumBandit:
    """A non-stationary bandit over task difficulties.

    Each arm is a difficulty (here a depth). The reward the bandit maximises is not
    accuracy -- a fully-solved arm teaches nothing and a never-solved arm teaches
    nothing either. It is *learnability*, the Bernoulli variance ``p*(1-p)`` of the
    group success rate, which is largest at ``p = 0.5``: the frontier where a program
    is hard but reachable and the GRPO advantage still has variance to work with
    (arXiv:2502.12272). The signal drifts as the model improves, so it is an EMA
    rather than a running mean, and arms are drawn Boltzmann over it with an epsilon
    floor so a temporarily-cold arm can be revisited (arXiv:2505.14970).
    """

    def __init__(self, arms: Sequence[int], ema: float = 0.85,
                 temp: float = 0.1, eps: float = 0.15, mastery: float = 0.8,
                 seed: int = 0):
        self.arms = list(arms)
        self.n = len(self.arms)
        self.ema = float(ema)
        self.temp = float(temp)
        self.eps = float(eps)
        self.mastery = float(mastery)
        self.val = [0.0] * self.n          # learnability EMA per arm
        self.succ = [0.0] * self.n         # success-rate EMA per arm
        self.seen = [0] * self.n           # times each arm was drawn
        self.rng = random.Random(seed)

    def _mastered(self, i: int) -> bool:
        return self.seen[i] > 0 and self.succ[i] >= self.mastery

    def sample(self) -> int:
        """Index of the arm to train on next.

        Learnability ``p*(1-p)`` alone cannot tell a *mastered* arm from a *hopeless*
        one -- both read ~0 -- and the smoke run showed the bandit then splitting its
        draws evenly and re-collapsing the shared policy on the solved easy arm. So the
        curriculum is explicit about the frontier: mastered arms are set aside, and the
        shallowest not-yet-mastered arm gets a bonus so training pushes forward rather
        than dithering among equally-uninformative arms.
        """
        for i in range(self.n):
            if self.seen[i] == 0:
                return i
        if self.rng.random() < self.eps:
            return self.rng.randrange(self.n)
        frontier = next((i for i in range(self.n) if not self._mastered(i)), self.n - 1)
        score = []
        for i in range(self.n):
            s = 0.0 if self._mastered(i) else self.val[i]
            if i == frontier:
                s += 0.05                     # push the frontier forward, not sideways
            score.append(s)
        m = max(score)
        w = [math.exp((v - m) / max(1e-6, self.temp)) for v in score]
        tot = sum(w)
        r = self.rng.random() * tot
        acc = 0.0
        for i, wi in enumerate(w):
            acc += wi
            if r <= acc:
                return i
        return self.n - 1

    def update(self, i: int, learnability: float, success: float = 0.0) -> None:
        self.seen[i] += 1
        self.val[i] = self.ema * self.val[i] + (1.0 - self.ema) * float(learnability)
        self.succ[i] = self.ema * self.succ[i] + (1.0 - self.ema) * float(success)

    def state(self) -> Dict[int, Dict[str, float]]:
        return {self.arms[i]: {"val": self.val[i], "succ": self.succ[i],
                               "seen": self.seen[i]} for i in range(self.n)}


class GRPOTrainer:
    """Train the latent core to *write* programs by outcome, sampling not blending.

    Wraps a :class:`RegMachineTrainer` sized for the deepest arm and reuses its latent
    core, program heads, register machine and optimiser wholesale. What is new is the
    loop: draw a difficulty from the bandit, sample a group of hard programs per
    problem, execute them exactly, and take a REINFORCE step on the group-relative
    advantage. The machine is over-provisioned (identity register on) so a shorter
    program is padded rather than unrepresentable, which is what lets one machine span
    the curriculum -- exactly the mechanism 3a-xiv built.
    """

    def __init__(self, cfg, tokenizer, depths: Sequence[int], model_cfg=None,
                 group: int = 8, process_coef: float = 0.25,
                 entropy_coef: float = 0.0, entropy_floor: float = 0.5,
                 entropy_anneal: Optional[int] = None,
                 explore_eps: float = 0.3, explore_eps_min: float = 0.02,
                 explore_anneal: Optional[int] = 400, calib_coef: float = 0.0,
                 mode: str = "grpo", replay: int = 4000, rft_verify: bool = True,
                 compose: bool = False,
                 bandit: bool = True, bandit_ema: float = 0.85,
                 bandit_temp: float = 0.1, bandit_eps: float = 0.15,
                 rational: bool = False, digits: Optional[int] = None,
                 ops_key: Optional[int] = None, seed: int = 0):
        self.depths = sorted(int(d) for d in depths)
        self.d_max = max(self.depths)
        if cfg.depth != self.d_max:
            raise ValueError(f"cfg.depth={cfg.depth} must equal max curriculum depth "
                             f"{self.d_max}; the machine is sized for the deepest arm")
        # ``shape=1`` is set on the *wrapped* config for one reason: it forces the
        # identity register on (``n_const=1``), which every below-max depth needs to
        # pad ``x * 1`` up to the fixed instruction budget. It does not touch the data
        # this trainer draws -- every batch here is sampled from an explicit descriptor
        # below, never from the wrapped trainer's own sampler.
        prev_shape = getattr(cfg, "shape", 0)
        try:
            cfg.shape = 1
            self.rmt = RegMachineTrainer(cfg, tokenizer, model_cfg,
                                         program_coef=0.0, answer_coef=0.0,
                                         rational=rational)
        finally:
            cfg.shape = prev_shape
        self.cfg = cfg
        self.device = self.rmt.device
        self.machine = self.rmt.machine
        self.rational = rational
        self.group = int(group)
        self.process_coef = float(process_coef)
        self.entropy_coef = float(entropy_coef)
        self.entropy_floor = float(entropy_floor)
        self.entropy_anneal = int(entropy_anneal) if entropy_anneal else 0
        self.explore_eps = float(explore_eps)
        self.explore_eps_min = float(explore_eps_min)
        self.explore_anneal = int(explore_anneal) if explore_anneal else 0
        self.calib_coef = float(calib_coef)
        if mode not in ("grpo", "rft"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.replay = int(replay)
        self.buffer: Dict[int, list] = {d: [] for d in depths}
        self._replay_rng = random.Random(seed + 1)
        self.rft_verify = rft_verify
        self._coincidences = 0
        self.compose = compose
        self._composed = 0
        self.digits = cfg.digits if digits is None else int(digits)
        self.ops_key = cfg.ops_key if ops_key is None else int(ops_key)
        self.grammar = TaskGrammar()
        self.opt = self.rmt.opt
        self.n_instr = self.rmt.n_instr
        self.n_operands = self.rmt.n_operands
        self.bandit = (CurriculumBandit(self.depths, ema=bandit_ema, temp=bandit_temp,
                                        eps=bandit_eps, seed=seed) if bandit else None)
        self._fixed = self.depths[-1]                    # arm when bandit is off
        self._succ = {d: 0.0 for d in self.depths}       # per-arm argmax-success EMA
        self._arm_steps = {d: 0 for d in self.depths}    # per-arm training steps, for eps
        self._seed = seed * 1_000_003 + 7

    # -- data --------------------------------------------------------------
    def _descriptor(self, depth: int) -> Descriptor:
        return Descriptor(depth, self.digits, self.ops_key, 0)

    def _out_reg(self, depth: int) -> int:
        """The register a depth-``d`` balanced program naturally terminates in.

        The machine is sized for the deepest arm, so its instruction budget is
        ``n_instr`` for every arm. Reading the answer from the *last* budget slot would
        force a shallow program to identity-route its result forward through the spare
        instructions -- and discovering that routing is itself a hard program that RL
        stalled on (depth 1 topped out at 0.59, the bandit never graduated). A depth-d
        balanced tree has ``2^d - 1`` instructions, so its answer lands in result slot
        ``2^d - 2`` (zero-indexed) after the operand block. The curriculum knows the
        depth, so it reads there and the spare instructions are simply unused. Supervised
        training pads with ``x*1`` because its loss needs a full-length gold program;
        outcome training has no such need.
        """
        return self.n_operands + (2 ** depth - 1) - 1

    def _sample_batch(self, depth: int, n: int) -> List[Tuple[str, str, List[int]]]:
        desc = self._descriptor(depth)
        out = []
        for _ in range(n):
            self._seed += 1
            out.append(self.grammar.sample_with_trace(desc, self._seed,
                                                      exclude_heldout=True))
        return out

    # -- reward ------------------------------------------------------------
    def _decode_answers(self, regs, index: int) -> List[Optional[Fraction]]:
        if self.rational:
            n = self.machine.sys.decode(self.machine.alg.unpack(regs.num[:, index]))
            d = self.machine.sys.decode(self.machine.alg.unpack(regs.den[:, index]))
            return [None if dd == 0 else Fraction(nn, dd) for nn, dd in zip(n, d)]
        return [Fraction(v) for v in regs.decode(index)]

    @torch.no_grad()
    def _reward(self, vals, programs, answers, traces, out_reg: int
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute one group of hard programs; return correctness and process credit.

        ``correct`` is 1 iff the decoded output equals the exact answer *at the output
        register* -- the only term that is the objective. ``process`` is the fraction of
        a problem's gold intermediate values *and its final answer* that appear anywhere
        in the executed register file. Including the answer is what gives a gradient
        ladder: produce the right value in some register (process), then route it to the
        output (outcome). Without it a depth-1 problem has an empty gold trace and so no
        shaping at all, and the smoke run could not consolidate it. It is order-free (48
        programs compute the same value) and a fraction, so it cannot outweigh a correct
        answer at ``process_coef <= 1``.
        """
        regs = run_program(self.machine, vals, programs, str(self.device))
        got = self._decode_answers(regs, out_reg)
        correct = torch.zeros(len(answers), device=self.device)
        process = torch.zeros(len(answers), device=self.device)
        # Only the result slots this depth's program *uses* count for process credit,
        # i.e. up to ``out_reg``. Scanning the full instruction budget was a real bug: at
        # a shallow depth the spare instructions are untrained and random, so when one of
        # them coincidentally produced the answer, the high process reward was
        # misattributed to the (trained) early instructions -- reinforcing whatever they
        # happened to sample. Measured as the argmax program oscillating and never
        # consolidating. Credit only the slots whose log-probability is in the loss.
        results = [self._decode_answers(regs, s)
                   for s in range(self.n_operands, out_reg + 1)]
        for i, a in enumerate(answers):
            fa = Fraction(int(a))
            g = got[i]
            correct[i] = 1.0 if (g is not None and g == fa) else 0.0
            if self.process_coef:
                produced = {results[s][i] for s in range(len(results))
                            if results[s][i] is not None}
                want = {Fraction(int(v)) for v in traces[i]} | {fa}
                process[i] = len(produced & want) / len(want)
        return correct, process

    # -- one batch ---------------------------------------------------------
    def _entropy(self, logits) -> torch.Tensor:
        tot = torch.zeros((), device=self.device)
        for x in logits:                                 # op, ptr-a, ptr-b
            lp = torch.log_softmax(x, dim=-1).clamp_min(-30.0)
            tot = tot + (-(lp.exp() * lp).sum(-1)).mean()
        return tot / 3.0

    def _program_ce(self, logits, progs, n_used: int) -> torch.Tensor:
        """Cross-entropy of the core's heads against given programs, first ``n_used``
        instructions only. Pointers in a sampled program were drawn from the legal set, so
        no target ever lands on a masked (-inf) slot."""
        dev = logits[0].device
        total = torch.zeros((), device=dev)
        for k, lg in enumerate(logits):                      # op, ptr-a, ptr-b
            tgt = torch.tensor([[p[t][k] for t in range(n_used)] for p in progs], device=dev)
            lp = torch.log_softmax(lg[:, :n_used], dim=-1)
            total = total - lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()
        return total / 3.0

    def stitch(self, left, right, root_op: str, depth: int) -> list:
        """Join two depth-(d-1) programs into one depth-d program in the full layout.

        Left half as-is; the right half's operand pointers move past the left's leaves and
        its result pointers past the left's instructions (the constant register stays put);
        then one instruction combines the two halves' last results with ``root_op``.
        """
        k, n_op = self.rmt.n_const, self.n_operands
        c_instr, c_leaves = 2 ** (depth - 1) - 1, 2 ** (depth - 1)

        def shifted(ptr):
            if ptr < k:
                return ptr                                  # the constant register
            if ptr < n_op:
                return ptr + c_leaves                       # operand of the right half
            return ptr + c_instr                            # result of the right half

        prog = [tuple(ins) for ins in left[:c_instr]]
        prog += [(o, shifted(a), shifted(b)) for o, a, b in right[:c_instr]]
        prog.append((self.machine.ops.index(root_op), n_op + c_instr - 1,
                     n_op + 2 * c_instr - 1))
        while len(prog) < self.n_instr:
            prog.append((0, 0, 0))
        return prog

    @torch.no_grad()
    def _compose(self, tasks, hits, depth: int):
        """Write a candidate depth-d program from the model's own depth-(d-1) programs.

        Sampling a 7-instruction program blind did not find a single verified depth-3
        program in 1500 steps, though the model was solid at depth 2. A balanced depth-d
        tree is two depth-(d-1) trees joined by one operator, so for each problem still
        without a hit the model is run on its own two halves, and their argmax programs are
        stitched into the full layout -- left half as-is, right half's operand pointers
        shifted past the left's leaves and its result pointers past the left's
        instructions -- plus one instruction combining the two halves' outputs with the
        root operator read from the input. The candidate is then verified like any sample
        (answer, then fresh operands), so a wrong half produces a rejected candidate, not
        a training target. Nothing gold: the structure is the input's, the halves are the
        model's own, the check is the answer. What the core then has to learn is to emit
        the whole program *directly*, which is what held-out evaluation measures.
        """
        from .alu import parse_expr
        from .selfcompile import tree_to_expr

        need = [i for i, h in enumerate(hits) if h is None]
        if not need:
            return hits
        trees = [parse_expr(tasks[i][0]) for i in need]
        halves = []
        for t in trees:
            halves += [(tree_to_expr(t[1]), "0", []), (tree_to_expr(t[2]), "0", [])]
        self.rmt._prepare(halves)
        lh = self.rmt._latents(halves)
        cnt = torch.tensor(self.rmt._counts, device=self.device)
        op_l, a_l, b_l = self.machine.logits(lh, cnt)
        oi, ai, bi = op_l.argmax(-1), a_l.argmax(-1), b_l.argmax(-1)
        c_instr = 2 ** (depth - 1) - 1
        out = list(hits)
        for j, (i, t) in enumerate(zip(need, trees)):
            halves_prog = [[(int(oi[r, st]), int(ai[r, st]), int(bi[r, st]))
                            for st in range(c_instr)] for r in (2 * j, 2 * j + 1)]
            out[i] = self.stitch(halves_prog[0], halves_prog[1], t[0], depth)
        # Compose proposals are candidates, so check the answer here; _generalises then
        # checks fresh operands.
        _, _, vals, answers, keep = self.rmt._prepare(tasks)
        cand = [i for i in need if out[i] is not None]
        if cand:
            correct, _ = self._reward([vals[i] for i in cand], [out[i] for i in cand],
                                      [answers[i] for i in cand],
                                      [tasks[i][2] for i in cand], self._out_reg(depth))
            for c, i in zip(correct.tolist(), cand):
                if c < 1.0 or not keep[i]:
                    out[i] = None
                else:
                    self._composed += 1
        return out

    @torch.no_grad()
    def _generalises(self, tasks, hits, out_reg: int, n_variants: int = 2):
        """Drop verified programs that were only right by coincidence.

        A wrong program can still hit the answer on one instance (``a-b+c`` equals
        ``a+b-c`` whenever ``b == c``), and those poison expert iteration: measured, a
        depth-2 buffer of 1538 "verified" programs left held-out depth-2 accuracy at 0.20.
        A real program is operand-invariant -- the property that makes self-compilation
        exact -- so it must also be right on the same structure with fresh operands. Each
        found program is run on ``n_variants`` re-drawn instances of its own problem, whose
        answers come from the environment exactly as the original's did. Still outcome-only:
        no gold program, no gold trace.
        """
        from .alu import parse_expr
        from .selfcompile import tree_to_expr
        from ._native import evaluate

        rng = self._replay_rng

        def redraw(t):
            if isinstance(t[1], int):
                return (t[0], self.grammar._num(self.digits, rng),
                        self.grammar._num(self.digits, rng))
            return (t[0], redraw(t[1]), redraw(t[2]))

        var_tasks, var_progs, owner = [], [], []
        for i, h in enumerate(hits):
            if h is None:
                continue
            tree = parse_expr(tasks[i][0])
            for _ in range(n_variants):
                e = tree_to_expr(redraw(tree))
                a = evaluate(e)
                if a is not None:
                    var_tasks.append((e, str(a), []))
                    var_progs.append(h)
                    owner.append(i)
        if not var_tasks:
            return hits
        _, _, vals, answers, keep = self.rmt._prepare(var_tasks)
        correct, _ = self._reward(vals, var_progs, answers, [t[2] for t in var_tasks],
                                  out_reg)
        bad = {owner[k] for k in range(len(owner))
               if float(correct[k]) < 1.0 or not keep[k]}
        self._coincidences += len(bad)
        return [None if i in bad else h for i, h in enumerate(hits)]

    def _rft_loss(self, tasks, hits, depth: int, logits, n_used: int) -> torch.Tensor:
        """Expert iteration: train on programs the model found that verifiably work.

        REINFORCE on these programs failed to consolidate (3e-i); the dispatch work found
        the same pattern, where policy gradient on accept/refuse failed and the identical
        observations as cross-entropy targets succeeded. So a sampled program whose
        executed answer equals the problem's answer becomes a supervised target -- the
        supervised path the register machine already solves at 1.000 -- with nothing gold
        in it: no gold program, no gold intermediate trace, only the final answer. Found
        programs also enter a per-depth replay buffer (STaR/ReST-style), and each step
        trains on a replayed batch too, so a rare find is not used once and forgotten.
        """
        rows = [i for i, h in enumerate(hits) if h is not None]
        buf = self.buffer[depth]
        for i in rows:
            buf.append((tasks[i], hits[i]))
        del buf[:max(0, len(buf) - self.replay)]
        parts = []
        if rows:
            sel = [lg[rows] for lg in logits]
            parts.append(self._program_ce(sel, [hits[i] for i in rows], n_used))
        if buf:
            k = min(len(buf), self.cfg.batch_size)
            picks = [buf[j] for j in self._replay_rng.sample(range(len(buf)), k)]
            r_tasks = [t for t, _ in picks]
            self.rmt._prepare(r_tasks)
            lh = self.rmt._latents(r_tasks)
            cnt = torch.tensor(self.rmt._counts, device=self.device)
            r_logits = self.machine.logits(lh, cnt)
            parts.append(self._program_ce(r_logits, [p for _, p in picks], n_used))
        if not parts:
            # A zero that stays in the graph. Not ``sum(logits) * 0``: the pointer logits
            # carry -inf at masked registers, and -inf * 0 is nan (the trap this repo has
            # hit before). The op logits are never masked.
            return logits[0].sum() * 0.0
        return sum(parts) / len(parts)

    def train_step(self, step: int) -> Dict[str, float]:
        self.rmt.inner.reasoner.train()
        self.machine.train()
        arm = self.bandit.sample() if self.bandit else self.depths.index(self._fixed)
        depth = self.depths[arm]
        tasks = self._sample_batch(depth, self.cfg.batch_size)
        for g in self.opt.param_groups:
            g["lr"] = self.rmt.inner._lr_at(step)

        _, _, vals, answers, keep = self.rmt._prepare(tasks)
        traces = [t[2] for t in tasks]
        latent_h = self.rmt._latents(tasks)
        cnt = torch.tensor(self.rmt._counts, device=self.device)
        op_l, a_l, b_l = self.machine.logits(latent_h, cnt)        # (B,I,*)
        keep_t = torch.tensor(keep, dtype=torch.float32, device=self.device)

        # Sample from an epsilon-*mixture* of the policy and the uniform over legal
        # actions, and score log-probability under that same mixture. Tempering failed
        # in the smoke run: once the logits grow large, dividing them by a fixed
        # temperature still leaves a near-deterministic distribution, so exploration
        # died and every group at the hard arm was identical (ACR 1.00). An epsilon
        # floor cannot be overwhelmed by logit growth -- every legal action keeps
        # probability at least ``eps / n_legal`` -- so the group always has variance for
        # GRPO to work on. It is the sampling that was the blocker, not the gradient:
        # REINFORCE raises a rare good action with gradient ~ adv*(1 - p), which does
        # not vanish; the action simply has to be sampled first.
        #
        # Epsilon cools on each arm's *own* training-step count, not the global step and
        # not the arm's success. A global anneal cannot serve a curriculum: by the time
        # the deepest arm becomes the frontier the schedule has cooled, so the hard arm
        # never explores. Coupling epsilon to success looked right and created a fixed
        # point instead -- low success holds epsilon high, high epsilon caps success, and
        # depth 1 stuck at 0.59/0.16. A per-arm step schedule cools each arm over its own
        # budget regardless of where it plateaus, so a late-unlocked deep arm still gets
        # full exploration from its own step zero.
        self._arm_steps[depth] += 1
        prog = (self._arm_steps[depth] / self.explore_anneal) if self.explore_anneal else 1.0
        eps = self.explore_eps_min + (self.explore_eps - self.explore_eps_min) * max(
            0.0, 1.0 - prog)
        mixes = []
        for x in (op_l, a_l, b_l):
            legal = torch.isfinite(x)
            p = torch.softmax(x, dim=-1)                          # masked -> 0
            unif = legal.to(p.dtype) / legal.sum(-1, keepdim=True).clamp_min(1.0)
            mixes.append((1.0 - eps) * p + eps * unif)            # (B,I,*)
        G, B = self.group, len(tasks)
        out_reg = self._out_reg(depth)
        # Only the instructions a depth-d program uses carry gradient. The machine emits
        # the full budget so the executor sees a complete program, but the spare
        # instructions past ``n_used`` are never read (the answer is at ``out_reg``), so
        # scoring their log-probability would spend gradient teaching a routing that does
        # not exist.
        n_used = 2 ** depth - 1
        logps, rewards = [], []
        hits: List[Optional[list]] = [None] * B
        best_lp = [float("-inf")] * B
        for _ in range(G):
            idx = [torch.multinomial(mp.reshape(-1, mp.size(-1)), 1).view(mp.shape[:-1])
                   for mp in mixes]                               # each (B,I)
            oi, ai, bi = idx
            lp = torch.zeros(B, device=self.device)
            for mp, ix in zip(mixes, idx):
                g = torch.log(mp.gather(-1, ix.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12))
                lp = lp + g[:, :n_used].sum(-1)                   # (B,)
            progs = [[(int(oi[i, t]), int(ai[i, t]), int(bi[i, t]))
                      for t in range(self.n_instr)] for i in range(B)]
            correct, process = self._reward(vals, progs, answers, traces, out_reg)
            for i in range(B):
                # Keep the *most probable* verified sample, not the first. A problem has
                # several correct programs, and training on a mix of them lets the argmax
                # splice instructions from different ones into a program that is none of
                # them; preferring the model's own mode keeps one program per structure.
                if keep[i] and float(correct[i]) >= 1.0 and float(lp[i].detach()) > best_lp[i]:
                    hits[i], best_lp[i] = progs[i], float(lp[i].detach())
            logps.append(lp)
            rewards.append(correct + self.process_coef * process)
        logp = torch.stack(logps)                                 # (G,B)
        R = torch.stack(rewards)                                  # (G,B)

        # Group-relative advantage, with DAPO dynamic sampling: a group with no reward
        # variance carries no gradient, so it is dropped from the loss and counted.
        #
        # **No division by the within-group std.** GRPO's ``(r-mean)/std`` was measured
        # here to make the argmax program oscillate (solve_hard flipping 0.38<->0.67 and
        # never consolidating): with a group of 6 and near-binary reward the std estimate
        # is tiny and noisy, and dividing by it amplifies exactly the batches that should
        # count least, so every step over-corrects. Dr. GRPO (arXiv:2503.20783) removes
        # this term for the same reason; the mean-subtracted advantage is the unbiased
        # baseline and is enough. ``live`` still gates on reward variance so an all-equal
        # group contributes nothing.
        mean = R.mean(0, keepdim=True)
        std = R.std(0, keepdim=True)
        adv = R - mean
        live = ((std.squeeze(0) > 1e-6) & (keep_t > 0)).float()   # (B,)
        denom = (G * live.sum()).clamp_min(1.0)
        pg = -((adv.detach() * logp) * live.unsqueeze(0)).sum() / denom
        ent = self._entropy((op_l, a_l, b_l))
        # A one-sided *floor* on entropy, not a bonus. The smoke run collapsed to a
        # deterministic policy by step 100: at depth 2 every sampled group was then
        # identical, so ``std`` was zero, every group was dropped, and no gradient ever
        # reached the hard arm (ACR 1.00). A weak bonus does not stop this because once
        # the easy arm is solved its groups are degenerate too, so the only live
        # gradient is the bonus and it is small. Penalising entropy *below a target*
        # keeps the policy sampling variety while the curriculum climbs, and the floor
        # anneals to zero so the policy is free to commit once the hard arm is reachable
        # -- a collapsed policy cannot explore, and an eternally-hot one cannot solve.
        floor = self.entropy_floor
        if self.entropy_anneal:
            floor = self.entropy_floor * max(0.0, 1.0 - step / self.entropy_anneal)
        loss = pg + self.entropy_coef * torch.relu(floor - ent)
        if self.mode == "rft":
            if self.compose and depth >= 2:
                hits = self._compose(tasks, hits, depth)
            if self.rft_verify:
                hits = self._generalises(tasks, hits, out_reg)
            loss = self._rft_loss(tasks, hits, depth, (op_l, a_l, b_l), n_used)

        if self.calib_coef:
            # The Jev/RLCD borrow: train an emitted confidence toward the empirical
            # outcome. A proxy for now -- the program's own geometric-mean probability
            # stands in for a confidence head -- so it is off (coef 0) for the headline.
            n_choice = 3 * self.n_instr
            conf = torch.exp(logp / n_choice)                     # (G,B) in (0,1]
            hit = (R >= 1.0).float()
            brier = (((conf - hit) ** 2) * live.unsqueeze(0)).sum() / denom
            loss = loss + self.calib_coef * brier

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {float(loss)}")
        for nm, prm in (("reasoner", self.rmt.inner.reasoner), ("machine", self.machine)):
            for q in prm.parameters():
                if q.grad is not None and not torch.isfinite(q.grad).all():
                    raise FloatingPointError(f"non-finite gradient in {nm} at {step}")
        torch.nn.utils.clip_grad_norm_(
            list(self.rmt.inner.reasoner.parameters())
            + list(self.machine.parameters()), self.cfg.grad_clip)
        self.opt.step()

        # Learnability for the bandit: mean over problems of p*(1-p), p = per-problem
        # success rate over the group. Highest at the frontier, near zero once an arm
        # is solved or hopeless -- which is exactly when it should stop being drawn.
        p = (R >= 1.0).float().mean(0)                            # (B,)
        learn = float((p * (1.0 - p) * keep_t).sum() / keep_t.sum().clamp_min(1.0))
        # Mastery and the epsilon schedule read the *argmax* program, not the sampled
        # one. A deep policy that is right at the mode still samples imperfectly under an
        # epsilon floor -- 21 actions at depth 3 means even a perfect policy solves only
        # ~0.76 of samples -- so a sampled-success mastery test would never let a deep
        # arm cool or graduate. One extra argmax rollout per step decouples the two.
        with torch.no_grad():
            aidx = [op_l.argmax(-1), a_l.argmax(-1), b_l.argmax(-1)]
            aprogs = [[(int(aidx[0][i, t]), int(aidx[1][i, t]), int(aidx[2][i, t]))
                       for t in range(self.n_instr)] for i in range(B)]
            acorrect, _ = self._reward(vals, aprogs, answers, traces, out_reg)
        success = float((acorrect * keep_t).sum() / keep_t.sum().clamp_min(1.0))
        self._succ[depth] = 0.9 * self._succ[depth] + 0.1 * success
        if self.bandit:
            self.bandit.update(arm, learn, success)
        acr = float((1.0 - live).mean())                         # advantage-collapse rate

        return {"loss": float(loss.detach()), "pg": float(pg.detach()),
                "entropy": float(ent.detach()),
                "reward": float((R * live.unsqueeze(0)).sum() / denom),
                "solve": float(((R >= 1.0).float() * keep_t.unsqueeze(0)).mean()),
                "solve_hard": success, "eps": eps,
                "learnability": learn, "acr": acr, "depth": float(depth),
                "dropped": 1.0 - sum(keep) / max(1, len(keep))}

    # -- eval --------------------------------------------------------------
    @torch.no_grad()
    def _heldout(self, depth: int, n: int) -> List[Tuple[str, str, List[int]]]:
        rng = random.Random(self.cfg.seed * 7 + 999 + depth)
        desc = self._descriptor(depth)
        out, seen = [], set()
        for _ in range(400 * max(1, n)):
            if len(out) >= n:
                break
            t = self.grammar.sample_with_trace(desc, rng.randint(0, 2 ** 31 - 1))
            if is_heldout(t[0]) and t[0] not in seen:
                seen.add(t[0])
                out.append(t)
        while len(out) < n and out:
            out.append(out[len(out) % len(out)])
        return out

    @torch.no_grad()
    def evaluate(self, n: int = 256) -> Dict[str, float]:
        """Held-out ``answer_acc_hard`` per arm: the argmax program, executed exactly.

        The soft program is never scored here. 3a-x is explicit that a blurred read
        decodes to nowhere ~30% of the time, so the only honest correctness measure is
        the decided program run as a real program -- the same choice ``evaluate`` makes
        in :mod:`lamb.regmachine`.
        """
        self.rmt.inner.reasoner.eval()
        self.machine.eval()
        out: Dict[str, float] = {}
        accs = []
        for depth in self.depths:
            tasks = self._heldout(depth, n)
            _, _, vals, answers, keep = self.rmt._prepare(tasks)
            latent_h = self.rmt._latents(tasks)
            cnt = torch.tensor(self.rmt._counts, device=self.device)
            op_l, a_l, b_l = self.machine.logits(latent_h, cnt)
            oi, ai, bi = op_l.argmax(-1), a_l.argmax(-1), b_l.argmax(-1)
            progs = [[(int(oi[i, t]), int(ai[i, t]), int(bi[i, t]))
                      for t in range(self.n_instr)] for i in range(len(tasks))]
            regs = run_program(self.machine, vals, progs, str(self.device))
            got = self._decode_answers(regs, self._out_reg(depth))
            n_keep = max(1, sum(keep))
            acc = sum(1 for g, a, k in zip(got, answers, keep)
                      if k and g is not None and g == Fraction(int(a))) / n_keep
            out[f"acc_hard_d{depth}"] = acc
            accs.append(acc)
        out["acc_hard"] = sum(accs) / len(accs)
        out["acc_hard_max_depth"] = out[f"acc_hard_d{self.d_max}"]
        return out
