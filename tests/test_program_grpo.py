"""GRPO-over-programs and the curriculum bandit (ROADMAP 3e).

These pin the mechanics, not the headline claim -- whether outcome-only GRPO reaches
the supervised arm at depth 3 is a multi-seed study question (H1), not a unit test.
What is testable cheaply is that the pieces do what they say: the bandit prefers the
learnable arm, a step produces finite gradients, degenerate groups are dropped rather
than fed zero-variance advantage, and eval scores a decided program.
"""

from __future__ import annotations

import math

import torch

from lamb import ArithmeticTokenizer, LotusConfig
from lamb.config import ModelConfig
from lamb.program_grpo import CurriculumBandit, GRPOTrainer


def _trainer(**kw):
    torch.manual_seed(0)
    depths = kw.pop("depths", [1, 2])
    mode = kw.pop("mode", "grpo")
    base = dict(steps=8, batch_size=16, n_latent=8, loops=3, depth=max(depths),
                eval_tasks=16, trace_coef=0.0, alu_coef=0.0, use_boundaries=False,
                switch_coef=0.0, device="cpu", alu_moduli=(16, 25, 27, 11, 37))
    base.update(kw)
    return GRPOTrainer(LotusConfig(**base), ArithmeticTokenizer(), depths=depths,
                       model_cfg=ModelConfig(d_model=48, n_heads=4, d_ff=96,
                                             recurrent_steps=3),
                       group=6, mode=mode)


def test_bandit_prefers_the_learnable_arm():
    # Three arms: one solved (learnability 0), one at the frontier (0.25), one hopeless
    # (0). The frontier is the only one worth sampling, and after the mandatory one-shot
    # cover of each arm the bandit should concentrate there.
    b = CurriculumBandit([1, 2, 3], ema=0.5, temp=0.05, eps=0.0, seed=0)
    signal = {0: 1.0, 1: 0.25, 2: 0.0}     # arm index -> learnability
    for _ in range(200):
        i = b.sample()
        b.update(i, signal[i])
    # Arm 0 is "solved" here only in the toy signal; the real solved-arm signal is 0.
    # Redo with the honest shape: solved and hopeless both read ~0, frontier reads high.
    b = CurriculumBandit([1, 2, 3], ema=0.5, temp=0.05, eps=0.0, seed=0)
    signal = {0: 0.0, 1: 0.25, 2: 0.0}
    counts = [0, 0, 0]
    for _ in range(300):
        i = b.sample()
        counts[i] += 1
        b.update(i, signal[i])
    assert counts[1] > counts[0] and counts[1] > counts[2]


def test_bandit_covers_every_arm_before_exploiting():
    # ``seen`` advances through ``update``, so a real loop samples then reports. The
    # first four draws must be the four distinct arms before any EMA is trusted.
    b = CurriculumBandit([1, 2, 3, 4], eps=0.0, seed=1)
    drawn = []
    for _ in range(4):
        i = b.sample()
        drawn.append(i)
        b.update(i, 0.0)
    assert set(drawn) == {0, 1, 2, 3}


def test_train_step_is_finite_and_updates():
    tr = _trainer()
    before = [p.detach().clone() for p in tr.machine.parameters()]
    m = tr.train_step(0)
    assert math.isfinite(m["loss"]) and math.isfinite(m["entropy"])
    assert 0.0 <= m["acr"] <= 1.0 and 0.0 <= m["solve"] <= 1.0
    after = list(tr.machine.parameters())
    assert any((a - b).abs().sum() > 0 for a, b in zip(after, before))


def test_degenerate_groups_do_not_break_the_step():
    # A fresh model at depth 3 produces all-wrong groups (ACR near 1). The step must
    # still be finite -- a zero-variance group is dropped, not divided by zero. This is
    # the exact regime (advantage collapse) that the entropy floor and process reward
    # exist to survive, and a NaN here would be the failure that hid a retracted claim.
    tr = _trainer(depths=[3], n_latent=8, batch_size=12)
    for step in range(3):
        m = tr.train_step(step)
        assert math.isfinite(m["loss"])


def test_evaluate_scores_a_decided_program_per_arm():
    tr = _trainer(depths=[1, 2])
    for step in range(4):
        tr.train_step(step)
    ev = tr.evaluate(n=24)
    assert "acc_hard_d1" in ev and "acc_hard_d2" in ev
    assert "acc_hard_max_depth" in ev
    for k, v in ev.items():
        assert 0.0 <= v <= 1.0


def test_curriculum_depth_must_match_machine():
    # The machine is sized for the deepest arm; a cfg.depth that disagrees is a silent
    # mis-size waiting to happen, so it is rejected at construction.
    import pytest

    with pytest.raises(ValueError):
        _trainer(depths=[1, 2], depth=3)


def test_stitching_two_correct_halves_gives_a_correct_whole_program():
    # Self-composition (3e-ii): a depth-d program built from the depth-(d-1) programs of
    # its two halves. Fed the halves' *gold* programs, the stitched program must execute to
    # the whole problem's answer -- this pins the pointer remapping, which is where a
    # composition goes silently wrong (an off-by-one reads a neighbouring register).
    from fractions import Fraction

    from lamb.alu import parse_expr
    from lamb.regmachine import run_program
    from lamb.selfcompile import tree_to_expr

    tr = _trainer(depths=[1, 2, 3], n_latent=8, batch_size=8)
    tasks = tr._sample_batch(3, 12)
    for task in tasks:
        tree = parse_expr(task[0])
        halves = [(tree_to_expr(tree[1]), "0", []), (tree_to_expr(tree[2]), "0", [])]
        _, golds, _, _, _ = tr.rmt._prepare(halves)
        prog = tr.stitch(golds[0], golds[1], tree[0], depth=3)
        _, _, vals, answers, _ = tr.rmt._prepare([task])
        regs = run_program(tr.machine, vals, [prog], "cpu")
        got = Fraction(regs.decode(tr._out_reg(3))[0])
        assert got == Fraction(int(task[1]))


def test_rft_rejects_a_program_that_is_right_only_by_coincidence():
    # A constant program that happens to hit one instance's answer must fail on fresh
    # operands. _generalises re-draws the operands of each found program's own problem.
    tr = _trainer(depths=[1, 2], mode="rft")
    tasks = tr._sample_batch(1, 6)
    _, golds, _, _, _ = tr.rmt._prepare(tasks)
    good = tr._generalises(tasks, [g for g in golds], tr._out_reg(1))
    assert all(h is not None for h in good)                  # gold programs generalise
    bogus = [[(0, 0, 0)] * tr.n_instr for _ in tasks]        # 1 + 1 on every instance
    kept = tr._generalises(tasks, bogus, tr._out_reg(1))
    assert all(h is None for h in kept)
