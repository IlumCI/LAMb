"""The model decides when to compile a new copy of itself (ROADMAP 3g, Phase C).

Pinned here: deployment with fallback is always correct, the trigger reads only logged
experience, and self-verification refuses a copy the core cannot actually back. The stream
results (self-directed compilation against never compiling and compiling up front) are
multi-minute runs recorded in the ROADMAP.
"""

from __future__ import annotations

import random
from fractions import Fraction

import pytest
import torch

from lamb import ArithmeticTokenizer, LotusConfig
from lamb.config import ModelConfig
from lamb.dispatch import build_gold_copies, emit_program, internal_nodes, parse_nodes
from lamb.regmachine import RegMachineTrainer
from lamb.selfcompile import execute_dispatch_program, tree_depth
from lamb.selfimprove import CoreCompiler, effective, ideal_calls, run_with_fallback
from lamb.selfplay.grammar import Descriptor, TaskGrammar


def _core(steps=1):
    torch.manual_seed(0)
    cfg = LotusConfig(steps=steps, batch_size=16, n_latent=8, loops=1, depth=3, digits=2,
                      ops_key=0, shape=1, trace_coef=0.0, alu_coef=0.0,
                      use_boundaries=False, switch_coef=0.0, device="cpu",
                      alu_moduli=(16, 25, 27, 11, 37))
    return RegMachineTrainer(cfg, ArithmeticTokenizer(),
                             ModelConfig(d_model=16, n_heads=2, d_ff=32, recurrent_steps=1))


def _tasks(n, seed):
    g, rng = TaskGrammar(), random.Random(seed)
    out = []
    while len(out) < n:
        e, a = g.sample(Descriptor(5, 2, 0, 1), rng.randint(0, 2 ** 31 - 1))
        if not parse_nodes(e).is_leaf():
            out.append((e, a))
    return out


@pytest.fixture(scope="module")
def setup():
    copies = build_gold_copies([(1, 0), (2, 0)], n=800)
    core = _core()
    comp = CoreCompiler(core, copies, k=3, forward_cost=0.5)   # cheap trigger: fires fast
    return copies, core, comp


def test_fallback_deployment_is_always_correct(setup):
    # Voting to dispatch everything, with no depth-3 coverage, refuses constantly -- and
    # still gets every answer right, because a refused hand-off is split instead.
    copies, _, _ = setup
    rng = random.Random(0)
    for e, a in _tasks(40, 1):
        root = parse_nodes(e)
        votes = {id(n): rng.random() < 0.7 for n in internal_nodes(root)}
        decide = lambda n: votes[id(n)]
        calls, handoffs, _ = run_with_fallback(root, decide, copies)
        got, _ = execute_dispatch_program(emit_program(root, effective(decide, copies)), copies)
        assert got == Fraction(int(a))
        assert calls >= ideal_calls(root, 2)
        assert all(ok is False for n, ok in handoffs if tree_depth(n.tree()) >= 3)


def test_the_compiler_only_logs_work_its_core_could_have_done(setup):
    copies, _, comp = setup
    for e, _ in _tasks(30, 2):
        root = parse_nodes(e)
        _, _, splits = run_with_fallback(root, lambda n: False, copies)
        comp.observe(splits)
    assert comp.count, "depth-3 subtrees split during work must be logged"
    for key, buf in comp.buffer.items():
        assert len(buf) <= comp.k
        assert all(v is not None for _, v in buf)      # values came from the trusted slow path


def test_self_verification_refuses_a_copy_the_core_cannot_back(setup):
    # An untrained core emits programs that do not reproduce the values the model already
    # computed through its trusted copies, so nothing may be adopted however often the
    # trigger fires. This is the check that stops the model growing a wrong copy of itself.
    copies, _, comp = setup
    for e, _ in _tasks(120, 3):
        _, _, splits = run_with_fallback(parse_nodes(e), lambda n: False, copies)
        comp.observe(splits)
    events = comp.maybe_compile()
    assert events, "the trigger should have fired on recurring keys"
    assert comp.adopted == 0 and len(comp.copy.library) == 0
    assert comp.forwards > 0                            # the checks were paid for
