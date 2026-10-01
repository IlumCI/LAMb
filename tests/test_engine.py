"""The packaged engine and its dependency-free executor.

``lamb.progexec`` is what a verifier runs instead of the model, so it is pinned to the parts
of the model-side stack it restates: the parser, the register layout, and the exact
execution. If any of these drift apart, a verifier would accept or reject the wrong work.
"""

from __future__ import annotations

import json
import random
from fractions import Fraction

import pytest
import torch

from lamb import progexec
from lamb.alu import parse_expr
from lamb.selfplay.grammar import Descriptor, TaskGrammar


def _exprs(n=60, seed=0, depths=(1, 2, 3), ops_key=1, shape=1):
    g, rng = TaskGrammar(), random.Random(seed)
    out = []
    for i in range(n):
        d = depths[i % len(depths)]
        out.append(g.sample(Descriptor(d, 2, ops_key, shape), rng.randint(0, 2 ** 31 - 1)))
    return out


def test_parser_agrees_with_the_model_side_parser():
    for e, a in _exprs():
        assert progexec.parse(e) == parse_expr(e)
        assert progexec.render(progexec.parse(e)) == e
        assert progexec.evaluate(progexec.parse(e)) == Fraction(int(a))


def test_structure_key_matches_selfcompile():
    from lamb.selfcompile import op_pattern

    for e, _ in _exprs():
        t = progexec.parse(e)
        assert progexec.structure(t) == op_pattern(t)


def test_pure_python_execution_matches_the_residue_machine():
    # The verifier executes with Fractions; the model trains through residues. Run the
    # *same* gold programs through both on the same layout and demand identical values.
    from lamb import ArithmeticTokenizer, LotusConfig
    from lamb.config import ModelConfig
    from lamb.regmachine import RegMachineTrainer, run_program

    cfg = LotusConfig(steps=1, batch_size=8, n_latent=8, loops=1, depth=3, digits=2, ops_key=1,
                      shape=1, trace_coef=0.0, alu_coef=0.0, use_boundaries=False,
                      switch_coef=0.0, device="cpu",
                      alu_moduli=(64, 125, 27, 11, 7, 13, 37, 101, 41))
    rmt = RegMachineTrainer(cfg, ArithmeticTokenizer(),
                            ModelConfig(d_model=16, n_heads=2, d_ff=32, recurrent_steps=1))
    tasks = rmt.inner._sample_batch(32)
    _, golds, vals, answers, keep = rmt._prepare(tasks)
    regs = run_program(rmt.machine, vals, golds, "cpu")
    residue = regs.decode(rmt.n_operands + rmt.n_instr - 1)
    spec = {"ops": list(rmt.machine.ops), "n_operands": rmt.n_operands,
            "n_const": rmt.n_const, "constants": [1] * rmt.n_const}
    for task, g, r, k in zip(tasks, golds, residue, keep):
        if not k:
            continue
        t = progexec.parse(task[0])
        assert progexec.run(g, progexec.leaves(t), spec) == Fraction(r)


def test_verify_accepts_honest_work_and_rejects_everything_else():
    spec = {"ops": ["+", "-", "*"], "n_operands": 5, "n_const": 1, "constants": [1]}
    expr = "(12+7)-(30-4)"
    prog = [[0, 1, 2], [1, 3, 4], [1, 5, 6]]          # r5=12+7, r6=30-4, r7=r5-r6
    assert progexec.verify(spec, expr, prog, -7)
    assert not progexec.verify(spec, expr, prog, -8)                    # wrong claim
    assert not progexec.verify(spec, expr, [[0, 1, 2], [0, 3, 4], [1, 5, 6]], -7)  # wrong op
    assert not progexec.verify(spec, expr, [[0, 1, 9]], 0)              # unwritten register
    assert not progexec.verify(spec, expr, [[0, 1, 2]], 19)  # right sub-result, wrong problem


def test_refusal_on_an_unknown_structure():
    lib = {"format": 1, "spec": {"ops": ["+"], "n_operands": 3, "n_const": 1,
                                 "constants": [1]},
           "programs": {progexec.key(progexec.parse("1+2")): {"depth": 1,
                                                               "program": [[0, 1, 2]]}}}
    assert progexec.solve(lib, "40+2") == 42                  # operand-invariant
    assert progexec.solve(lib, "40-2") is None               # never compiled: refused


def test_engine_round_trip_train_save_load_compile(tmp_path):
    from lamb import engine

    tr = engine.build_trainer([1, 2], steps=30, batch_size=12)
    engine.train(tr, 30, log=lambda *_: None)
    path = tmp_path / "e.pt"
    engine.save(tr, str(path))
    tr2 = engine.load(str(path))
    exprs = [e for e, _ in _exprs(10, 1, depths=(1, 2), ops_key=0, shape=0)]
    assert engine.core_programs(tr, exprs) == engine.core_programs(tr2, exprs)
    lib = engine.compile_library(tr2, n_per_depth=100, log=lambda *_: None)
    json.dumps(lib)                                          # portable
    for k, entry in lib["programs"].items():                 # every entry holds when checked
        t = progexec.parse(progexec.render(_example(k)))
        assert progexec.run(entry["program"], progexec.leaves(t), lib["spec"]) == \
            progexec.evaluate(t)


def _example(key):
    """An instance of a structure key, operands drawn fresh."""
    rng = random.Random(7)

    def build(s):
        if len(s) == 1:
            return (s[0], rng.randint(10, 99), rng.randint(10, 99))
        return (s[0], build(s[1]), build(s[2]))

    return build(json.loads(key))
