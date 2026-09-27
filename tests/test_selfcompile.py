"""Self-compilation: the model compiles into an exact, executor-only copy of itself.

The claim these pin (ROADMAP 3g): because the register machine's pointers index slots,
not values, the program the core emits is operand-invariant, so reading it off the core
once gives a program library that reproduces the core's own output exactly with the neural
net removed. The strong test is *fidelity* -- compiled output equals the core's own output
-- not just accuracy, because a faithful copy tracks the core whether the core is right or
wrong. Measured for real depth-2 and depth-3 teachers: 8 and 128 patterns, one program
each, fidelity 1.000.
"""

from __future__ import annotations

import torch

from lamb import ArithmeticTokenizer, LotusConfig
from lamb.config import ModelConfig
from lamb.regmachine import RegMachineTrainer
from lamb.selfcompile import CompiledModel, compile_model, op_pattern, verify
from lamb.alu import parse_expr


def _teacher(depth=2, steps=400, d_model=48):
    torch.manual_seed(0)
    cfg = LotusConfig(steps=steps, batch_size=64, n_latent=16, loops=3, depth=depth,
                      digits=2, eval_tasks=64, trace_coef=0.0, alu_coef=0.0,
                      use_boundaries=False, switch_coef=0.0, device="cpu", lr=2e-3,
                      alu_moduli=(16, 25, 27, 11, 37))
    mc = ModelConfig(d_model=d_model, n_heads=4, d_ff=2 * d_model, recurrent_steps=3)
    t = RegMachineTrainer(cfg, ArithmeticTokenizer(), mc)
    for s in range(steps):
        t.train_step(s)
    return t


def test_op_pattern_is_the_post_order_operator_signature():
    # "(6+6)-(4-8)" parses to ('-', ('+',6,6), ('-',4,8)); post-order ops are +, -, -.
    assert op_pattern(parse_expr("(6+6)-(4-8)")) == ("+", "-", "-")
    assert op_pattern(parse_expr("12+7")) == ("+",)


def test_solve_runs_the_library_program_not_the_core():
    # Harness test, no reliance on training: seed the library with each task's own gold
    # program (known correct), then the executor-only solve must return the true answers.
    # A pattern is operand-invariant, so one gold per pattern solves every task with it.
    t = _teacher(depth=2, steps=1)                       # untrained: this tests mechanics
    tasks = t.inner._sample_batch(16)
    _, golds, _, answers, _ = t._prepare(tasks)
    library = {}
    for task, gold in zip(tasks, golds):
        library.setdefault(op_pattern(parse_expr(task[0])), gold)
    comp = CompiledModel(library=library, n_operands=t.n_operands, n_instr=t.n_instr,
                         out_reg=t.n_operands + t.n_instr - 1, machine=t.machine,
                         prepare=t._prepare, rational=False)
    got, covered = comp.solve(tasks)
    assert all(covered)
    from fractions import Fraction
    assert all(g == Fraction(int(a)) for g, a in zip(got, answers))


def test_compiled_copy_is_an_exact_1to1_reproduction_of_the_core():
    # The headline: a trained core compiles to a library that reproduces it exactly.
    t = _teacher(depth=2, steps=400)
    r = t.evaluate(128)
    assert r["answer_acc_hard"] >= 0.95, "teacher must actually work for the claim to mean anything"
    comp = compile_model(t, n_samples=1024)
    v = verify(t, comp, n=256)
    # Operand-invariance: every operator pattern needs exactly one program.
    assert v["max_variants"] == 1.0
    assert v["coverage"] == 1.0
    # 1:1 fidelity: the executor-only copy matches the core's own output on every problem.
    assert v["fidelity"] == 1.0
    # And a faithful copy of a correct core is itself correct.
    assert v["compiled_acc"] == v["core_acc"]


def test_uncovered_structure_is_reported_not_silently_bridged():
    # A copy that quietly falls back to the core where its library is empty is not a
    # measurement of the copy. An unknown pattern must come back covered=False.
    t = _teacher(depth=2, steps=1)
    comp = CompiledModel(library={}, n_operands=t.n_operands, n_instr=t.n_instr,
                         out_reg=t.n_operands + t.n_instr - 1, machine=t.machine,
                         prepare=t._prepare, rational=False)
    _, covered = comp.solve(t.inner._sample_batch(4))
    assert not any(covered)
