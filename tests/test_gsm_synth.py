"""Generated GSM8K-style problems are only useful if they are *valid* GSM8K-format data.

They are emitted in GSM8K's own annotation format and pushed through the real parser and
aligner, so these pin what the bridge relies on: every family aligns, every aligned program
executes to the stated answer, answers are non-negative integers, and generation is
deterministic per seed (a training stream must be reproducible).
"""

from __future__ import annotations

import random

from lamb.bridge_train import BridgeConfig, build_examples, verify_alignment
from lamb.gsm_synth import FAMILIES, generate, generate_rows


def test_every_family_aligns_and_executes():
    from lamb.algebra import ResidueSystem
    from lamb.regmachine import RegisterMachine

    cfg = BridgeConfig()
    rng = random.Random(0)
    machine = RegisterMachine(8, cfg.n_operands, cfg.n_instr,
                              ResidueSystem(tuple(cfg.rational_moduli)), rational=True)
    for fam in FAMILIES:
        rows = [generate(rng, [fam]) for _ in range(60)]
        ex, stats = build_examples(rows, cfg.n_operands, cfg.n_instr, cfg.constants,
                                   cfg.lexical)
        assert stats["program_coverage"] == 1.0, fam.__name__
        assert verify_alignment(ex, machine) == 1.0, fam.__name__


def test_answers_are_nonnegative_integers_and_problems_vary():
    rows = generate_rows(500, seed=3)
    answers = [r["answer"].split("####")[1].strip() for r in rows]
    assert all(a.lstrip("-").isdigit() and int(a) >= 0 for a in answers)
    assert len({r["question"] for r in rows}) > 480        # essentially never repeats


def test_generation_is_deterministic_per_seed():
    assert generate_rows(50, seed=11) == generate_rows(50, seed=11)
    assert generate_rows(50, seed=11) != generate_rows(50, seed=12)


def test_bridging_recovers_a_percent_step_the_annotation_skipped():
    """``40%`` used as ``0.4`` with no division written: alignment used to fail on it.

    That shape was 5.5% of GSM8K test, and one-step gaps were 13.6%, against ~2% of problems that
    lack a number the extractor could find. Bridging inserts ``40 / 100`` from the constant
    register, and the program must still execute to the dataset's own answer. With bridging
    off, the same row must stay unaligned, so earlier measurements reproduce.
    """
    from lamb.algebra import ResidueSystem
    from lamb.regmachine import RegisterMachine

    cfg = BridgeConfig()
    row = {"question": "A shirt costs $50. It is on sale for 40% off. How much is the discount?",
           "answer": "<<50*0.4=20>>\n#### 20"}
    off, _ = build_examples([row], cfg.n_operands, cfg.n_instr, cfg.constants, cfg.lexical)
    on, _ = build_examples([row], cfg.n_operands, cfg.n_instr, cfg.constants, cfg.lexical,
                           bridge=True)
    assert off[0].program is None and on[0].program is not None
    machine = RegisterMachine(8, cfg.n_operands, cfg.n_instr,
                              ResidueSystem(tuple(cfg.rational_moduli)), rational=True)
    assert verify_alignment(on, machine) == 1.0
