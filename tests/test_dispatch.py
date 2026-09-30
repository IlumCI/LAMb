"""The core emits its own dispatch program (ROADMAP 3g, Phase B).

Mechanics are pinned here; the learning results (probe mode reaching the cost-optimal
program on held-out depth 4, 0.94-0.98 on unseen depth 5) are multi-minute runs recorded in
the ROADMAP, not unit tests. What is cheap and must never regress: the span parser agrees
with the tree every other component sees, spans land on the right tokens for a causal
readout, votes turn into an executable program, and the coverage oracle used during
training agrees with actually executing.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest
import torch

from lamb import ArithmeticTokenizer
from lamb.alu import parse_expr
from lamb.config import ModelConfig
from lamb.dispatch import (DispatchTrainer, build_gold_copies, covering_depth,
                           emit_program, gold_decide, internal_nodes, parse_nodes,
                           program_outcome)
from lamb.selfcompile import execute_dispatch_program, tree_depth
from lamb.selfplay.grammar import Descriptor, TaskGrammar


@pytest.fixture(scope="module")
def copies():
    # Depth-3 copy covers only *balanced* depth-3 trees, which is what makes the choice real.
    return build_gold_copies([(1, 0), (2, 0), (3, 0)], n=2000)


def _sample(depth, n, seed):
    g, rng = TaskGrammar(), random.Random(seed)
    out = []
    while len(out) < n:
        e, a = g.sample(Descriptor(depth, 2, 0, 1), rng.randint(0, 2 ** 31 - 1))
        if not parse_nodes(e).is_leaf():
            out.append((e, a))
    return out


def test_span_parser_matches_the_shared_tree():
    for e, _ in _sample(4, 60, 0):
        root = parse_nodes(e)
        assert root.tree() == parse_expr(e)
        for n in internal_nodes(root):
            # The span is exactly the node's own text, so its end is where a causal core
            # has seen the whole subtree.
            assert parse_expr(e[n.start:n.end + 1]) == n.tree()


def test_span_ends_map_to_tokens_one_to_one():
    # Token 0 is BOS, then one token per character; the readout indexes end + 1.
    tok = ArithmeticTokenizer()
    e = "((12+3)-(4+5))+(6-7)"
    ids = tok.encode_prompt(e).ids
    assert len(ids) == len(e) + 2                     # BOS ... EQ
    for n in internal_nodes(parse_nodes(e)):
        ch = e[n.end]
        expect = tok.encode_prompt(ch if not ch.isdigit() else ch).ids[1]
        assert ids[n.end + 1] == expect


def test_gold_votes_give_an_exact_cheaper_program(copies):
    calls = split = 0
    for e, a in _sample(5, 60, 1):
        root = parse_nodes(e)
        got, c = execute_dispatch_program(emit_program(root, gold_decide(copies)), copies)
        assert got == Fraction(int(a))
        calls += c
        split += program_outcome(root, lambda n: False, copies)[1]
    assert calls < split                              # dispatching high saves calls


def test_the_choice_is_two_sided(copies):
    # Some depth-3 subtrees are covered (balanced) and some are not (unbalanced). If every
    # depth-3 subtree fell on one side, the policy would have nothing to learn.
    seen = set()
    for e, _ in _sample(4, 200, 2):
        for n in internal_nodes(parse_nodes(e)):
            if tree_depth(n.tree()) == 3:
                seen.add(covering_depth(n, copies) is not None)
    assert seen == {True, False}


def test_coverage_oracle_agrees_with_execution(copies):
    # Training scores programs by coverage instead of executing them. That is only
    # legitimate if the two agree; check it on random (often wrong) vote patterns.
    rng = random.Random(3)
    for e, a in _sample(4, 60, 4):
        root = parse_nodes(e)
        votes = {id(n): rng.random() < 0.5 for n in internal_nodes(root)}
        decide = lambda n: votes[id(n)]
        ok, c_pred, _ = program_outcome(root, decide, copies)
        got, c = execute_dispatch_program(emit_program(root, decide), copies)
        assert ok == (got is not None and got == Fraction(int(a)))
        assert c <= c_pred                            # executor stops calling after a refusal


@pytest.mark.parametrize("mode", ["probe", "outcome", "feedback", "experience"])
def test_each_mode_takes_a_finite_step(copies, mode):
    tr = DispatchTrainer(copies, ModelConfig(d_model=32, n_heads=2, d_ff=64,
                                             recurrent_steps=2),
                         ArithmeticTokenizer(), depth=4, mode=mode, batch_size=8, group=2)
    before = [p.detach().clone() for p in tr.policy.head.parameters()]
    st = tr.train_step(0)
    assert math.isfinite(st["loss"])
    after = list(tr.policy.head.parameters())
    assert any((x - y).abs().sum() > 0 for x, y in zip(after, before)) or mode != "probe"
    ev = tr.evaluate(n=10)
    assert 0.0 <= ev["accuracy"] <= 1.0 and ev["oracle_agree"] == 1.0


def test_experience_learns_only_from_attempted_handoffs(copies):
    # The experience signal must come solely from subtrees the model actually handed off
    # while solving. A policy that never dispatches (all logits very negative, no
    # exploration) attempts nothing and so has nothing to learn from -- unlike probe,
    # which would still get a label for every node.
    tr = DispatchTrainer(copies, ModelConfig(d_model=32, n_heads=2, d_ff=64,
                                             recurrent_steps=2),
                         ArithmeticTokenizer(), depth=4, mode="experience",
                         batch_size=8, group=2, explore=0.0)
    exprs = tr._batch(8)
    roots = [parse_nodes(e) for e in exprs]
    never = [torch.full((len(internal_nodes(r)),), -50.0) for r in roots]
    _, st = tr._experience_loss(roots, never)
    assert st["tried"] == 0.0


def test_category_rates_cover_the_four_kinds_of_node(copies):
    tr = DispatchTrainer(copies, ModelConfig(d_model=32, n_heads=2, d_ff=64,
                                             recurrent_steps=2),
                         ArithmeticTokenizer(), depth=4, mode="experience", batch_size=8)
    rates = tr.category_rates(n=60)
    assert {"d2", "d3_bal", "d3_unbal", "d4+"} <= set(rates)
    assert all(0.0 <= r <= 1.0 for r, _ in rates.values())
