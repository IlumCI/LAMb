"""LAMb engine: train from answers alone, compile into exact programs, solve, verify.

The runnable package of what this repo has shown works (ROADMAP 3e-ii, 3g):

* ``train``   -- a register-machine core learns to write programs from final answers only:
                 expert iteration over its own verified samples, a fresh-operand check that
                 rejects coincidences, and self-composition of deeper programs from shallower
                 ones. No gold program, no gold trace. Reached 1.000 at depths 1-3 on 5/5 seeds.
* ``compile`` -- the core reads its own programs off itself into a JSON library, one program per
                 structure, each kept only if it reproduces the right answer on its own instance
                 *and* on fresh operands (operand invariance is what makes a copy exact).
* ``solve``   -- compiled program if the structure is in the library (no network involved),
                 otherwise the neural core, otherwise a refusal.
* ``verify``  -- recheck any (expression, program, answer) with plain integer arithmetic;
                 see :mod:`lamb.progexec`, which needs nothing but the standard library.

Scope, stated plainly: balanced arithmetic expressions over the operators the core was trained
on, at the depths it was trained on. Anything else is refused rather than guessed.

    python -m lamb.engine train   --depths 1 2 3 --steps 1000 --ckpt engine.pt
    python -m lamb.engine compile --ckpt engine.pt --library engine.json
    python -m lamb.engine solve   --library engine.json [--ckpt engine.pt] "(12+7)-(30-4)"
    python -m lamb.engine bench   --library engine.json --ckpt engine.pt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from . import progexec

FORMAT = 1


# -- training ------------------------------------------------------------------
def build_trainer(depths: Sequence[int], digits: int = 2, d_model: int = 64,
                  batch_size: int = 24, group: int = 6, seed: int = 0, lr: float = 2e-3,
                  ops_key: int = 0, steps: int = 1000):
    """The configuration that reached 1.000 at depths 1-3 (ROADMAP 3e-ii)."""
    import torch

    from . import ArithmeticTokenizer, LotusConfig
    from .config import ModelConfig
    from .program_grpo import GRPOTrainer

    torch.manual_seed(seed)
    cfg = LotusConfig(steps=steps, batch_size=batch_size, n_latent=max(8, 2 ** max(depths)),
                      loops=3, depth=max(depths), digits=digits, ops_key=ops_key, seed=seed,
                      eval_tasks=128, trace_coef=0.0, alu_coef=0.0, use_boundaries=False,
                      switch_coef=0.0, device="cpu", lr=lr, alu_moduli=(16, 25, 27, 11, 37))
    mc = ModelConfig(d_model=d_model, n_heads=4, d_ff=2 * d_model, recurrent_steps=3)
    return GRPOTrainer(cfg, ArithmeticTokenizer(), depths=list(depths), model_cfg=mc,
                       group=group, process_coef=0.0, mode="rft", rft_verify=True,
                       compose=True, seed=seed)


def train(trainer, steps: int, log_every: int = 250, log=print) -> Dict[str, float]:
    t0 = time.time()
    ev: Dict[str, float] = {}
    for s in range(steps):
        trainer.train_step(s)
        if (s + 1) % log_every == 0 or s == steps - 1:
            ev = trainer.evaluate(n=128)
            log(f"step {s + 1:>5}  " + "  ".join(f"{k}={v:.3f}" for k, v in ev.items()
                                                  if k.startswith("acc_hard_d"))
                + f"  [{time.time() - t0:.0f}s]")
    return ev


def save(trainer, path: str) -> None:
    import torch

    torch.save({"format": FORMAT, "depths": trainer.depths, "digits": trainer.digits,
                "lotus_cfg": asdict(trainer.cfg),
                "model_cfg": asdict(trainer.rmt.inner.reasoner.model.cfg),
                "reasoner": trainer.rmt.inner.reasoner.state_dict(),
                "machine": trainer.machine.state_dict()}, path)


def load(path: str):
    """Rebuild a trained core from a checkpoint written by :func:`save`."""
    import torch

    from . import ArithmeticTokenizer, LotusConfig
    from .config import ModelConfig
    from .program_grpo import GRPOTrainer

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != FORMAT:
        raise ValueError(f"unknown checkpoint format {ck.get('format')!r}")
    cfg = LotusConfig(**ck["lotus_cfg"])
    mc = ModelConfig(**ck["model_cfg"])
    tr = GRPOTrainer(cfg, ArithmeticTokenizer(), depths=ck["depths"], model_cfg=mc,
                     process_coef=0.0, mode="rft", digits=ck["digits"])
    tr.rmt.inner.reasoner.load_state_dict(ck["reasoner"])
    tr.machine.load_state_dict(ck["machine"])
    tr.rmt.inner.reasoner.eval()
    tr.machine.eval()
    return tr


# -- the core's own programs -------------------------------------------------------
def spec_of(trainer) -> Dict:
    n_const = trainer.rmt.n_const
    return {"ops": list(trainer.machine.ops), "n_operands": trainer.n_operands,
            "n_const": n_const, "constants": [1] * n_const, "n_instr": trainer.n_instr}


def core_programs(trainer, exprs: Sequence[str]) -> List[Optional[List[List[int]]]]:
    """The neural core's argmax program for each expression, trimmed to what its depth uses.

    ``None`` for an expression the core's machine cannot hold (too deep or too many operands).
    """
    import torch

    out: List[Optional[List[List[int]]]] = [None] * len(exprs)
    ok = []
    for i, e in enumerate(exprs):
        t = progexec.parse(e)
        if (progexec.depth(t) in trainer.depths
                and len(progexec.leaves(t)) + trainer.rmt.n_const <= trainer.n_operands):
            ok.append(i)
    if not ok:
        return out
    tasks = [(exprs[i], "0", []) for i in ok]
    with torch.no_grad():
        trainer.rmt._prepare(tasks)
        lh = trainer.rmt._latents(tasks)
        cnt = torch.tensor(trainer.rmt._counts, device=trainer.device)
        op_l, a_l, b_l = trainer.machine.logits(lh, cnt)
        oi, ai, bi = op_l.argmax(-1), a_l.argmax(-1), b_l.argmax(-1)
    for j, i in enumerate(ok):
        n_used = 2 ** progexec.depth(progexec.parse(exprs[i])) - 1
        out[i] = [[int(oi[j, t]), int(ai[j, t]), int(bi[j, t])] for t in range(n_used)]
    return out


def _fresh(t, digits: int, rng: random.Random):
    """The same structure with new operands -- the check that a program is not a fluke."""
    lo, hi = (0, 9) if digits <= 1 else (10 ** (digits - 1), 10 ** digits - 1)
    if progexec.is_leaf(t):
        return (t[0], rng.randint(lo, hi), rng.randint(lo, hi))
    return (t[0], _fresh(t[1], digits, rng), _fresh(t[2], digits, rng))


def program_holds(program, t, spec: Dict, digits: int, rng: random.Random,
                  n_fresh: int = 3) -> bool:
    """Right on this instance and on ``n_fresh`` re-drawn instances of its structure."""
    for inst in [t] + [_fresh(t, digits, rng) for _ in range(n_fresh)]:
        try:
            if progexec.run(program, progexec.leaves(inst), spec) != progexec.evaluate(inst):
                return False
        except (ValueError, ZeroDivisionError):
            return False
    return True


def compile_library(trainer, n_per_depth: int = 3000, batch: int = 256, seed: int = 0,
                    log=print) -> Dict:
    """Read the core's programs off itself into a verified, portable library.

    For each trained depth, run the core over ``n_per_depth`` problems from its training
    partition, vote the programs it emits per structure, and keep the most common one that
    holds on its own instance and on fresh operands. A structure whose programs never hold is
    left out -- the library refuses it rather than storing something that only looked right.
    """
    rng = random.Random(seed)
    spec = spec_of(trainer)
    votes: Dict[str, Counter] = {}
    examples: Dict[str, tuple] = {}
    depth_of: Dict[str, int] = {}
    for d in trainer.depths:
        seen = 0
        while seen < n_per_depth:
            tasks = trainer._sample_batch(d, min(batch, n_per_depth - seen))
            seen += len(tasks)
            exprs = [t[0] for t in tasks]
            for e, p in zip(exprs, core_programs(trainer, exprs)):
                if p is None:
                    continue
                t = progexec.parse(e)
                k = progexec.key(t)
                votes.setdefault(k, Counter())[json.dumps(p)] += 1
                examples.setdefault(k, t)
                depth_of[k] = d
    programs, rejected = {}, 0
    for k, c in votes.items():
        for prog_s, _ in c.most_common():
            prog = json.loads(prog_s)
            if program_holds(prog, examples[k], spec, trainer.digits, rng):
                programs[k] = {"depth": depth_of[k], "program": prog}
                break
        else:
            rejected += 1
    log(f"compiled {len(programs)} structures across depths {trainer.depths}; "
        f"{rejected} structures had no program that held on fresh operands")
    return {"format": FORMAT, "spec": spec, "programs": programs,
            "meta": {"depths": trainer.depths, "digits": trainer.digits,
                     "n_per_depth": n_per_depth}}


# -- serving -------------------------------------------------------------------------
def solve(expr: str, library: Optional[Dict] = None, trainer=None) -> Dict:
    """Compiled program first (no network involved), then the core, else a refusal.

    Returns ``{"value", "source", "program"}``; ``value`` is ``None`` on a refusal.
    """
    if library is not None:
        prog, t = progexec.lookup(library, expr)
        if prog is not None:
            return {"value": progexec.run(prog, progexec.leaves(t), library["spec"]),
                    "source": "compiled", "program": prog}
    if trainer is not None:
        prog = core_programs(trainer, [expr])[0]
        if prog is not None:
            t = progexec.parse(expr)
            return {"value": progexec.run(prog, progexec.leaves(t), spec_of(trainer)),
                    "source": "core", "program": prog}
    return {"value": None, "source": "refused", "program": None}


def bench(library: Dict, trainer=None, n: int = 600, seed: int = 4242, log=print) -> Dict:
    """Held-out accuracy and cost of the compiled library against the neural core."""
    from .selfplay.grammar import Descriptor, TaskGrammar

    g, rng = TaskGrammar(), random.Random(seed)
    depths = library["meta"]["depths"]
    digits = library["meta"]["digits"]
    probs = []
    for i in range(n):
        d = depths[i % len(depths)]
        probs.append(g.sample_heldout(Descriptor(d, digits, 0, 0), rng.randint(0, 2 ** 31 - 1)))
    t0 = time.perf_counter()
    lib_res = [solve(e, library=library) for e, _ in probs]
    t_lib = time.perf_counter() - t0
    out = {"n": n, "library_coverage": sum(r["value"] is not None for r in lib_res) / n,
           "library_acc": sum(r["value"] == Fraction(int(a)) for r, (_, a) in
                              zip(lib_res, probs)) / n,
           "library_ms_per_problem": 1000 * t_lib / n}
    if trainer is not None:
        t0 = time.perf_counter()
        progs = core_programs(trainer, [e for e, _ in probs])
        t_core = time.perf_counter() - t0
        spec = spec_of(trainer)
        core_ok = 0
        for p, (e, a) in zip(progs, probs):
            if p is not None:
                core_ok += progexec.run(p, progexec.leaves(progexec.parse(e)), spec) == Fraction(int(a))
        out.update({"core_acc": core_ok / n, "core_ms_per_problem": 1000 * t_core / n})
    log(json.dumps(out, indent=1))
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="LAMb engine: train, compile, solve, verify")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train a core from answers alone")
    t.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    t.add_argument("--digits", type=int, default=2)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--d-model", type=int, default=64)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--threads", type=int, default=1)
    t.add_argument("--ckpt", required=True)
    c = sub.add_parser("compile", help="compile a trained core into a JSON program library")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--library", required=True)
    c.add_argument("--n", type=int, default=3000, help="problems per depth")
    s = sub.add_parser("solve", help="answer expressions")
    s.add_argument("--library")
    s.add_argument("--ckpt")
    s.add_argument("exprs", nargs="+")
    b = sub.add_parser("bench", help="held-out accuracy and speed")
    b.add_argument("--library", required=True)
    b.add_argument("--ckpt")
    b.add_argument("--n", type=int, default=600)
    a = p.parse_args(argv)

    if a.cmd == "train":
        import torch

        torch.set_num_threads(a.threads)
        tr = build_trainer(a.depths, digits=a.digits, d_model=a.d_model, seed=a.seed,
                           steps=a.steps)
        train(tr, a.steps)
        save(tr, a.ckpt)
        print(f"saved {a.ckpt}")
    elif a.cmd == "compile":
        lib = compile_library(load(a.ckpt), n_per_depth=a.n)
        with open(a.library, "w") as f:
            json.dump(lib, f)
        print(f"wrote {a.library}")
    elif a.cmd == "solve":
        lib = progexec.load(a.library) if a.library else None
        tr = load(a.ckpt) if a.ckpt else None
        for e in a.exprs:
            r = solve(e, lib, tr)
            print(f"{e} = {r['value']}  [{r['source']}]" if r["value"] is not None
                  else f"{e}: refused")
    elif a.cmd == "bench":
        bench(progexec.load(a.library), load(a.ckpt) if a.ckpt else None, n=a.n)


if __name__ == "__main__":
    main()
