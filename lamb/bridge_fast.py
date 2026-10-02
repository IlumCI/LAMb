"""The bridge trainer with the frozen encoder run once, and a paired multi-seed study. ROADMAP 3c.

:mod:`lamb.bridge_synth` re-runs the frozen encoder on every batch and builds every batch in
DataLoader workers. With the encoder frozen that is wasted work: its output for a problem never
changes. Here every problem is aligned, tokenised and encoded **once**. The token states live in one
flat fp16 tensor on the GPU, and a batch is a single gather. What remains per step is the 3.4M
trainable head, which is small enough that launch overhead dominates, so the loss step is
``torch.compile``-d with sequence lengths bucketed to a few fixed sizes.

The model, the aligner, the anchors and the exact evaluation are the ones in
:mod:`lamb.bridge_train` and :mod:`lamb.bridge_synth`. Only the data path and the step change, so
results from both trainers are comparable.

``--study`` runs paired arms over seeds: the same seed, model init and sampler for ``aug``
(GSM8K train plus GSM8K-Aug) and ``real`` (GSM8K train only), compared by the exact paired
permutation test in :mod:`lamb.study`.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from fractions import Fraction
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .bridge import all_quantities
from .bridge_synth import HOLDOUT_FAMILIES
from .bridge_train import BridgeConfig, BridgeReasoner, build_examples, fetch_gsm8k
from .gsm_synth import FAMILIES, generate

BUCKET = 32       # sequence lengths round up to a multiple of this: few shapes, few compiles


class Bank:
    """Aligned problems with their encoder states precomputed, ready to gather on the GPU."""

    def __init__(self, examples, cfg: BridgeConfig, tok, enc_model, device: str,
                 batch: int = 1024):
        self.ex, self.cfg, self.device = list(examples), cfg, device
        n_const, room = len(cfg.constants), cfg.n_operands - len(cfg.constants)
        texts = [e.text for e in self.ex]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # length-sorted batches
        lengths = torch.zeros(len(texts), dtype=torch.long)
        anchors = torch.full((len(texts), cfg.n_operands), -1, dtype=torch.long)
        chunks, offsets, total = [], torch.zeros(len(texts), dtype=torch.long), 0
        enc_model.eval()
        for lo in range(0, len(order), batch):
            idx = order[lo:lo + batch]
            t = tok([texts[i] for i in idx], padding=True, truncation=True,
                    max_length=cfg.max_len, return_offsets_mapping=True, return_tensors="pt")
            with torch.no_grad():
                h = enc_model(input_ids=t["input_ids"].to(device),
                              attention_mask=t["attention_mask"].to(device)).last_hidden_state
            lens = t["attention_mask"].sum(1)
            for j, i in enumerate(idx):
                L = int(lens[j])
                chunks.append(h[j, :L].half())
                offsets[i], lengths[i] = total, L
                total += L
                # Same rule as bridge_synth._anchors: first token ending past the quantity start.
                ends = t["offset_mapping"][j, :L].numpy()
                good = ends[:, 1] > ends[:, 0]
                for k, q in enumerate(all_quantities(texts[i], lexical=cfg.lexical)[:room]):
                    hit = np.nonzero(good & (ends[:, 1] > q.start))[0]
                    if len(hit):
                        anchors[i, n_const + k] = int(hit[0])
        self.flat = torch.cat(chunks).to(device)
        self.off, self.len = offsets.to(device), lengths.to(device)
        self.anchors = anchors.to(device)
        self.counts = torch.tensor([e.count for e in self.ex], device=device)
        prog = [e.program if e.program is not None else [(0, 0, 0)] * cfg.n_instr
                for e in self.ex]
        self.prog = torch.tensor(prog, dtype=torch.long, device=device)   # (N, n_instr, 3)

    def __len__(self):
        return len(self.ex)

    def gather(self, idx: torch.Tensor):
        L = self.len[idx]
        tb = int(L.max())
        tb = min(-(-tb // BUCKET) * BUCKET, self.cfg.max_len)
        ar = torch.arange(tb, device=self.device)
        valid = ar.view(1, -1) < L.view(-1, 1)
        pos = torch.where(valid, self.off[idx].view(-1, 1) + ar.view(1, -1), 0)
        return (self.flat[pos].float(), ~valid, self.counts[idx], self.anchors[idx],
                self.prog[idx])


def make_step(model: BridgeReasoner, compile_: bool):
    def loss_fn(enc, pad, cnt, anc, prog):
        op_l, a_l, b_l = model.logits(enc, pad, cnt, anc)

        def ce(logit, target):
            return F.cross_entropy(logit.reshape(-1, logit.size(-1)), target.reshape(-1))

        return (ce(op_l, prog[..., 0]) + ce(a_l, prog[..., 1]) + ce(b_l, prog[..., 2])) / 3.0

    return torch.compile(loss_fn, dynamic=False) if compile_ else loss_fn


@torch.no_grad()
def exact_match(model: BridgeReasoner, bank: Bank, batch: int = 512) -> float:
    """Execute the argmax program exactly; fraction whose answer matches. Same as bridge_synth."""
    from .regmachine import run_program

    model.eval()
    ok, sysm, alg = 0, model.machine.sys, model.machine.alg
    out = model.cfg.n_operands + model.cfg.n_instr - 1
    for lo in range(0, len(bank), batch):
        idx = torch.arange(lo, min(lo + batch, len(bank)), device=bank.device)
        enc, pad, cnt, anc, _ = bank.gather(idx)
        op_l, a_l, b_l = model.logits(enc, pad, cnt, anc)
        progs = torch.stack([op_l.argmax(-1), a_l.argmax(-1), b_l.argmax(-1)], -1).tolist()
        rows = bank.ex[lo:lo + len(idx)]
        regs = run_program(model.machine, [r.fractions for r in rows], progs, "cpu")
        nums = sysm.decode(alg.unpack(regs.num[:, out]))
        dens = sysm.decode(alg.unpack(regs.den[:, out]))
        for r, n_, d in zip(rows, nums, dens):
            ok += d != 0 and Fraction(n_, d) == r.answer
    model.train()
    return ok / max(1, len(bank))


def train_one(cfg: BridgeConfig, train: Bank, pool: torch.Tensor, evals: Dict[str, Bank],
              steps: int, batch: int, seed: int, eval_at: Sequence[int], compile_: bool,
              d_enc: int, log=print) -> Dict[str, object]:
    torch.manual_seed(seed)
    model = BridgeReasoner(cfg, d_enc, cfg.device).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step_fn = make_step(model, compile_)
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    hist, t0 = [], time.time()
    for step in range(1, steps + 1):
        pick = pool[torch.randint(len(pool), (batch,), device=cfg.device, generator=gen)]
        loss = step_fn(*train.gather(pick))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step % 500 == 0:
            log(f"  seed {seed} step {step} loss {float(loss):.4f} [{time.time() - t0:.0f}s]")
        if step in eval_at:
            r = {"step": step, "loss": float(loss), **{k: exact_match(model, b)
                                                       for k, b in evals.items()}}
            hist.append(r)
            log(f"  seed {seed} " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else
                                              f"{k} {v}" for k, v in r.items()))
    return {"seed": seed, "history": hist, "seconds": time.time() - t0}


def build_banks(cfg: BridgeConfig, aug_jsonl: Optional[str], log=print,
                limit: Optional[int] = None):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.encoder)
    enc_model = AutoModel.from_pretrained(cfg.encoder).to(cfg.device).half()
    t0 = time.time()
    real_rows = fetch_gsm8k("train", cfg.cache_dir)
    real_ex = [e for e in build_examples(real_rows, cfg.n_operands, cfg.n_instr, cfg.constants,
                                         cfg.lexical)[0] if e.program is not None]
    aug_ex = []
    if aug_jsonl:
        rows = [json.loads(line) for line in open(aug_jsonl)][:limit]
        aug_ex = [e for e in build_examples(rows, cfg.n_operands, cfg.n_instr, cfg.constants,
                                            cfg.lexical)[0] if e.program is not None]
    log(f"aligned: real {len(real_ex)}, aug {len(aug_ex)} [{time.time() - t0:.0f}s]")
    train = Bank(real_ex + aug_ex, cfg, tok, enc_model, cfg.device)
    log(f"train bank: {len(train)} problems, {train.flat.shape[0]} token states, "
        f"{train.flat.numel() * 2 / 2**30:.1f} GiB [{time.time() - t0:.0f}s]")
    test_ex = build_examples(fetch_gsm8k("test", cfg.cache_dir), cfg.n_operands, cfg.n_instr,
                             cfg.constants, cfg.lexical)[0]
    rng = random.Random(99)     # identical eval sets to bridge_synth
    train_fams = [f.__name__ for f in FAMILIES if f.__name__ not in HOLDOUT_FAMILIES]
    held = [generate(rng, [f for f in FAMILIES if f.__name__ in HOLDOUT_FAMILIES])
            for _ in range(600)]
    infam = [generate(rng, [f for f in FAMILIES if f.__name__ in train_fams]) for _ in range(600)]
    evals = {name: Bank(build_examples(rows, cfg.n_operands, cfg.n_instr, cfg.constants,
                                       cfg.lexical)[0], cfg, tok, enc_model, cfg.device)
             for name, rows in (("held", held), ("infam", infam))}
    evals = {"gsm8k": Bank(test_ex, cfg, tok, enc_model, cfg.device), **evals}
    log(f"eval banks: " + ", ".join(f"{k} {len(v)}" for k, v in evals.items()))
    d_enc = enc_model.config.hidden_size
    del enc_model
    torch.cuda.empty_cache()
    return train, len(real_ex), evals, d_enc


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--aug-jsonl", default=None)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--arms", nargs="+", default=["aug"], choices=["aug", "real"])
    p.add_argument("--eval-at", type=int, nargs="+", default=[4000, 8000])
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--aug-limit", type=int, default=None, help="first N aug rows only (smoke runs)")
    p.add_argument("--out", default="bridge_fast.json")
    a = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = BridgeConfig(content_pointer=True, d_model=256, lr=3e-4, device=dev, answer_coef=0.0)
    train, n_real, evals, d_enc = build_banks(cfg, a.aug_jsonl, limit=a.aug_limit)
    pools = {"real": torch.arange(n_real, device=dev), "aug": torch.arange(len(train), device=dev)}
    results: List[Dict[str, object]] = []
    for seed in a.seeds:                       # seed-major: each pair completes before the next
        for arm in a.arms:
            print(f"arm {arm} seed {seed}: pool {len(pools[arm])}", flush=True)
            r = train_one(cfg, train, pools[arm], evals, a.steps, a.batch, seed,
                          set(a.eval_at), not a.no_compile, d_enc,
                          log=lambda s: print(s, flush=True))
            results.append({"arm": arm, **r})
            json.dump(results, open(a.out, "w"), indent=1)
    if set(a.arms) == {"aug", "real"}:
        report(results)


def report(results: List[Dict[str, object]]) -> None:
    from .study import _perm_p_paired

    final = {(r["arm"], r["seed"]): r["history"][-1] for r in results}
    seeds = sorted({s for _, s in final if ("aug", s) in final and ("real", s) in final})
    for key in ("gsm8k", "held", "infam"):
        aug = [final[("aug", s)][key] for s in seeds]
        real = [final[("real", s)][key] for s in seeds]
        d = [x - y for x, y in zip(aug, real)]
        print(f"{key:>6}: aug {np.mean(aug):.4f} (sd {np.std(aug, ddof=1):.4f})  "
              f"real {np.mean(real):.4f} (sd {np.std(real, ddof=1):.4f})  "
              f"paired diff {np.mean(d):+.4f}  aug>real {sum(x > 0 for x in d)}/{len(d)}  "
              f"exact paired permutation p={_perm_p_paired(d):.4f}", flush=True)


if __name__ == "__main__":
    main()
