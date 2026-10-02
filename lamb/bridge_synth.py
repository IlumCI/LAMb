"""Train the language bridge on generated GSM8K-style problems plus real ones. ROADMAP 3c.

Pipeline, sized for one machine with a small GPU:

* **CPU workers** generate problems (:mod:`lamb.gsm_synth`), run them through the *real* GSM8K
  parser and aligner (:func:`lamb.bridge_train.build_examples`), and tokenise them with the frozen
  encoder's tokenizer, including each quantity's token anchor for the content pointer. Real GSM8K
  training rows with an aligned program are mixed in at ``real_frac``.
* **The GPU** runs the frozen encoder on the fly (no cache: 100k problems at 192 tokens would be
  ~15 GB) and trains on program logits only -- the exact executor is not in the training loop.
* **Evaluation** executes the decided program exactly on three sets, because generated-data
  accuracy alone would prove nothing (a model can memorise the *generator*):
  real GSM8K test (the target), generated problems from *held-out families* never seen in training
  (did it learn to read quantities and relations, or my templates?), and in-family generated test.
"""

from __future__ import annotations

import argparse
import random
import time
from fractions import Fraction
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .bridge import all_quantities
from .bridge_train import (BridgeConfig, BridgeReasoner, build_examples, fetch_gsm8k)
from .gsm_synth import FAMILIES, generate

HOLDOUT_FAMILIES = ("ages", "recipe", "combined_rate", "dozens")


def _anchors(tok_out, texts, cfg: BridgeConfig) -> List[List[int]]:
    """Token index of each operand register's quantity (same rule as ``quantity_anchors``)."""
    n_const, room = len(cfg.constants), cfg.n_operands - len(cfg.constants)
    out = []
    for i, t in enumerate(texts):
        offs = tok_out["offset_mapping"][i]
        row = [-1] * cfg.n_operands
        for k, q in enumerate(all_quantities(t, lexical=cfg.lexical)[:room]):
            for ti, (a, b) in enumerate(offs):
                if b > a and b > q.start:
                    row[n_const + k] = ti
                    break
        out.append(row)
    return out


class Stream(torch.utils.data.IterableDataset):
    """Endless batches of ``(texts, programs, values, counts, ids, mask, anchors)``."""

    def __init__(self, cfg: BridgeConfig, batch: int, real_rows: Sequence[dict],
                 real_frac: float, families: Sequence[str], seed: int):
        self.cfg, self.batch, self.real_rows = cfg, batch, list(real_rows)
        self.real_frac, self.families, self.seed = real_frac, list(families), seed

    def __iter__(self):
        from transformers import AutoTokenizer

        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(self.seed * 1009 + wid * 7919 + 1)
        fams = [f for f in FAMILIES if f.__name__ in self.families]
        tok = AutoTokenizer.from_pretrained(self.cfg.encoder)
        while True:
            rows = []
            for _ in range(self.batch):
                if self.real_rows and rng.random() < self.real_frac:
                    rows.append(rng.choice(self.real_rows))
                else:
                    rows.append(generate(rng, fams))
            ex, _ = build_examples(rows, self.cfg.n_operands, self.cfg.n_instr,
                                   self.cfg.constants, self.cfg.lexical)
            ex = [e for e in ex if e.program is not None]
            if not ex:
                continue
            texts = [e.text for e in ex]
            t = tok(texts, padding=True, truncation=True, max_length=self.cfg.max_len,
                    return_offsets_mapping=True, return_tensors="pt")
            yield {"ids": t["input_ids"], "mask": t["attention_mask"],
                   "anchors": torch.tensor(_anchors(t, texts, self.cfg)),
                   "counts": torch.tensor([e.count for e in ex]),
                   "programs": [e.program for e in ex],
                   "values": [e.fractions for e in ex],
                   "answers": [e.answer for e in ex]}


class Encoder:
    """The pretrained encoder, run on the fly on the GPU.

    ``unfreeze=0`` keeps every weight fixed and runs it in fp16. ``unfreeze=N`` trains the top
    ``N`` transformer layers: the weights stay fp32 and the forward runs under autocast. Dropout
    stays off, so frozen and unfrozen runs differ in exactly one thing: whether the top layers move.
    """

    def __init__(self, cfg: BridgeConfig, device: str, unfreeze: int = 0):
        from transformers import AutoModel, AutoTokenizer

        self.cfg, self.device, self.unfreeze = cfg, device, unfreeze
        self.tok = AutoTokenizer.from_pretrained(cfg.encoder)
        self.model = AutoModel.from_pretrained(cfg.encoder).to(device).eval()
        self.model.requires_grad_(False)
        if unfreeze:
            for layer in self.model.encoder.layer[-unfreeze:]:
                layer.requires_grad_(True)
        else:
            self.model.half()

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def __call__(self, ids, mask):
        ids, mask = ids.to(self.device), mask.to(self.device)
        if self.unfreeze and torch.is_grad_enabled():
            with torch.autocast(self.device, dtype=torch.float16,
                                enabled=self.device == "cuda"):
                h = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        else:
            with torch.no_grad(), torch.autocast(self.device, dtype=torch.float16,
                                                 enabled=self.device == "cuda" and
                                                 bool(self.unfreeze)):
                h = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        return h.float(), mask == 0

    def batch(self, texts: Sequence[str]):
        t = self.tok(list(texts), padding=True, truncation=True, max_length=self.cfg.max_len,
                     return_offsets_mapping=True, return_tensors="pt")
        return t, torch.tensor(_anchors(t, list(texts), self.cfg))


def program_loss(model: BridgeReasoner, logits, programs) -> torch.Tensor:
    mask = torch.ones(len(programs), device=logits[0].device)
    return model.machine.program_loss(logits, programs, mask)


@torch.no_grad()
def exact_match(model: BridgeReasoner, encoder: Encoder, examples, batch: int = 128) -> float:
    """Execute the argmax program for each example; fraction whose answer is exact."""
    from .regmachine import run_program

    model.eval()
    ok, sysm, alg = 0, model.machine.sys, model.machine.alg
    out = model.cfg.n_operands + model.cfg.n_instr - 1
    for lo in range(0, len(examples), batch):
        rows = examples[lo:lo + batch]
        t, anc = encoder.batch([r.text for r in rows])
        enc, pad = encoder(t["input_ids"], t["attention_mask"])
        cnt = torch.tensor([r.count for r in rows], device=encoder.device)
        op_l, a_l, b_l = model.logits(enc, pad, cnt, anc.to(encoder.device))
        oi, ai, bi = op_l.argmax(-1), a_l.argmax(-1), b_l.argmax(-1)
        progs = [[(int(oi[i, s]), int(ai[i, s]), int(bi[i, s]))
                  for s in range(model.cfg.n_instr)] for i in range(len(rows))]
        regs = run_program(model.machine, [r.fractions for r in rows], progs, "cpu")
        nums = sysm.decode(alg.unpack(regs.num[:, out]))
        dens = sysm.decode(alg.unpack(regs.den[:, out]))
        for r, n_, d in zip(rows, nums, dens):
            ok += d != 0 and Fraction(n_, d) == r.answer
    model.train()
    return ok / max(1, len(examples))


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--real-frac", type=float, default=0.5,
                   help="share of each batch drawn from real GSM8K train (0 = generated only)")
    p.add_argument("--no-synth", action="store_true", help="real GSM8K only (the baseline)")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None)
    p.add_argument("--unfreeze", type=int, default=0,
                   help="train the top N encoder layers (0 = frozen encoder)")
    p.add_argument("--enc-lr", type=float, default=3e-5)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--train-jsonl", default=None,
                   help="extra GSM8K-format rows ({question, answer}) added to the real pool")
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = BridgeConfig(content_pointer=True, d_model=a.d_model, lr=a.lr, device=dev,
                       answer_coef=0.0)
    train_fams = [f.__name__ for f in FAMILIES if f.__name__ not in HOLDOUT_FAMILIES]
    real = [r for r, e in zip(*_real_rows(cfg, "train")) if e.program is not None]
    if a.train_jsonl:
        import json
        extra = [json.loads(line) for line in open(a.train_jsonl)]
        ex, stats = build_examples(extra, cfg.n_operands, cfg.n_instr, cfg.constants, cfg.lexical)
        # build_examples drops rows whose answer does not parse, so rows and examples are not
        # positionally aligned; match by question text instead of zipping.
        ok = {e.text for e in ex if e.program is not None}
        kept = [r for r in extra if r["question"] in ok]
        print(f"{a.train_jsonl}: {len(kept)}/{len(extra)} rows align to a program", flush=True)
        real += kept
    real_frac = 1.0 if a.no_synth else a.real_frac
    stream = Stream(cfg, a.batch, real, real_frac, train_fams, a.seed)
    loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=a.workers,
                                         prefetch_factor=4, persistent_workers=True)
    encoder = Encoder(cfg, dev, a.unfreeze)
    model = BridgeReasoner(cfg, encoder.model.config.hidden_size, dev).to(dev)
    groups = [{"params": list(model.parameters()), "lr": a.lr}]
    if a.unfreeze:
        groups.append({"params": encoder.parameters(), "lr": a.enc_lr})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    trainable = list(model.parameters()) + encoder.parameters()

    test_rows, test_ex = _real_rows(cfg, "test")
    rng = random.Random(99)
    held = [generate(rng, [f for f in FAMILIES if f.__name__ in HOLDOUT_FAMILIES])
            for _ in range(600)]
    infam = [generate(rng, [f for f in FAMILIES if f.__name__ in train_fams]) for _ in range(600)]
    held_ex, _ = build_examples(held, cfg.n_operands, cfg.n_instr, cfg.constants, cfg.lexical)
    infam_ex, _ = build_examples(infam, cfg.n_operands, cfg.n_instr, cfg.constants, cfg.lexical)
    print(f"real train rows with programs: {len(real)}; mix real_frac={real_frac}; "
          f"held-out families {HOLDOUT_FAMILIES}; d_model={a.d_model}; "
          f"unfrozen encoder layers {a.unfreeze} ({sum(p.numel() for p in encoder.parameters())} "
          f"params at lr {a.enc_lr})", flush=True)

    t0, it = time.time(), iter(loader)
    for step in range(1, a.steps + 1):
        b = next(it)
        enc, pad = encoder(b["ids"], b["mask"])
        logits = model.logits(enc, pad, b["counts"].to(dev), b["anchors"].to(dev))
        loss = program_loss(model, logits, b["programs"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        opt.step()
        if step % a.log_every == 0:      # a heartbeat the watchdog reads: a stalled run is visible
            print(f"step {step:>6}  loss {float(loss.detach()):.4f}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            gsm = exact_match(model, encoder, test_ex)
            ho = exact_match(model, encoder, held_ex)
            inf = exact_match(model, encoder, infam_ex)
            print(f"step {step:>6}  loss {float(loss.detach()):.4f}  GSM8K-test {gsm:.4f}  "
                  f"held-out-families {ho:.3f}  in-family {inf:.3f}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            if a.save:        # checkpoint at every evaluation: a long run must survive the machine
                torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                            "encoder": encoder.model.state_dict() if a.unfreeze else None,
                            "opt": opt.state_dict(), "step": step}, a.save)
    if a.save:
        torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                    "encoder": encoder.model.state_dict() if a.unfreeze else None}, a.save)


def _real_rows(cfg: BridgeConfig, split: str):
    rows = fetch_gsm8k(split, cfg.cache_dir)
    ex, _ = build_examples(rows, cfg.n_operands, cfg.n_instr, cfg.constants, cfg.lexical)
    return rows, ex


if __name__ == "__main__":
    main()
