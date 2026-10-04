"""Train and evaluate the handicapped translator. ROADMAP 3c.

SmolLM2-135M, full fine-tune (at this size Unsloth's memory savings buy nothing on an A100).
Slots are added as single new tokens so the digit ban can be absolute. Generation runs with
two hard constraints: no token containing a digit, and only tokens of the sentence's closed
vocabulary (its input words, the glue list, its slots, punctuation). Every output is validated;
a failure falls back to the deterministic renderer. Loss is on the target sentence only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from typing import List

import torch

from .translator import GLUE, OP_CUES, SLOT_TOKENS, allowed_words, covered

MODEL = "HuggingFaceTB/SmolLM2-135M"
PUNCT = list(" .,;:!?$%/=+-*()'\"")


class ClosedVocab:
    """Logits processor: per row, only the token ids of that row's closed vocabulary survive."""

    def __init__(self, masks: torch.Tensor):
        self.masks = masks                  # (B, V) bool

    def __call__(self, input_ids, scores):
        return scores.masked_fill(~self.masks[: scores.size(0), : scores.size(1)], float("-inf"))


def vocab_mask(tok, prompt_text: str, n_vocab: int, digit_ids: set) -> torch.Tensor:
    ids = set(tok.convert_tokens_to_ids(SLOT_TOKENS)) | {tok.eos_token_id}
    for w in allowed_words(prompt_text):
        for form in (w, w.capitalize()):
            for variant in (form, " " + form):
                ids.update(tok.encode(variant, add_special_tokens=False))
    for p in PUNCT:
        ids.update(tok.encode(p, add_special_tokens=False))
    m = torch.zeros(n_vocab, dtype=torch.bool)
    m[[i for i in ids if i not in digit_ids and i < n_vocab]] = True
    return m


def load(path: str, limit=None) -> List[dict]:
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if covered(r["prompt"], r["target"])]
    return rows[:limit] if limit else rows


def main(argv=None):
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                              DataCollatorForSeq2Seq)

    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", default="translator_ckpt")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--eval-n", type=int, default=600)
    a = p.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.add_special_tokens({"additional_special_tokens": SLOT_TOKENS})
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
    model.resize_token_embeddings(len(tok))

    train = [r for f in a.train for r in load(f)]
    random.Random(0).shuffle(train)
    print(f"train pairs (closed-vocabulary covered): {len(train)}", flush=True)

    def encode(r):
        pi = tok.encode(r["prompt"], add_special_tokens=False)
        ti = tok.encode(r["target"], add_special_tokens=False) + [tok.eos_token_id]
        ids = (pi + ti)[-512:]
        return {"input_ids": ids, "labels": ([-100] * len(pi) + ti)[-512:]}

    ds = [encode(r) for r in train]
    total = int(-(-len(ds) // 64) * a.epochs)
    args = TrainingArguments(a.out, per_device_train_batch_size=64, num_train_epochs=a.epochs,
                             learning_rate=3e-4, warmup_steps=max(1, int(0.03 * total)),
                             lr_scheduler_type="cosine",
                             bf16=True, logging_steps=100, save_strategy="no", report_to=[])
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()
    model.save_pretrained(a.out); tok.save_pretrained(a.out)

    # ---- evaluation under the hard constraints ----
    test = load(a.test)
    random.Random(1).shuffle(test)
    test = test[: a.eval_n]
    digit_ids = {i for t, i in tok.get_vocab().items() if re.search(r"\d", t) and t not in SLOT_TOKENS}
    model.eval(); tok.padding_side = "left"
    n_vocab = model.get_output_embeddings().weight.size(0)

    def generate(prompts):
        outs = []
        for lo in range(0, len(prompts), 64):
            batch = prompts[lo:lo + 64]
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            masks = torch.stack([vocab_mask(tok, q, n_vocab, digit_ids) for q in batch]).cuda()
            g = model.generate(**enc, max_new_tokens=60, do_sample=False,
                               logits_processor=[ClosedVocab(masks)], pad_token_id=tok.eos_token_id)
            outs += tok.batch_decode(g[:, enc["input_ids"].size(1):], skip_special_tokens=False)
        return [o.replace(tok.eos_token, "").strip() for o in outs]

    def slots_ok(q, out):
        res = re.search(r"result: (<R\d+>)", q).group(1)
        allowed = set(re.findall(r"<[QKRN]\d+>", q))
        used = set(re.findall(r"<[QKRN]\d+>", out))
        return res in used and used <= allowed and not re.search(r"\d", re.sub(r"<[QKRN]\d+>", "", out))

    def cue(op, out):
        return bool(re.search(OP_CUES[op], out.lower()))

    prompts = [r["prompt"] for r in test]
    outs = generate(prompts)
    valid = [slots_ok(q, o) and covered(q, o) for q, o in zip(prompts, outs)]
    agree = [cue(r["op"], o) for r, o in zip(test, outs)]
    # Faithfulness: rotate each op to a different one; the sentence should follow the new op.
    words = {0: "add", 1: "subtract", 2: "multiply", 3: "divide"}
    swapped_ops = [(r["op"] + 1) % 4 for r in test]
    sw_prompts = [q.replace(f"op: {words[r['op']]}", f"op: {words[o]}", 1)
                  for q, r, o in zip(prompts, test, swapped_ops)]
    sw_outs = generate(sw_prompts)
    sw_agree = [cue(o, out) for o, out in zip(swapped_ops, sw_outs)]
    n = len(test)
    print(f"valid without fallback: {sum(valid)/n:.3f}", flush=True)
    print(f"op-cue agreement, true ops: {sum(agree)/n:.3f}; swapped ops: {sum(sw_agree)/n:.3f}", flush=True)
    for q, o, so in list(zip(prompts, outs, sw_outs))[:8]:
        print("----\n" + q + "\n=> " + o + "\n=> (op swapped) " + so, flush=True)


if __name__ == "__main__":
    main()
