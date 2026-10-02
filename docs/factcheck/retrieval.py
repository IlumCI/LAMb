"""Retrieval baseline: run the programs of the nearest training problems on the test problem's registers."""
import torch, time
from collections import Counter
from fractions import Fraction
from lamb.bridge_train import BridgeConfig, fetch_gsm8k, build_examples
from transformers import AutoModel, AutoTokenizer

cfg = BridgeConfig(); nc = len(cfg.constants); n_op = cfg.n_operands
OPS = "+-*/"
def ex_of(split):
    rows = fetch_gsm8k(split, cfg.cache_dir)
    ex, _ = build_examples(rows, cfg.n_operands, cfg.n_instr, cfg.constants, cfg.lexical)
    return ex
tr, te = ex_of("train"), ex_of("test")
tr = [e for e in tr if e.program]
tok = AutoTokenizer.from_pretrained(cfg.encoder); m = AutoModel.from_pretrained(cfg.encoder).eval()
torch.set_num_threads(12)
@torch.no_grad()
def emb(texts):
    out = []
    for i in range(0, len(texts), 256):
        t = tok(texts[i:i+256], padding=True, truncation=True, max_length=192, return_tensors="pt")
        h = m(**t).last_hidden_state; w = t["attention_mask"].unsqueeze(-1)
        v = (h*w).sum(1)/w.sum(1); out.append(torch.nn.functional.normalize(v, dim=-1))
    return torch.cat(out)
import os
C = os.path.expanduser("~/.cache/lamb/retrieval_emb.pt")
t0 = time.time()
if os.path.exists(C): E_tr, E_te = torch.load(C)
else:
    E_tr, E_te = emb([e.text for e in tr]), emb([e.text for e in te]); torch.save((E_tr, E_te), C)
print(f"encoded in {time.time()-t0:.0f}s")

def run(prog, vals, count):
    regs = list(vals)
    for (o, a, b) in prog:
        for x in (a, b):
            if x < n_op and x >= count: return None
        x, y = regs[a], regs[b]
        if o == 3 and y == 0: return None
        regs.append([x+y, x-y, x*y, x/y if o == 3 else None][o])
    return regs[-1]

sim = E_te @ E_tr.T
g = torch.Generator().manual_seed(0)
for k in (25, 100, 300, 1000):
    for mode in ("nearest", "random"):
        top = sim.topk(k, dim=1).indices if mode == "nearest" else \
            torch.stack([torch.randperm(len(tr), generator=g)[:k] for _ in te])
        vote = anyhit = 0
        for i, e in enumerate(te):
            answers = [run(tr[j].program, e.fractions, e.count) for j in top[i].tolist()]
            answers = [a for a in answers if a is not None]
            if answers:
                vote += Counter(answers).most_common(1)[0][0] == e.answer
                anyhit += e.answer in answers
        n = len(te)
        print(f"k={k:>4} {mode:>7}: majority vote {vote/n:.4f}   gold among the k answers {anyhit/n:.4f}", flush=True)
