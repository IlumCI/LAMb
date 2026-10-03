"""Ternary weights: what is quantised, that it is really three values, and that QAT still learns.

The failure shapes worth pinning: a target list that silently skips attention's packed
``in_proj_weight`` (it is not an ``nn.Linear``, so a module-type filter misses it and the
"ternary" model is partly float), and an STE that blocks the gradient (QAT then trains nothing
and the comparison to post-training rounding is meaningless).
"""

import torch
import torch.nn as nn

from lamb.ternary import coverage, int8, quantize_, targets, ternarize_qat, ternary


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 8)
        self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
        self.norm = nn.LayerNorm(8)
        self.out = nn.Linear(8, 3)

    def forward(self, x):
        h = self.emb(x)
        h = self.norm(h + self.attn(h, h, h, need_weights=False)[0])
        return self.out(h.mean(1))


def test_targets_include_packed_attention_and_skip_embeddings_and_norms():
    names = {n for _, n in targets(Tiny())}
    assert "in_proj_weight" in names
    m = Tiny()
    hit = {id(getattr(mod, n)) for mod, n in targets(m)}
    assert id(m.emb.weight) not in hit and id(m.norm.weight) not in hit
    assert 0 < coverage(m) < 1


def test_ternary_is_three_values_and_int8_is_close():
    torch.manual_seed(0)
    w = torch.randn(16, 32)
    assert len(torch.unique(ternary(w))) <= 3
    assert (int8(w) - w).abs().max() <= w.abs().max() / 127


def test_post_training_quantisation_rewrites_every_target():
    torch.manual_seed(0)
    m = quantize_(Tiny(), ternary)
    for mod, n in targets(m):
        assert len(torch.unique(getattr(mod, n))) <= 3


def test_qat_forward_is_ternary_and_still_learns():
    torch.manual_seed(0)
    m = ternarize_qat(Tiny())
    for mod, n in targets(m):
        assert len(torch.unique(getattr(mod, n).detach())) <= 3
    x, y = torch.randint(0, 10, (64, 5)), torch.randint(0, 3, (64,))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        loss = nn.functional.cross_entropy(m(x), y)
        first = first if first is not None else float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < first * 0.8         # gradient reaches the latent weights through the STE
