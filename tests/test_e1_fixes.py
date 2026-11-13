"""
Regression tests for the E1 switches (contrastive_valid_mask, gat_keep_self_loop,
ceca_mask_padding).

    F:\\anaconda_workspace\\envs\\fuxian\\python.exe tests\\test_e1_fixes.py

1. Backward compatibility: all switches off -> official Full checkpoints reproduce the
   per-sample P(fake) saved by the read-only audit before any code change
   (experiments/audit_20261015/preds_ref_<ds>_test.csv, rounded to 4 decimals).
2. CECA padding mask == running CECA on the valid evidence rows only; all-zero evidence
   stays finite.
3. InfoNCE valid_mask == loss on the selected subset; <2 valid samples -> 0, backward ok.
4. GAT keep_self_loop: a self-loop-only graph never collapses to zero under dropout.
5. Real FakeTT batch, train mode, all switches on: finite loss and no gradient explosion
   in ClaimGAT (the original code gives ~1e11 on this batch).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
AUDIT = ROOT / "experiments" / "audit_20261015"
sys.argv = [sys.argv[0], "--log-dir", str(AUDIT / "_test_logdir")]

import train as T  # noqa: E402
from forward_check import REF_RUN, forward, make_loader  # noqa: E402
from model.ceca import CECA  # noqa: E402
from model.contrastive_loss import InfoNCELoss  # noqa: E402
from model.gat import ClaimGAT  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = {"FakeSV": ROOT / "configs/fakesv_v2/best.yaml", "FakeTT": ROOT / "configs/fakett_v2/full.yaml"}
E1_ON = {"contrastive_valid_mask": True, "gat_keep_self_loop": True, "ceca_mask_padding": True}


def load_cfg(ds, **model_overrides):
    cfg = yaml.safe_load(open(CFG[ds], encoding="utf-8"))
    cfg["model"].update(model_overrides)
    return cfg


def built_model(cfg, batch):
    T.set_seed(cfg["seed"])
    m = T.build_model(cfg).to(DEV)
    with torch.no_grad():
        forward(m.eval(), batch, cfg, DEV)   # materialise LazyLinear
    return m


def test_backward_compat():
    for ds in CFG:
        cfg = load_cfg(ds)
        loader = make_loader(cfg, "test")
        m = built_model(cfg, next(iter(loader)))
        m.load_state_dict(torch.load(ROOT / REF_RUN[ds] / "best_model.pth", map_location=DEV, weights_only=False))
        m.eval()
        probs, vids = [], []
        with torch.no_grad():
            for b in loader:
                probs.append(torch.softmax(forward(m, b, cfg, DEV)["pred"], 1)[:, 1].cpu())
                vids += list(b["vids"])
        ref = pd.read_csv(AUDIT / f"preds_ref_{ds}_test.csv", dtype={"vid": str})
        assert ref.vid.tolist() == vids, f"{ds}: sample order changed"
        diff = np.abs(torch.cat(probs).numpy() - ref.p_fake.values).max()
        assert diff < 2e-4, f"{ds}: max |dP| = {diff}"
        print(f"  [ok] {ds} flags-off reproduces audit predictions (max |dP|={diff:.1e})")


def test_ceca_mask():
    torch.manual_seed(0)
    ceca = CECA(dim=64, num_heads=4, dropout=0.0, mask_padding=True).eval()
    ref = CECA(dim=64, num_heads=4, dropout=0.0, mask_padding=False).eval()
    ref.load_state_dict(ceca.state_dict())
    claims = torch.randn(3, 64)
    ev = torch.zeros(5, 64)
    ev[:2] = torch.randn(2, 64)
    with torch.no_grad():
        a = ceca(claims, ev)
        b = ref(claims, ev[:2])
        c = ceca(claims, torch.zeros(5, 64))
    assert torch.allclose(a, b, atol=1e-5), "masked CECA != CECA on valid rows"
    assert torch.isfinite(c).all() and c.abs().sum() == 0
    print("  [ok] CECA padding mask equals valid-rows-only; all-zero evidence finite")


def test_infonce_mask():
    torch.manual_seed(0)
    loss = InfoNCELoss(temperature=0.05)
    hc, he = torch.randn(6, 16, requires_grad=True), torch.randn(6, 16)
    m = torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.bool)
    assert torch.allclose(loss(hc, he, valid_mask=m), loss(hc[m], he[m]))
    assert torch.allclose(loss(hc, he), loss(hc, he, valid_mask=None))
    z = loss(hc, he, valid_mask=torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.bool))
    z.backward()
    assert z.item() == 0.0 and torch.isfinite(hc.grad).all()
    print("  [ok] InfoNCE valid_mask selects the subset; <2 valid -> 0")


def test_gat_self_loop():
    torch.manual_seed(0)
    x, adj, mask = torch.randn(1, 32), torch.ones(1, 1), torch.ones(1, dtype=torch.bool)
    on = ClaimGAT(in_dim=32, hidden_dim=64, out_dim=16, heads=4, dropout=0.3, keep_self_loop=True).train()
    off = ClaimGAT(in_dim=32, hidden_dim=64, out_dim=16, heads=4, dropout=0.3, keep_self_loop=False).train()
    zeros_on = sum(int(on(x, adj, mask).abs().sum() == 0) for _ in range(200))
    zeros_off = sum(int(off(x, adj, mask).abs().sum() == 0) for _ in range(200))
    assert zeros_on == 0 and zeros_off > 0, (zeros_on, zeros_off)
    print(f"  [ok] GAT keep_self_loop: zero outputs {zeros_off}/200 -> {zeros_on}/200")


def test_no_grad_explosion():
    cfg = load_cfg("FakeTT", **E1_ON)
    b = next(iter(make_loader(cfg, "train")))
    m = built_model(cfg, b).train()
    m.zero_grad()
    out = forward(m, b, cfg, DEV)
    loss = m.compute_loss(out["pred"], b["labels"].to(DEV), out["contrastive"])
    loss.backward()
    g = sum(p.grad.norm() ** 2 for p in m.claim_gat.parameters() if p.grad is not None) ** 0.5
    assert torch.isfinite(loss) and g < 1e3, f"loss={loss.item()} gat_grad={g}"
    print(f"  [ok] FakeTT train batch, E1 on: loss={loss.item():.4f}, ClaimGAT grad={g:.3g}")


if __name__ == "__main__":
    for fn in (test_ceca_mask, test_infonce_mask, test_gat_self_loop, test_no_grad_explosion, test_backward_compat):
        print(fn.__name__)
        fn()
    import shutil
    shutil.rmtree(AUDIT / "_test_logdir", ignore_errors=True)
    print("ALL PASSED")
