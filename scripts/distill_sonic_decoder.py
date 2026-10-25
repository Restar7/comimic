#!/usr/bin/env python3
"""Distill the official SONIC release decoder from recorded deploy data.

Why: HA raw records, per frame, ``decoder_obs (T,994)`` and the decoder's own output
``decoder_raw_action (T,29)`` (plus ``decoder_target_action``).  The release weights
(``model_decoder.onnx``) are not in this workspace, but the *function* is fully captured by
634k recorded pairs -- so a 994 -> 29 MLP can recover it.  This is the executable path to the
**official** closed loop (VLA token -> decoder -> action), which is not the same controller as
the Isaac SuperMimic WBC (1054 inputs) we have been using.

Acceptance: held-out mean |a_pred - a_recorded| must be small vs the recorded action scale
(|a_raw| ~ 0.85), and the per-dim correlation high.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=120, help="0 = all raw npz")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/sonic_decoder_distilled.pt")
    ap.add_argument("--target-key", default="final_body_action_29dof",
                    help="蒸馏目标：final_body_action_29dof（实际下发动作）或 decoder_raw_action")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    files = sorted(glob.glob(str(ROOT / "data/ha_raw/**/*.npz"), recursive=True))
    if a.files:
        files = files[: a.files]
    xs, ys, n_files = [], [], 0
    for p in files:
        try:
            z = np.load(p, allow_pickle=True)
            if "decoder_obs" not in z.files or a.target_key not in z.files:
                continue
            xs.append(np.asarray(z["decoder_obs"], dtype=np.float32))
            ys.append(np.asarray(z[a.target_key], dtype=np.float32))
            n_files += 1
        except Exception:
            continue
    if not xs:
        raise SystemExit("no decoder records found")
    split = max(1, int(0.9 * len(xs)))
    Xtr = torch.from_numpy(np.concatenate(xs[:split])); Ytr = torch.from_numpy(np.concatenate(ys[:split]))
    Xva = torch.from_numpy(np.concatenate(xs[split:])); Yva = torch.from_numpy(np.concatenate(ys[split:]))
    print(f"[data] files={n_files} train={tuple(Xtr.shape)} val={tuple(Xva.shape)} "
          f"|a|={float(Ytr.abs().mean()):.3f}")

    dev = torch.device(a.device)
    model = nn.Sequential(
        nn.Linear(994, a.hidden), nn.SiLU(),
        nn.Linear(a.hidden, a.hidden), nn.SiLU(),
        nn.Linear(a.hidden, 29),
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    losses = nn.MSELoss()

    @torch.no_grad()
    def evaluate():
        model.eval()
        errs, corrs = [], []
        for i in range(0, Xva.shape[0], 8192):
            xb, yb = Xva[i:i+8192].to(dev), Yva[i:i+8192].to(dev)
            pred = model(xb)
            errs.append(float((pred - yb).abs().mean()))
            a_, b_ = pred.flatten(), yb.flatten()
            corrs.append(float(((a_-a_.mean())*(b_-b_.mean())).mean() / (a_.std()*b_.std()+1e-9)))
        model.train()
        return float(np.mean(errs)), float(np.mean(corrs))

    for it in range(1, a.steps + 1):
        idx = torch.randint(0, Xtr.shape[0], (a.batch,))
        xb, yb = Xtr[idx].to(dev), Ytr[idx].to(dev)
        loss = losses(model(xb), yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it % max(1, a.steps // 5) == 0:
            mae, corr = evaluate()
            print(f"  [{it:>4}] train_mse={float(loss):.4f}  val_mae={mae:.4f}  val_corr={corr:.3f}", flush=True)
    mae, corr = evaluate()
    print(f"[final] val_mae={mae:.4f} (recorded |a|={float(Yva.abs().mean()):.3f})  val_corr={corr:.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden": a.hidden,
                "in_dim": 994, "out_dim": 29, "val_mae": mae, "val_corr": corr,
                "source": f"HA raw decoder_obs -> {a.target_key}", "target_key": a.target_key}, a.out)
    print(f"[saved] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
