"""Shared harness for the flow-matching gradient ablation.

Two modes:

* ``toy``  — a small MLP velocity field + quadratic execution loss.  Cheap,
  deterministic, and isolates the *gradient estimator* from model engineering.
* ``tiny`` — the real GR00T N1.7 action-head class with tiny dims
  (``diff_vla_wbc.gr00t_tiny``), so the same backends run against the actual
  ``action_encoder -> DiT -> action_decoder`` code path.

The downstream loss is intentionally a differentiable surrogate; swapping in
the SuperMimic rollout is a separate step (see ``README.md``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backends import BackendResult, SamplingBackend, _grad_norm, make_backend


# ---------------------------------------------------------------------------
# toy model
# ---------------------------------------------------------------------------
class TinyVelocity(nn.Module):
    """v_theta(a, k): MLP over [a, condition, k/K]."""

    def __init__(self, dim: int = 64, cond_dim: int = 16, hidden: int = 128, total_steps: int = 8):
        super().__init__()
        self.dim = dim
        self.total_steps = total_steps
        self.cond = nn.Parameter(torch.randn(cond_dim) * 0.5)
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, a: torch.Tensor, k: int) -> torch.Tensor:
        batch = a.shape[0]
        t = torch.full((batch, 1), k / max(self.total_steps, 1), device=a.device, dtype=a.dtype)
        cond = self.cond.unsqueeze(0).expand(batch, -1)
        return self.net(torch.cat([a, cond, t], dim=-1))


class QuadraticTask:
    """Frozen random projection + target: a cheap differentiable execution loss."""

    def __init__(self, dim: int, out_dim: int = 8, seed: int = 0, device: str | torch.device = "cpu"):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.W = torch.randn(dim, out_dim, generator=g).to(device)
        self.target = torch.randn(out_dim, generator=g).to(device)

    def per_sample(self, a: torch.Tensor) -> torch.Tensor:
        flat = a.reshape(a.shape[0], -1)
        pred = flat @ self.W
        mse = ((pred - self.target) ** 2).mean(dim=-1)
        reg = 1e-3 * flat.pow(2).mean(dim=-1)
        return mse + reg

    def __call__(self, a: torch.Tensor) -> torch.Tensor:
        return self.per_sample(a).mean()


class OracleCritic(nn.Module):
    """Q(a) = -task_loss(a): the exact value, used to validate the Q-VGM mechanism."""

    def __init__(self, task):
        super().__init__()
        self.task = task

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return -self.task.per_sample(a)


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
@dataclass
class Setup:
    field: nn.Module
    task: QuadraticTask
    a0: torch.Tensor
    dim: int
    device: torch.device


def make_toy_setup(steps: int, device: torch.device, batch: int = 8, dim: int = 64, seed: int = 0) -> Setup:
    torch.manual_seed(seed)
    field = TinyVelocity(dim=dim, total_steps=steps).to(device)
    task = QuadraticTask(dim, seed=seed, device=device)
    a0 = torch.randn(batch, dim, device=device)
    return Setup(field=field, task=task, a0=a0, dim=dim, device=device)


def make_tiny_gr00t_setup(
    steps: int,
    device: torch.device,
    batch: int = 4,
    horizon: int = 8,
    action_dim: int = 64,
    seed: int = 0,
    hidden: int = 32,
    layers: int = 1,
    heads: int = 2,
) -> Setup:
    """Real GR00T action-head class with tiny dims (fast, real code path)."""
    from diff_vla_wbc.gr00t_tiny import TinyGr00tModel, make_tiny_action_head, tiny_model_inputs

    torch.manual_seed(seed)
    head = make_tiny_action_head(
        action_horizon=horizon,
        action_dim=action_dim,
        backbone_dim=16,
        state_dim=8,
        hidden=hidden,
        num_layers=layers,
        num_attention_heads=heads,
    ).to(device)
    model = TinyGr00tModel(head, backbone_dim=16).to(device)
    inputs = tiny_model_inputs(batch, seq_len=5, device=device)
    backbone_inputs, action_inputs = model.prepare_input(inputs)
    with torch.no_grad():
        backbone_out = model.backbone(backbone_inputs)
        feats = head._encode_features(backbone_out, action_inputs)
        emb_id = action_inputs.embodiment_id

    buckets = int(getattr(head, "num_timestep_buckets", 1000))

    class Gr00tVelocity(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = head
            self.feats = feats
            self.emb_id = emb_id
            self.horizon = horizon
            self.total_steps = steps
            self.buckets = buckets
            self.add_pos = bool(getattr(head.config, "add_pos_embed", False))

        def forward(self, a: torch.Tensor, k: int) -> torch.Tensor:
            b = a.shape[0]
            timestep = int(k / max(self.total_steps, 1) * self.buckets)
            timesteps = torch.full((b,), timestep, dtype=torch.long, device=a.device)
            emb = self.head.action_encoder(a, timesteps, self.emb_id)
            if self.add_pos:
                pos = torch.arange(emb.shape[1], dtype=torch.long, device=a.device)
                emb = emb + self.head.position_embedding(pos).unsqueeze(0)
            joined = torch.cat((self.feats.state_features, emb), dim=1)
            hidden = self.head.model(
                hidden_states=joined,
                encoder_hidden_states=self.feats.backbone_features,
                timestep=timesteps,
            )
            pred = self.head.action_decoder(hidden, self.emb_id)
            return pred[:, -self.horizon :]

    field = Gr00tVelocity().to(device)
    task = QuadraticTask(horizon * action_dim, seed=seed, device=device)
    a0 = torch.randn(batch, horizon, action_dim, device=device)
    return Setup(field=field, task=task, a0=a0, dim=horizon * action_dim, device=device)


# ---------------------------------------------------------------------------
# critic pre-training for the Q-VGM backend
# ---------------------------------------------------------------------------
def train_critic(setup: Setup, steps: int, backend: SamplingBackend, iters: int = 300, lr: float = 1e-3) -> Critic:
    """Regress critic(a_K) -> -task_loss(a_K) on pathwise samples."""
    critic = Critic(setup.dim).to(setup.device)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    # a_0 must be grad-free; collect states with an unrolled no-grad sample
    field = setup.field
    dt = 1.0 / steps
    for _ in range(iters):
        # train on late-step states: sample k, roll to a_k, continue to a_K
        with torch.no_grad():
            k = int(torch.randint(max(1, steps // 2), steps + 1, (1,)).item())
            a = setup.a0
            for j in range(k):
                a = a + dt * field(a, j)
            a_k = a
            for j in range(k, steps):
                a = a + dt * field(a, j)
            target = -setup.task.per_sample(a).detach()
        pred = critic(a_k)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    critic.eval()
    return critic


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def grad_agreement(ref: dict[str, torch.Tensor], other: dict[str, torch.Tensor]) -> tuple[float, float]:
    """Return (cosine, relative L2 error) between two gradient dictionaries."""
    names = sorted(ref)
    r = torch.cat([ref[n].reshape(-1) for n in names]).double()
    o = torch.cat([other[n].reshape(-1) for n in names]).double()
    denom = (r.norm() * o.norm()).clamp_min(1e-12)
    cos = float((r @ o) / denom)
    rel = float((o - r).norm() / r.norm().clamp_min(1e-12))
    return cos, rel


def _apply_grads(field: nn.Module, grads: dict[str, torch.Tensor]) -> None:
    for name, p in field.named_parameters():
        if p.requires_grad:
            p.grad = grads[name].detach().clone() if name in grads else torch.zeros_like(p)


@dataclass
class Row:
    mode: str
    backend: str
    steps: int
    loss: float
    grad_norm: float
    cosine: float
    rel_err: float
    peak_mem_mb: float
    time_s: float
    train_loss_before: float
    train_loss_after: float
    notes: str = ""


def run_backend_once(
    setup: Setup, backend_name: str, steps: int, critic: nn.Module | None = None, **backend_kwargs
) -> BackendResult:
    backend = make_backend(backend_name, **backend_kwargs)
    return backend.run(setup.field, setup.a0, steps, setup.task, critic=critic)


def evaluate(
    *,
    mode: str,
    steps_list: list[int],
    backends: list[str],
    trials: int = 3,
    train_steps: int = 30,
    lr: float = 1e-3,
    device: str = "cuda",
    seed: int = 0,
    qvgm_critic: str = "oracle",
    scale_preset: str = "small",
) -> list[Row]:
    dev = torch.device(device)
    rows: list[Row] = []
    for steps in steps_list:
        if mode == "toy":
            setup = make_toy_setup(steps, dev, seed=seed)
        elif mode == "tiny":
            setup = make_tiny_gr00t_setup(steps, dev, seed=seed)
        elif mode == "scale":
            if scale_preset == "large":
                setup = make_tiny_gr00t_setup(
                    steps, dev, seed=seed, batch=4, horizon=40, hidden=512, layers=8, heads=8
                )
            else:
                setup = make_tiny_gr00t_setup(
                    steps, dev, seed=seed, batch=2, horizon=40, hidden=256, layers=4, heads=8
                )
        else:
            raise ValueError(f"unknown mode {mode!r}")
        # reference gradients from pathwise (the exact estimator)
        ref_backend = make_backend("pathwise")
        ref = ref_backend.run(setup.field, setup.a0, steps, setup.task)
        critic = None
        if "qvgm_lite" in backends:
            if qvgm_critic == "oracle":
                critic = OracleCritic(setup.task).to(dev)
            else:
                critic = train_critic(setup, steps, ref_backend, iters=300 if mode == "toy" else 100)
        for name in backends:
            # --- repeated timing / memory (no parameter updates) ---
            best_loss, best_mem, times, grads = math.inf, 0.0, [], None
            for _ in range(max(1, trials)):
                res = run_backend_once(setup, name, steps, critic=critic)
                best_loss = min(best_loss, res.loss)
                best_mem = max(best_mem, res.peak_mem_mb)
                times.append(res.time_s)
                grads = res.grads
            cos, rel = grad_agreement(ref.grads, grads or {})
            # --- short learning run from the same initial parameters ---
            with torch.no_grad():
                a_init = setup.a0
                for k in range(steps):
                    a_init = a_init + (1.0 / steps) * setup.field(a_init, k)
                before = float(setup.task(a_init).detach())
            train_field = _clone_parameters(setup.field)
            opt = torch.optim.Adam(train_field.parameters(), lr=lr)
            for _ in range(train_steps):
                opt.zero_grad(set_to_none=True)
                res = make_backend(name).run(train_field, setup.a0, steps, setup.task, critic=critic)
                if res.grad_norm > 0:
                    # unused action-head parameters (e.g. pos-embed) have no grad; zero-fill
                    for p in train_field.parameters():
                        if p.requires_grad and p.grad is None:
                            p.grad = torch.zeros_like(p)
                    opt.step()
            with torch.no_grad():
                a = setup.a0
                for k in range(steps):
                    a = a + (1.0 / steps) * train_field(a, k)
                after = float(setup.task(a).detach())
            rows.append(
                Row(
                    mode=mode,
                    backend=name,
                    steps=steps,
                    loss=float(best_loss),
                    grad_norm=_grad_norm(grads) if grads else 0.0,
                    cosine=cos,
                    rel_err=rel,
                    peak_mem_mb=best_mem,
                    time_s=float(sum(times) / len(times)),
                    train_loss_before=before,
                    train_loss_after=after,
                    notes="",
                )
            )
            print(
                f"[{mode}] K={steps:2d} {name:12s} loss={best_loss:.4f} cos={cos:+.4f} "
                f"rel={rel:.3e} mem={best_mem:7.1f}MB t={sum(times)/len(times):.3f}s "
                f"train {before:.3f}->{after:.3f}",
                flush=True,
            )
    return rows


def _clone_parameters(field: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(field)


def rows_to_markdown(rows: list[Row]) -> str:
    lines = [
        "| mode | K | backend | loss | cosine vs pathwise | rel err | peak mem (MB) | time (s) | train loss before → after |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.mode} | {r.steps} | {r.backend} | {r.loss:.4f} | {r.cosine:+.4f} | "
            f"{r.rel_err:.2e} | {r.peak_mem_mb:.1f} | {r.time_s:.3f} | "
            f"{r.train_loss_before:.3f} → {r.train_loss_after:.3f} |"
        )
    return "\n".join(lines)
