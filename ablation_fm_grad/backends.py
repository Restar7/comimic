"""Flow-matching gradient backends for the CoMimic execution-feedback ablation.

Every backend samples the same Euler ODE

    a_{k+1} = a_k + dt * v_theta(a_k, t_k),    dt = 1/K,  t_k = k/K

and differentiates the same downstream loss ``loss_fn(a_K)``.  They differ only
in *how* the gradient reaches ``theta``:

==================  =========================================================
pathwise            full autograd graph through all K Euler steps (BPTT)
checkpointed        same math, per-step ``torch.utils.checkpoint`` (recompute)
truncated           stop-gradient before the last ``tail`` steps (cheap BC)
adjoint             discrete adjoint: no graph, explicit reverse recursion
qvgm_lite           local velocity-matching from a critic value gradient
==================  =========================================================

``qvgm_lite`` is a reimplementation of the *local approximation* described in
Q-VGM (Wang et al., arXiv:2606.08015), not the official code: the official
method trains an action-sensitive chunk critic with IQL/TD; here the critic is
pre-trained by regression on the same task loss so that the gradient mechanism
can be isolated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


LossFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class BackendResult:
    name: str
    loss: float
    grad_norm: float
    grads: dict[str, torch.Tensor] = dc_field(default_factory=dict)
    peak_mem_mb: float = 0.0
    time_s: float = 0.0
    notes: str = ""


def _zero_grads(params: Iterable[torch.Tensor]) -> None:
    for p in params:
        p.grad = None


def _collect_grads(field: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for name, p in field.named_parameters()
        if p.requires_grad
    }


def _grad_norm(grads: dict[str, torch.Tensor]) -> float:
    total = torch.zeros((), device=next(iter(grads.values())).device, dtype=torch.float64)
    for g in grads.values():
        total = total + g.double().pow(2).sum()
    return float(total.sqrt())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_mem_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


class SamplingBackend:
    name = "base"

    def __init__(self, **kwargs) -> None:
        self.options = kwargs

    # ------------------------------------------------------------------ api
    def run(
        self,
        field: nn.Module,
        a0: torch.Tensor,
        steps: int,
        loss_fn: LossFn,
        *,
        critic: nn.Module | None = None,
    ) -> BackendResult:
        raise NotImplementedError

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _measure(fn, device: torch.device):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        dt = time.perf_counter() - t0
        return out, _peak_mem_mb(device), dt

    @staticmethod
    def euler(field: nn.Module, a: torch.Tensor, steps: int, dt: float) -> torch.Tensor:
        for k in range(steps):
            a = a + dt * field(a, k)
        return a


class PathwiseBackend(SamplingBackend):
    """Current prototype: unrolled Euler + autograd (pathwise / BPTT)."""

    name = "pathwise"

    def run(self, field, a0, steps, loss_fn, *, critic=None) -> BackendResult:
        params = [p for p in field.parameters() if p.requires_grad]
        _zero_grads(params)
        dt = 1.0 / steps
        device = a0.device

        def _fwd():
            a = self.euler(field, a0, steps, dt)
            loss = loss_fn(a)
            loss.backward()
            return loss

        loss, mem, dt_s = self._measure(_fwd, device)
        grads = _collect_grads(field)
        return BackendResult(
            name=self.name,
            loss=float(loss.detach()),
            grad_norm=_grad_norm(grads),
            grads=grads,
            peak_mem_mb=mem,
            time_s=dt_s,
            notes=f"K={steps} graph retained",
        )


class CheckpointedBackend(SamplingBackend):
    """Per-step activation checkpointing: same gradients, less activation memory."""

    name = "checkpointed"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def run(self, field, a0, steps, loss_fn, *, critic=None) -> BackendResult:
        from torch.utils.checkpoint import checkpoint

        params = [p for p in field.parameters() if p.requires_grad]
        _zero_grads(params)
        dt = 1.0 / steps
        device = a0.device

        def step(a: torch.Tensor, k: int) -> torch.Tensor:
            return a + dt * field(a, k)

        def _fwd():
            a = a0
            for k in range(steps):
                a = checkpoint(step, a, k, use_reentrant=False)
            loss = loss_fn(a)
            loss.backward()
            return loss

        loss, mem, dt_s = self._measure(_fwd, device)
        grads = _collect_grads(field)
        return BackendResult(
            name=self.name,
            loss=float(loss.detach()),
            grad_norm=_grad_norm(grads),
            grads=grads,
            peak_mem_mb=mem,
            time_s=dt_s,
            notes=f"K={steps} recompute per step",
        )


class TruncatedBackend(SamplingBackend):
    """Stop-gradient before the last ``tail`` Euler steps (a cheap warm-start)."""

    name = "truncated"

    def __init__(self, tail: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tail = int(tail)

    def run(self, field, a0, steps, loss_fn, *, critic=None) -> BackendResult:
        params = [p for p in field.parameters() if p.requires_grad]
        _zero_grads(params)
        dt = 1.0 / steps
        device = a0.device
        tail = min(self.tail, steps)

        def _fwd():
            with torch.no_grad():
                a = self.euler(field, a0, steps - tail, dt)
            for k in range(steps - tail, steps):
                a = a + dt * field(a, k)
            loss = loss_fn(a)
            loss.backward()
            return loss

        loss, mem, dt_s = self._measure(_fwd, device)
        grads = _collect_grads(field)
        return BackendResult(
            name=self.name,
            loss=float(loss.detach()),
            grad_norm=_grad_norm(grads),
            grads=grads,
            peak_mem_mb=mem,
            time_s=dt_s,
            notes=f"K={steps} no-grad first {steps - tail}, grad last {tail}",
        )


class AdjointBackend(SamplingBackend):
    """Discrete adjoint: forward under no_grad, explicit reverse recursion.

    Forward:  a_{k+1} = a_k + dt v(a_k, k)
    Reverse:  lam_{k} = lam_{k+1} + dt (dv/da_k)^T lam_{k+1}
              grad_theta += dt (dv/dtheta)^T lam_{k+1}

    Only one Euler step graph is alive at a time, so activation memory is O(1)
    in K (we still keep the K forward states, which are tiny compared with the
    MLP activations).  Gradients are mathematically equal to BPTT up to
    floating-point error.
    """

    name = "adjoint"

    def run(self, field, a0, steps, loss_fn, *, critic=None) -> BackendResult:
        params = [p for p in field.parameters() if p.requires_grad]
        _zero_grads(params)
        dt = 1.0 / steps
        device = a0.device

        def _fwd():
            with torch.no_grad():
                states = [a0.detach()]
                for k in range(steps):
                    states.append(states[-1] + dt * field(states[-1], k))
            a_k = states[-1].detach().requires_grad_(True)
            loss = loss_fn(a_k)
            lam = torch.autograd.grad(loss, a_k)[0].detach()
            grads = {name: torch.zeros_like(p) for name, p in field.named_parameters() if p.requires_grad}
            name_by_param = {id(p): name for name, p in field.named_parameters() if p.requires_grad}
            for k in reversed(range(steps)):
                a_state = states[k].detach().requires_grad_(True)
                v = field(a_state, k)
                grad_a, *grad_p = torch.autograd.grad(
                    v, [a_state, *params], grad_outputs=lam * dt, retain_graph=False, allow_unused=True
                )
                for p, gp in zip(params, grad_p):
                    if gp is not None:
                        grads[name_by_param[id(p)]] += gp.detach()
                lam = lam + grad_a.detach()
            for name, p in field.named_parameters():
                if p.requires_grad:
                    p.grad = grads[name]
            return loss

        loss, mem, dt_s = self._measure(_fwd, device)
        grads = _collect_grads(field)
        return BackendResult(
            name=self.name,
            loss=float(loss.detach()),
            grad_norm=_grad_norm(grads),
            grads=grads,
            peak_mem_mb=mem,
            time_s=dt_s,
            notes=f"K={steps} O(1) step graph, K vjps",
        )


class QVGMLiteBackend(SamplingBackend):
    """Q-VGM-style local value-gradient matching (reimplementation).

    At a late denoising step k we form a first-order clean-action estimate
    ``a_hat = a_{k+1} + (1 - t_{k+1}) * v``, take the critic gradient
    ``g = dQ/da_hat``, and regress the local velocity toward
    ``v + (1 - t_{k+1}) * g``.  Gradients touch only the local velocity
    prediction, never the K-step trajectory.

    The critic must be pre-trained; ``harness.train_critic`` does that by
    regression on the same downstream loss.
    """

    name = "qvgm_lite"

    def __init__(self, late_fraction: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.late_fraction = float(late_fraction)

    def run(self, field, a0, steps, loss_fn, *, critic=None) -> BackendResult:
        if critic is None:
            raise ValueError("qvgm_lite needs a pre-trained critic")
        params = [p for p in field.parameters() if p.requires_grad]
        _zero_grads(params)
        dt = 1.0 / steps
        device = a0.device
        k_start = int(steps * (1.0 - self.late_fraction))
        k_start = max(0, min(k_start, steps - 1))

        def _fwd():
            with torch.no_grad():
                a = self.euler(field, a0, k_start, dt)
            k = min(steps - 1, k_start)
            t_k = k / steps
            v = field(a, k)
            # First-order clean-action estimate from a_k; gradient touches only
            # this local velocity prediction, never the K-step trajectory.
            a_hat = a + (1.0 - t_k) * v
            q = critic(a_hat).mean()
            loss = -q  # ascend the critic value through the local velocity
            loss.backward()
            return loss

        loss, mem, dt_s = self._measure(_fwd, device)
        grads = _collect_grads(field)
        return BackendResult(
            name=self.name,
            loss=float(loss.detach()),
            grad_norm=_grad_norm(grads),
            grads=grads,
            peak_mem_mb=mem,
            time_s=dt_s,
            notes=f"K={steps} one local velocity step (k={k_start})",
        )


BACKENDS: dict[str, type[SamplingBackend]] = {
    "pathwise": PathwiseBackend,
    "checkpointed": CheckpointedBackend,
    "truncated": TruncatedBackend,
    "adjoint": AdjointBackend,
    "qvgm_lite": QVGMLiteBackend,
}


def make_backend(name: str, **kwargs) -> SamplingBackend:
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)
