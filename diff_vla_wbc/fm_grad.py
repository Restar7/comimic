"""Gradient-estimation backends for the flow-matching action head.

``diff_vla_wbc.gr00t_fm.sample_fm_from_features`` unrolls the Euler ODE and lets
autograd do the rest — that is the **pathwise** (exact BPTT) estimator.  The
joint trainer pays for it in activation memory: every one of the ``K`` DiT
forward passes keeps its activations alive so the backward can traverse them.

This module makes the other estimators switchable inside the *real* training
loop (they already existed for the toy/tiny harness in ``ablation_fm_grad``):

======================  ===========================================  ==============  =========
backend                 gradient                                     act. memory     cost
======================  ===========================================  ==============  =========
``pathwise``            exact dL/dtheta through the unrolled ODE      O(K)            1x
``checkpointed``        exact (recompute per Euler step)              O(1) + recompute ~2x
``truncated``           biased: only the last ``tail`` steps get grad  O(tail)        1x
``adjoint``             discrete adjoint, math. equal to pathwise     O(K) states     ~2x
``qvgm_lite``           NOT available here (needs a chunk critic)     -               -
======================  ===========================================  ==============  =========

Why ``qvgm_lite`` is missing: the local-approximation estimator needs a chunk
critic ``Q(s, a)``.  The toy harness can use an oracle critic; the real joint
trainer has no trained critic yet, so shipping it here would silently train on a
garbage value signal.  Use ``ablation_fm_grad`` for that backend.

Correctness contract: with the same weights and the same initial noise,
``pathwise`` / ``checkpointed`` / ``adjoint`` must agree to floating-point
(``tests/test_fm_grad_backends.py``), and ``truncated`` must agree in direction
(cosine) while being measurably biased in relative error.
"""
from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import nn

StepFn = Callable[[torch.Tensor, int], torch.Tensor]

BACKENDS = ("pathwise", "checkpointed", "truncated", "adjoint")

BACKEND_DOC = {
    "pathwise": "unrolled Euler + autograd (exact BPTT, O(K) activations)",
    "checkpointed": "per-step torch.utils.checkpoint (exact, ~2x time, O(1) activations)",
    "truncated": "first K-tail steps under no_grad (biased, cheap, warm-start style)",
    "adjoint": "discrete adjoint / vjp replay (exact, ~2x time, O(K) tiny states)",
}


def resolve_backend(name: str) -> str:
    key = (name or "pathwise").strip().lower().replace("-", "_")
    if key == "qvgm_lite" or key == "qvgm":
        raise SystemExit(
            "qvgm_lite needs a chunk critic and is not wired into the real joint trainer; "
            "run `python -m ablation_fm_grad.run_ablation` for that backend instead."
        )
    if key not in BACKENDS:
        raise SystemExit(f"unknown --fm-grad-backend {name!r}; choose from {BACKENDS}")
    return key


def _parameters_of(step_fn: StepFn) -> list[torch.Tensor]:
    """Parameters reachable from the closure's ``__self__``/cells (best effort)."""
    seen: dict[int, torch.Tensor] = {}
    owner = getattr(step_fn, "__self__", None)
    if isinstance(owner, nn.Module):
        for p in owner.parameters():
            seen[id(p)] = p
    for cell in (getattr(step_fn, "__closure__", None) or ()):
        try:
            value = cell.cell_contents
        except ValueError:  # empty cell
            continue
        if isinstance(value, nn.Module):
            for p in value.parameters():
                seen[id(p)] = p
        elif isinstance(value, torch.Tensor) and value.requires_grad:
            seen[id(value)] = value
    return list(seen.values())


class _AdjointSample(torch.autograd.Function):
    """Forward under no_grad; replay the ODE backwards one step at a time.

    The backward loop is the textbook discrete adjoint: for ``k = K-1 .. 0``
    recompute ``a_{k+1} = f(a_k, k)`` with grad enabled, hand it the upstream
    gradient, and accumulate both ``dL/da_k`` and this step's parameter gradient.
    Summing the per-step parameter gradients over the loop equals the full
    unrolled gradient (to fp error) while never holding more than one Euler step
    of activations at a time.
    """

    @staticmethod
    def forward(ctx, a0: torch.Tensor, steps: int, step_fn: StepFn, params: list[torch.Tensor]):
        states = [a0.detach()]
        with torch.no_grad():
            a = states[0]
            for k in range(steps):
                a = step_fn(a, k)
                states.append(a)
        ctx.states = states
        ctx.steps = steps
        ctx.step_fn = step_fn
        ctx.params = params
        return states[-1]

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        states, step_fn, params = ctx.states, ctx.step_fn, ctx.params
        upstream = grad_out
        # Grad mode is OFF while a Function's backward runs; re-enable it so the
        # replayed Euler steps build a real graph we can take vjps of.
        with torch.enable_grad():
            for k in reversed(range(ctx.steps)):
                a_prev = states[k].detach().requires_grad_(True)
                a_next = step_fn(a_prev, k)
                grads = torch.autograd.grad(
                    a_next,
                    [a_prev] + params,
                    grad_outputs=upstream,
                    retain_graph=False,
                    allow_unused=True,
                )
                upstream = grads[0]
                for i, g in enumerate(grads[1:]):
                    if g is None:
                        continue
                    if params[i].grad is None:
                        params[i].grad = g.detach().clone()
                    else:
                        params[i].grad.add_(g)
        return None, None, None, None


def sample_with_backend(
    step_fn: StepFn,
    initial: torch.Tensor,
    steps: int,
    *,
    backend: str = "pathwise",
    params: Iterable[torch.Tensor] | None = None,
    truncated_tail: int = 1,
) -> torch.Tensor:
    """Run ``steps`` Euler steps of ``step_fn`` with the requested gradient policy."""
    backend = resolve_backend(backend)
    if steps <= 0:
        raise ValueError("steps must be positive")

    if backend == "pathwise":
        a = initial
        for k in range(steps):
            a = step_fn(a, k)
        return a

    if backend == "checkpointed":
        from torch.utils.checkpoint import checkpoint

        a = initial
        for k in range(steps):
            a = checkpoint(step_fn, a, k, use_reentrant=False)
        return a

    if backend == "truncated":
        tail = max(1, min(int(truncated_tail), steps))
        a = initial
        with torch.no_grad():
            for k in range(steps - tail):
                a = step_fn(a, k)
        a = a.detach().requires_grad_(True)
        for k in range(steps - tail, steps):
            a = step_fn(a, k)
        return a

    if backend == "adjoint":
        plist = list(params) if params is not None else _parameters_of(step_fn)
        plist = [p for p in plist if p.requires_grad]
        if not plist:
            raise RuntimeError(
                "adjoint backend found no trainable parameters; pass `params=` explicitly"
            )
        # autograd only attaches a node to tensor *inputs* that require grad, so the
        # seed is cloned into a grad-requiring leaf.  Its own gradient is unused
        # (we never differentiate w.r.t. the initial noise) and returned as None.
        seed = initial.detach().clone().requires_grad_(True)
        return _AdjointSample.apply(seed, steps, step_fn, plist)

    raise AssertionError(backend)  # pragma: no cover


__all__ = ["BACKENDS", "BACKEND_DOC", "resolve_backend", "sample_with_backend"]
