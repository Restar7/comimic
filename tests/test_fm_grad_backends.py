"""Gradient-correctness tests for the switchable FM backends.

``pathwise`` is the reference (exact BPTT).  The other exact estimators must
reproduce it to floating-point; ``truncated`` is documented as biased and only
has to point in the same direction.  These are pure-CPU torch tests — they do
not touch GR00T, SuperMimic or any checkpoint.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from diff_vla_wbc.fm_grad import BACKENDS, resolve_backend, sample_with_backend


class Velocity(nn.Module):
    """Small stand-in for the DiT velocity field: v(a, step) = tanh(W_k a + b)."""

    def __init__(self, dim: int = 16, steps: int = 4) -> None:
        super().__init__()
        self.steps = steps
        self.weight = nn.Parameter(torch.randn(steps, dim, dim) * 0.1)
        self.bias = nn.Parameter(torch.randn(steps, dim) * 0.05)
        self.cond = nn.Parameter(torch.randn(dim))

    def forward(self, a: torch.Tensor, k: int) -> torch.Tensor:
        return torch.tanh(a @ self.weight[k].t() + self.bias[k] + 0.1 * self.cond)


def _setup(dim: int = 16, steps: int = 4, batch: int = 3, seed: int = 0, truncated_tail: int = 1):
    torch.manual_seed(seed)
    field = Velocity(dim=dim, steps=steps)
    a0 = torch.randn(batch, dim)
    target = torch.randn(batch, dim)
    dt = 1.0 / steps

    def step_fn(a: torch.Tensor, k: int) -> torch.Tensor:
        return a + dt * field(a, k)

    def grad_of(backend: str) -> torch.Tensor:
        field.zero_grad(set_to_none=True)
        out = sample_with_backend(
            step_fn,
            a0.clone(),
            steps,
            backend=backend,
            params=list(field.parameters()),
            truncated_tail=truncated_tail,
        )
        loss = ((out - target) ** 2).mean()
        loss.backward()
        return torch.cat([p.grad.reshape(-1).clone() for p in field.parameters()])

    return field, grad_of


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def test_backend_names() -> None:
    assert resolve_backend("checkpointed") == "checkpointed"
    assert resolve_backend("adjoint") == "adjoint"
    with pytest.raises(SystemExit):
        resolve_backend("qvgm_lite")
    with pytest.raises(SystemExit):
        resolve_backend("nope")
    assert set(BACKENDS) == {"pathwise", "checkpointed", "truncated", "adjoint"}


@pytest.mark.parametrize("backend", ["checkpointed", "adjoint"])
def test_exact_backends_match_pathwise(backend: str) -> None:
    _, grad_of = _setup()
    ref = grad_of("pathwise")
    got = grad_of(backend)
    assert torch.isfinite(got).all()
    assert _rel(got, ref) < 1e-5, f"{backend} rel error {_rel(got, ref):.2e}"
    assert _cos(got, ref) > 1 - 1e-6


def test_truncated_is_biased_but_aligned() -> None:
    """Truncation keeps the direction for the steps it keeps and drops the rest.

    ``Velocity`` holds one weight matrix per step, so a tail of ``K-1`` means the
    gradient of step 0 is simply missing.  That is the documented bias: high
    cosine, non-zero relative error, and *exactly zero* gradient on the dropped
    step's block.
    """
    steps = 6
    tail = steps - 1
    field, grad_of = _setup(steps=steps, truncated_tail=tail)
    ref = grad_of("pathwise")
    got = grad_of("truncated")

    block = field.weight[0].numel()
    first_block = got[:block]
    assert torch.count_nonzero(first_block) == 0, "step-0 parameters must get no gradient"
    assert _cos(got, ref) > 0.9, f"truncated direction off: cos={_cos(got, ref):.3f}"
    assert 1e-6 < _rel(got, ref) < 1.0


def test_truncated_with_full_tail_equals_pathwise() -> None:
    steps = 4
    _, grad_of = _setup(steps=steps, truncated_tail=steps)
    assert _rel(grad_of("truncated"), grad_of("pathwise")) < 1e-5


def test_adjoint_matches_pathwise_on_output() -> None:
    """Forward pass must be bit-identical (same Euler iterate, no grad needed)."""
    field, _ = _setup()
    dt = 1.0 / field.steps
    a0 = torch.randn(2, 16)

    def step_fn(a, k):
        return a + dt * field(a, k)

    with torch.no_grad():
        outs = {
            b: sample_with_backend(step_fn, a0.clone(), field.steps, backend=b,
                                   params=list(field.parameters()))
            for b in ("pathwise", "checkpointed", "truncated", "adjoint")
        }
    ref = outs["pathwise"]
    for name, out in outs.items():
        assert torch.allclose(out, ref, atol=1e-6), f"{name} forward drifted"


def test_truncated_tail_bounds() -> None:
    field, _ = _setup()
    dt = 1.0 / field.steps

    def step_fn(a, k):
        return a + dt * field(a, k)

    # tail larger than K is clamped instead of raising
    out = sample_with_backend(step_fn, torch.randn(2, 16), field.steps, backend="truncated", truncated_tail=99)
    assert out.shape == (2, 16)


def test_adjoint_requires_parameters() -> None:
    def step_fn(a, k):
        return a * 0.5

    with pytest.raises(RuntimeError):
        sample_with_backend(step_fn, torch.randn(2, 4), 3, backend="adjoint", params=[])


# --------------------------------------------------------------------------- integration
# The tests above pin the math on a synthetic field.  These run the real
# ``sample_model_with_grad`` (tiny GR00T action head + tiny backbone, CPU) so a
# wiring mistake — params not forwarded, backend name dropped — cannot pass.
import sys as _sys
from pathlib import Path as _Path

_DEPS = _Path(__file__).resolve().parents[1] / "third_party" / "gr00t_deps"
if _DEPS.is_dir() and str(_DEPS) not in _sys.path:
    _sys.path.insert(0, str(_DEPS))


def _tiny_stack():
    pytest.importorskip("tyro", reason="GR00T action head needs third_party/gr00t_deps on sys.path")
    from diff_vla_wbc.gr00t_fm import sample_model_with_grad
    from diff_vla_wbc.gr00t_tiny import TinyGr00tModel, make_tiny_action_head, tiny_model_inputs

    torch.manual_seed(0)
    model = TinyGr00tModel(make_tiny_action_head())
    head_params = [p for p in model.action_head.parameters() if p.requires_grad]
    inputs = tiny_model_inputs(2)
    noise = torch.randn(2, 12, 64)
    return sample_model_with_grad, model, head_params, inputs, noise


def _head_grad(sample_model_with_grad, model, head_params, inputs, noise, backend: str) -> torch.Tensor:
    for p in model.parameters():
        p.grad = None
    tokens = sample_model_with_grad(
        model, inputs, initial_noise=noise, num_steps=3, grad_backend=backend
    )
    tokens.square().mean().backward()
    return torch.cat(
        [(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in head_params]
    )


@pytest.mark.parametrize("backend", ["checkpointed", "adjoint"])
def test_tiny_gr00t_head_grad_matches_pathwise(backend: str) -> None:
    try:
        sample_model_with_grad, model, head_params, inputs, noise = _tiny_stack()
    except pytest.skip.Exception:
        raise
    ref = _head_grad(sample_model_with_grad, model, head_params, inputs, noise, "pathwise")
    got = _head_grad(sample_model_with_grad, model, head_params, inputs, noise, backend)
    assert _rel(got, ref) < 1e-5, f"{backend} rel={_rel(got, ref):.2e}"
    assert _cos(got, ref) > 1 - 1e-6


def test_tiny_gr00t_truncated_is_biased() -> None:
    sample_model_with_grad, model, head_params, inputs, noise = _tiny_stack()
    ref = _head_grad(sample_model_with_grad, model, head_params, inputs, noise, "pathwise")
    got = _head_grad(sample_model_with_grad, model, head_params, inputs, noise, "truncated")
    assert _cos(got, ref) > 0.99
    assert _rel(got, ref) > 1e-2


def test_unknown_backend_rejected_before_sampling() -> None:
    sample_model_with_grad, model, _, inputs, noise = _tiny_stack()
    with pytest.raises(SystemExit):
        sample_model_with_grad(model, inputs, initial_noise=noise, num_steps=2, grad_backend="bogus")
